from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from transformers.integrations.integration_utils import is_wandb_available

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from aomp_opsd.data_collator import AOMPOPSDDataCollator
from aomp_opsd.mrosi_compat import ensure_mrosi_on_path, mrosi_root
from aomp_opsd.trainer import AOMPOPSDTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
ensure_mrosi_on_path()


def _first_nonempty(example, candidates):
    for name in candidates:
        value = example.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def add_sft_text_column(dataset):
    def add_text(example):
        problem = _first_nonempty(example, ("problem", "Question", "question", "prompt"))
        return {"text": f"Problem: {problem}"}

    return dataset.map(add_text, desc="Normalizing SFT compatibility text column")


@dataclass
class AOMPOPSDScriptArguments(ScriptArguments):
    run_config: str | None = field(default=None, metadata={"help": "Run name suffix."})
    presence_penalty: float = field(default=0.0, metadata={"help": "Sampling presence penalty for rollouts."})
    fixed_teacher: bool = field(
        default=True,
        metadata={"help": "Use the frozen base model as masked-route teacher by disabling LoRA adapters."},
    )
    top_k_loss: int = field(default=200, metadata={"help": "Teacher top-k vocabulary entries for OPSD/JSD."})
    jsd_token_clip: float = field(default=0.05, metadata={"help": "Per-token JSD clipping value."})
    student_thinking: bool = field(default=False, metadata={"help": "Enable thinking mode in student rollouts."})
    teacher_thinking: bool = field(default=True, metadata={"help": "Enable thinking mode in masked-teacher prompts."})
    dataset_name_or_path: str = field(
        default_factory=lambda: str(mrosi_root() / "openthoughts_math_30k_masked.jsonl"),
        metadata={"help": "HF dataset name or local JSON/JSONL with problem, solution, masked_derivation."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split for HF datasets."})
    masked_derivation_column: str = field(default="masked_derivation")
    skip_missing_masked_derivation: bool = field(default=True)
    mrosi_self_imitation_weight: float = field(
        default=0.0,
        metadata={"help": "Legacy MRO-SI self-imitation weight. Default 0 for AOMP-OPSD."},
    )
    mrosi_self_imitation_start_step: int = field(default=75)
    mrosi_self_imitation_start_ratio: float = field(default=-1.0)
    mrosi_self_imitation_token_scope: str = field(default="all")
    mrosi_self_imitation_tail_tokens: int = field(default=256)

    aomp_opsd_enabled: bool = field(default=True)
    variant: str = field(default="full_aomp_opsd")
    step_size: float = field(default=1.0)
    lookahead_lr: float = field(default=1.0e-5)
    group_size: int = field(default=4)
    audit_source: str = field(default="auto")
    audit_temperature: float = field(default=1.0)
    positive_path_weight: float = field(default=1.0)
    negative_path_weight: float = field(default=1.0)
    teacher_kl_weight: float = field(default=1.0)
    self_distill_weight: float = field(default=1.0)
    token_weight_normalization: str = field(default="sequence_sum")
    use_teacher_reliability: bool = field(default=True)
    use_student_uncertainty: bool = field(default=True)
    use_prefix_reliability: bool = field(default=True)
    prefix_decay_lambda: float = field(default=1.0)
    max_token_weight: float = field(default=5.0)
    min_token_weight: float = field(default=0.0)
    audit_budget: int = field(default=0)
    audit_start_step: int = field(
        default=0,
        metadata={"help": "Global step before which audit is skipped and proxy OPSD training is used."},
    )
    audit_every_n_steps: int = field(default=1)
    vllm_approx_lookahead: bool = field(
        default=False,
        metadata={
            "help": (
                "When --use_vllm is enabled, skip temporary exact-lookahead rollout "
                "and audit the current-policy vLLM rollout instead."
            )
        },
    )
    log_diagnostics: bool = field(default=True)
    advantage_clip_range: float = field(default=5.0)
    compute_grad_cosine: bool = field(
        default=False,
        metadata={"help": "Reserved for future hint/lookahead cosine diagnostics."},
    )


def _resolve_torch_dtype(model_args):
    import torch

    value = getattr(model_args, "torch_dtype", None)
    if value is None:
        value = getattr(model_args, "dtype", None)
    if value is None:
        return torch.bfloat16
    if not isinstance(value, str):
        return value
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(value.lower(), torch.bfloat16)


def _load_train_dataset(name_or_path: str, split: str):
    if os.path.isfile(name_or_path):
        return load_dataset("json", data_files=name_or_path)["train"]
    return load_dataset(name_or_path)[split]


def _validate_args(script_args, training_args, model_args) -> None:
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires --use_peft.")
    if script_args.mrosi_self_imitation_token_scope not in {"all", "tail"}:
        raise ValueError("mrosi_self_imitation_token_scope must be one of {'all', 'tail'}.")
    if script_args.variant not in {"vanilla_opsd", "outcome_weighted_opsd", "aomp_uniform", "full_aomp_opsd"}:
        raise ValueError("variant must be one of vanilla_opsd, outcome_weighted_opsd, aomp_uniform, full_aomp_opsd.")
    if script_args.group_size <= 0:
        raise ValueError("group_size must be positive.")
    if script_args.audit_every_n_steps <= 0:
        raise ValueError("audit_every_n_steps must be positive.")
    if script_args.audit_start_step < 0:
        raise ValueError("audit_start_step must be non-negative.")
    if script_args.min_token_weight > script_args.max_token_weight:
        raise ValueError("min_token_weight must be <= max_token_weight.")
    if script_args.lookahead_lr < 0:
        raise ValueError("lookahead_lr must be non-negative.")
    if script_args.mrosi_self_imitation_start_ratio >= 0.0 and training_args.max_steps <= 0:
        raise ValueError("mrosi_self_imitation_start_ratio requires --max_steps > 0.")


def main():
    parser = TrlParser((AOMPOPSDScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    _validate_args(script_args, training_args, model_args)

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        run_name = f"aomp_opsd_{script_args.variant}_{model_name}_lr{lr_str}_bs{effective_batch_size}"

    print("\n" + "=" * 80)
    print("AOMP-OPSD RUN CONFIGURATION")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Dataset: {script_args.dataset_name_or_path}")
    print(f"Variant: {script_args.variant}")
    print("=" * 80 + "\n")

    if os.environ.get("LOCAL_RANK", "0") == "0" and is_wandb_available():
        import wandb

        wandb.init(
            entity=getattr(training_args, "wandb_entity", None),
            project=getattr(training_args, "wandb_project", "AOMP-OPSD"),
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "dataset_name_or_path": script_args.dataset_name_or_path,
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": effective_batch_size,
                "max_steps": training_args.max_steps,
                "max_completion_length": training_args.max_completion_length,
                "variant": script_args.variant,
                "aomp_opsd_enabled": script_args.aomp_opsd_enabled,
                "lookahead_lr": script_args.lookahead_lr,
                "group_size": script_args.group_size,
                "audit_source": script_args.audit_source,
                "audit_start_step": script_args.audit_start_step,
                "audit_temperature": script_args.audit_temperature,
                "positive_path_weight": script_args.positive_path_weight,
                "negative_path_weight": script_args.negative_path_weight,
                "teacher_kl_weight": script_args.teacher_kl_weight,
                "self_distill_weight": script_args.self_distill_weight,
                "token_weight_normalization": script_args.token_weight_normalization,
                "use_teacher_reliability": script_args.use_teacher_reliability,
                "use_student_uncertainty": script_args.use_student_uncertainty,
                "use_prefix_reliability": script_args.use_prefix_reliability,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args.presence_penalty = script_args.presence_penalty
    training_args.remove_unused_columns = False
    if script_args.aomp_opsd_enabled:
        training_args.gradient_checkpointing = False
        training_args.gradient_checkpointing_kwargs = None

    model_dtype = _resolve_torch_dtype(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    dataset_kwargs = dict(getattr(training_args, "dataset_kwargs", None) or {})
    dataset_kwargs.setdefault("skip_prepare_dataset", True)
    training_args.dataset_kwargs = dataset_kwargs

    train_dataset = _load_train_dataset(script_args.dataset_name_or_path, script_args.dataset_split)
    if script_args.masked_derivation_column not in train_dataset.column_names:
        raise ValueError(f"Dataset must contain {script_args.masked_derivation_column!r}.")
    if script_args.skip_missing_masked_derivation:
        column = script_args.masked_derivation_column
        before_count = len(train_dataset)
        train_dataset = train_dataset.filter(lambda example: bool(str(example.get(column, "")).strip()))
        print(f"Filtered missing masked derivations: {before_count} -> {len(train_dataset)}")
    train_dataset = add_sft_text_column(train_dataset)

    trainer = AOMPOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=AOMPOPSDDataCollator(
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            student_thinking=script_args.student_thinking,
            teacher_thinking=script_args.teacher_thinking,
            masked_derivation_column=script_args.masked_derivation_column,
        ),
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        masked_derivation_column=script_args.masked_derivation_column,
        self_imitation_weight=script_args.mrosi_self_imitation_weight,
        self_imitation_start_step=script_args.mrosi_self_imitation_start_step,
        self_imitation_start_ratio=script_args.mrosi_self_imitation_start_ratio,
        self_imitation_token_scope=script_args.mrosi_self_imitation_token_scope,
        self_imitation_tail_tokens=script_args.mrosi_self_imitation_tail_tokens,
        aomp_opsd_enabled=script_args.aomp_opsd_enabled,
        variant=script_args.variant,
        step_size=script_args.step_size,
        lookahead_lr=script_args.lookahead_lr,
        group_size=script_args.group_size,
        audit_source=script_args.audit_source,
        audit_temperature=script_args.audit_temperature,
        positive_path_weight=script_args.positive_path_weight,
        negative_path_weight=script_args.negative_path_weight,
        teacher_kl_weight=script_args.teacher_kl_weight,
        self_distill_weight=script_args.self_distill_weight,
        token_weight_normalization=script_args.token_weight_normalization,
        use_teacher_reliability=script_args.use_teacher_reliability,
        use_student_uncertainty=script_args.use_student_uncertainty,
        use_prefix_reliability=script_args.use_prefix_reliability,
        prefix_decay_lambda=script_args.prefix_decay_lambda,
        max_token_weight=script_args.max_token_weight,
        min_token_weight=script_args.min_token_weight,
        audit_budget=script_args.audit_budget,
        audit_start_step=script_args.audit_start_step,
        audit_every_n_steps=script_args.audit_every_n_steps,
        vllm_approx_lookahead=script_args.vllm_approx_lookahead,
        log_diagnostics=script_args.log_diagnostics,
        advantage_clip_range=script_args.advantage_clip_range,
        compute_grad_cosine=script_args.compute_grad_cosine,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        trainer.add_callback(LogCompletionsCallback(trainer, generation_config, num_prompts=8))

    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
