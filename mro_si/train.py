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

from mro_si.trainer import MROSITrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


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
class MROSIScriptArguments(ScriptArguments):
    """Arguments specific to MRO-SI."""

    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name used for output_dir suffix and WandB run name."},
    )
    presence_penalty: float = field(default=0.0, metadata={"help": "Sampling presence penalty for rollouts."})
    fixed_teacher: bool = field(
        default=True,
        metadata={"help": "Use the frozen base model as the masked-route teacher by disabling LoRA adapters."},
    )
    top_k_loss: int = field(
        default=200,
        metadata={"help": "Restrict fast-channel JSD to the teacher top-k vocabulary entries. Set 0 for full vocab."},
    )
    jsd_token_clip: float = field(default=0.05, metadata={"help": "Per-token JSD clipping value. Set 0 to disable."})
    student_thinking: bool = field(default=False, metadata={"help": "Enable thinking mode in student rollouts."})
    teacher_thinking: bool = field(default=True, metadata={"help": "Enable thinking mode in masked-teacher prompts."})
    dataset_name_or_path: str = field(
        default="data/openthoughts_math_30k_masked.jsonl",
        metadata={"help": "HF dataset name or local JSON/JSONL file with problem, solution, masked_derivation."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split for HF datasets."})
    masked_derivation_column: str = field(
        default="masked_derivation",
        metadata={"help": "Dataset column containing final-answer-masked derivation routes."},
    )
    skip_missing_masked_derivation: bool = field(
        default=True,
        metadata={"help": "Drop examples with empty masked derivations before training."},
    )
    mrosi_self_imitation_weight: float = field(
        default=0.01,
        metadata={"help": "Multiplier for verifier-correct rollout self-imitation."},
    )
    mrosi_self_imitation_start_step: int = field(
        default=75,
        metadata={"help": "Global step at which verifier-gated self-imitation starts."},
    )
    mrosi_self_imitation_start_ratio: float = field(
        default=-1.0,
        metadata={"help": "Training-progress ratio for self-imitation start. Use -1 to use start_step."},
    )
    mrosi_self_imitation_token_scope: str = field(
        default="all",
        metadata={"help": "Self-imitation token scope: all or tail."},
    )
    mrosi_self_imitation_tail_tokens: int = field(
        default=256,
        metadata={"help": "Number of final tokens used when token_scope=tail."},
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


def main():
    parser = TrlParser((MROSIScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("MRO-SI fixed_teacher=True requires --use_peft.")
    if script_args.mrosi_self_imitation_start_step < 0:
        raise ValueError("mrosi_self_imitation_start_step must be non-negative.")
    if script_args.mrosi_self_imitation_token_scope not in {"all", "tail"}:
        raise ValueError("mrosi_self_imitation_token_scope must be one of {'all', 'tail'}.")
    if not -1.0 <= script_args.mrosi_self_imitation_start_ratio <= 1.0:
        raise ValueError("mrosi_self_imitation_start_ratio must be in [-1, 1].")
    if script_args.mrosi_self_imitation_start_ratio >= 0.0 and training_args.max_steps <= 0:
        raise ValueError("mrosi_self_imitation_start_ratio requires --max_steps > 0.")

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
        run_name = (
            f"mrosi_{model_name}_lr{lr_str}_bs{effective_batch_size}_"
            f"si{script_args.mrosi_self_imitation_start_step}"
        )

    print("\n" + "=" * 80)
    print("RUN CONFIGURATION")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Dataset: {script_args.dataset_name_or_path}")
    print("=" * 80 + "\n")

    if os.environ.get("LOCAL_RANK", "0") == "0" and is_wandb_available():
        import wandb

        wandb.init(
            entity=getattr(training_args, "wandb_entity", None),
            project=getattr(training_args, "wandb_project", "MRO-SI"),
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "dataset_name_or_path": script_args.dataset_name_or_path,
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": effective_batch_size,
                "max_steps": training_args.max_steps,
                "max_completion_length": training_args.max_completion_length,
                "masked_derivation_column": script_args.masked_derivation_column,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss,
                "jsd_token_clip": script_args.jsd_token_clip,
                "mrosi_self_imitation_weight": script_args.mrosi_self_imitation_weight,
                "mrosi_self_imitation_start_step": script_args.mrosi_self_imitation_start_step,
                "mrosi_self_imitation_start_ratio": script_args.mrosi_self_imitation_start_ratio,
                "mrosi_self_imitation_token_scope": script_args.mrosi_self_imitation_token_scope,
            },
        )

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
    dataset_kwargs = dict(getattr(training_args, "dataset_kwargs", None) or {})
    dataset_kwargs.setdefault("skip_prepare_dataset", True)
    training_args.dataset_kwargs = dataset_kwargs

    train_dataset = _load_train_dataset(script_args.dataset_name_or_path, script_args.dataset_split)
    if script_args.masked_derivation_column not in train_dataset.column_names:
        raise ValueError(
            f"Dataset must contain {script_args.masked_derivation_column!r}. "
            "Build it with scripts/build_masked_derivations.py first."
        )
    if script_args.skip_missing_masked_derivation:
        before_count = len(train_dataset)
        column = script_args.masked_derivation_column
        train_dataset = train_dataset.filter(lambda example: bool(str(example.get(column, "")).strip()))
        print(f"Filtered missing masked derivations: {before_count} -> {len(train_dataset)}")

    train_dataset = add_sft_text_column(train_dataset)

    trainer = MROSITrainer(
        model=model_args.model_name_or_path,
        args=training_args,
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
