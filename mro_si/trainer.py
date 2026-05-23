from __future__ import annotations

import os
import random
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import disable_dropout_in_model, empty_cache, ensure_master_addr_port
from trl.experimental.gold.gold_config import GOLDConfig

from .data_collator import MROSIDataCollator
from .verifier_utils import batch_verify_answer


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb
else:
    wandb = None

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class MROSIVLLMSyncCallback(TrainerCallback):
    """Synchronize updated student weights to vLLM after optimizer steps."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
            and getattr(self.trainer.accelerator, "sync_gradients", False)
        ):
            self.trainer._move_model_to_vllm()
            self.trainer._last_vllm_sync_step = state.global_step


class MROSITrainer(SFTTrainer):
    """MRO-SI trainer.

    The fast channel is masked-route on-policy distillation. The slow channel is
    a verifier-gated self-imitation loss that is applied only to rollout tokens
    whose final boxed answer matches the reference solution.
    """

    _tag_names = ["trl", "mro-si"]
    _name = "MRO-SI"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        fixed_teacher: bool = True,
        top_k_loss: int | None = 200,
        jsd_token_clip: float | None = 0.05,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        masked_derivation_column: str = "masked_derivation",
        self_imitation_weight: float = 0.01,
        self_imitation_start_step: int = 75,
        self_imitation_start_ratio: float = -1.0,
        self_imitation_token_scope: str = "all",
        self_imitation_tail_tokens: int = 256,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        self.masked_derivation_column = masked_derivation_column
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        if args is not None:
            args.remove_unused_columns = False
            dataset_kwargs = dict(getattr(args, "dataset_kwargs", None) or {})
            dataset_kwargs.setdefault("skip_prepare_dataset", True)
            args.dataset_kwargs = dataset_kwargs

        if data_collator is None:
            data_collator = MROSIDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
                masked_derivation_column=masked_derivation_column,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(self.model)

        self.beta = args.beta
        self.temperature = args.temperature
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.fixed_teacher = fixed_teacher
        self.self_imitation_weight = float(self_imitation_weight)
        self.self_imitation_start_step = int(self_imitation_start_step)
        self.self_imitation_start_ratio = float(self_imitation_start_ratio)
        self.self_imitation_token_scope = self_imitation_token_scope
        self.self_imitation_tail_tokens = int(self_imitation_tail_tokens)
        self._pending_outcome_correct = None

        if self.fixed_teacher and peft_config is None:
            raise ValueError("fixed_teacher=True requires PEFT/LoRA. Pass --use_peft for MRO-SI runs.")
        if self.self_imitation_weight < 0.0:
            raise ValueError("self_imitation_weight must be non-negative.")
        if self.self_imitation_start_step < 0:
            raise ValueError("self_imitation_start_step must be non-negative.")
        if not -1.0 <= self.self_imitation_start_ratio <= 1.0:
            raise ValueError("self_imitation_start_ratio must be in [-1, 1]. Use -1 to disable ratio mode.")
        if self.self_imitation_token_scope not in {"all", "tail"}:
            raise ValueError("self_imitation_token_scope must be one of {'all', 'tail'}.")
        if self.self_imitation_tail_tokens <= 0:
            raise ValueError("self_imitation_tail_tokens must be positive.")

        print("\n" + "=" * 80)
        print("MRO-SI mode enabled")
        print(f"Masked derivation column: {self.masked_derivation_column}")
        print(f"Fixed teacher: {self.fixed_teacher}")
        print(f"Top-k JSD loss: {self.top_k_loss}")
        print(f"JSD token clip: {self.jsd_token_clip}")
        print(
            "Self-imitation: "
            f"weight={self.self_imitation_weight}, "
            f"start_step={self.self_imitation_start_step}, "
            f"start_ratio={self.self_imitation_start_ratio}, "
            f"effective_start_step={self._self_imitation_effective_start_step()}, "
            f"token_scope={self.self_imitation_token_scope}"
        )
        print("=" * 80 + "\n")

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if getattr(self.model.generation_config, "eos_token_id", None) is not None:
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        self.log_completions = getattr(args, "log_completions", False)
        self.log_completion_steps = getattr(args, "log_completions_steps", 1)
        self.wandb_log_unique_prompts = getattr(args, "wandb_log_unique_prompts", False)
        self.num_completions_to_print = getattr(args, "num_completions_to_print", None)
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * getattr(
            args, "steps_per_generation", 1
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError("vLLM is not available. Install vllm or remove --use_vllm.")
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                if self.accelerator.num_processes % self.vllm_tensor_parallel_size != 0:
                    raise ValueError(
                        "vllm_tensor_parallel_size must divide the number of training processes evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=self.model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )
                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1
            self.add_callback(MROSIVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
            self.masked_derivation_column,
            f"{self.masked_derivation_column}_source",
            "masked_derivation_source",
            "source",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        top_k=None,
        token_clip=None,
    ):
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        if reduction == "sum":
            return jsd.sum()
        if reduction == "mean":
            return jsd.mean()
        return jsd

    def _teacher_adapter_context(self, model):
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _record_metric(self, name: str, value: float, mode: str | None = None) -> None:
        if mode is None:
            mode = "train" if self.model.training else "eval"
        self._metrics[mode][name].append(value)

    def _self_imitation_effective_start_step(self) -> int:
        if self.self_imitation_start_ratio >= 0.0:
            max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
            if max_steps <= 0:
                raise ValueError("self_imitation_start_ratio requires --max_steps > 0.")
            return int((self.self_imitation_start_ratio * max_steps) + 0.999999)
        return self.self_imitation_start_step

    def _self_imitation_is_active(self) -> bool:
        return self.self_imitation_weight > 0.0 and int(self.state.global_step) >= self._self_imitation_effective_start_step()

    def _self_imitation_loss(
        self,
        student_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._self_imitation_is_active() or self._pending_outcome_correct is None:
            return None

        correct = self._pending_outcome_correct.to(device=student_logits.device, dtype=torch.bool)
        if correct.numel() == 0 or not bool(correct.any().item()):
            self._record_metric("self_imitation_active_rate", 0.0)
            self._record_metric("self_imitation_loss", 0.0)
            return student_logits.sum() * 0.0

        token_mask = (labels != -100) & correct.view(-1, 1)
        if self.self_imitation_token_scope == "tail":
            tail_mask = torch.zeros_like(token_mask)
            for row_idx in range(token_mask.size(0)):
                if not bool(correct[row_idx].item()):
                    continue
                active_positions = torch.nonzero(token_mask[row_idx], as_tuple=False).squeeze(-1)
                if active_positions.numel() > 0:
                    tail_mask[row_idx, active_positions[-self.self_imitation_tail_tokens :]] = True
            token_mask = tail_mask

        if not bool(token_mask.any().item()):
            self._record_metric("self_imitation_active_rate", 0.0)
            self._record_metric("self_imitation_loss", 0.0)
            return student_logits.sum() * 0.0

        log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        sampled_log_probs = torch.gather(log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)).squeeze(-1)
        token_weights = token_mask.float()
        imitation_loss = -(sampled_log_probs * token_weights).sum() / token_weights.sum().clamp_min(1.0)
        self._record_metric("self_imitation_active_rate", float(correct.float().mean().detach().item()))
        self._record_metric("self_imitation_loss", float(imitation_loss.detach().item()))
        return imitation_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        del outputs_student
        empty_cache()

        with torch.no_grad(), self._teacher_adapter_context(model):
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            del outputs_teacher
            empty_cache()

        loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        self._record_metric("masked_fast_channel_loss", float(loss.detach().item()))

        imitation_loss = self._self_imitation_loss(
            student_logits=student_logits,
            sampled_token_ids=sampled_token_ids,
            labels=shifted_labels,
        )
        if imitation_loss is not None:
            weighted_imitation = self.self_imitation_weight * imitation_loss
            loss = loss + weighted_imitation
            self._record_metric("self_imitation_weighted_loss", float(weighted_imitation.detach().item()))

        del student_logits, teacher_logits
        empty_cache()

        if return_outputs:
            class MinimalOutput:
                def __init__(self, loss):
                    self.loss = loss

            return loss, MinimalOutput(loss)
        return loss

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache
        model.config.use_cache = True
        generation_config.use_cache = True
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            generated_tokens = generated_outputs.sequences
        finally:
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        attention_mask = torch.ones_like(generated_tokens)
        labels = generated_tokens.clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
            attention_mask[generated_tokens == pad_token_id] = 0
        return generated_tokens, attention_mask, labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm]
        prompts_text_with_special = prompts_text_for_vllm

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        top_p = getattr(self.args, "top_p", 1.0)
        repetition_penalty = getattr(self.args, "repetition_penalty", 1.0)
        min_p = getattr(self.args, "min_p", 0.0)
        presence_penalty = getattr(self.args, "presence_penalty", 0.0)

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            guided_decoding = (
                GuidedDecodingParams(backend="outlines", regex=self.vllm_guided_decoding_regex)
                if self.vllm_guided_decoding_regex
                else None
            )
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        device = self.accelerator.device
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids
        prompt_lengths = prompt_tokenized.attention_mask.sum(dim=1).to(dtype=torch.long)

        padded_completion_ids = []
        for ids in completion_ids:
            completion_tensor = torch.tensor(ids, device=device)
            if len(completion_tensor) > max_completion_length:
                completion_tensor = completion_tensor[:max_completion_length]
            elif len(completion_tensor) < max_completion_length:
                completion_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (max_completion_length - len(completion_tensor),),
                            pad_token_id,
                            device=device,
                            dtype=completion_tensor.dtype,
                        ),
                    ]
                )
            padded_completion_ids.append(completion_tensor)

        padded_completion_ids = torch.stack(padded_completion_ids)
        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)
        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        completion_texts = [
            self.processing_class.decode(ids, skip_special_tokens=False) for ids in completion_ids
        ]
        return (
            new_input_ids,
            new_attention_mask,
            new_labels,
            prompts_text_with_special,
            completion_texts,
            prompt_ids.shape[1],
            prompt_lengths,
        )

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                if self.is_fsdp_enabled:
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    for name, param in self.model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name or "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                self.model.unmerge_adapter()
        else:
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        if not self.accelerator.is_main_process or not self._generation_outputs_buffer:
            return
        import json
        from pathlib import Path

        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        output_file = generations_dir / f"generations_step_{step}.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": step,
                    "num_samples": len(self._generation_outputs_buffer),
                    "generations": self._generation_outputs_buffer,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if self.use_vllm:
            self._wake_vllm_if_needed()
            (
                generated_ids,
                generated_attention_mask,
                _,
                prompt_texts,
                completion_texts,
                student_prompt_len,
                prompt_lengths,
            ) = self._generate_on_policy_outputs_vllm(inputs, self.generation_config, self.processing_class.pad_token_id)
            inputs["student_prompt_length"] = student_prompt_len
            inputs["student_prompt_lengths_per_example"] = prompt_lengths
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generated_ids, generated_attention_mask, _ = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            student_prompt_len = inputs["student_prompt_length"]
            prompt_texts = self.processing_class.batch_decode(inputs["student_prompts"], skip_special_tokens=False)
            completion_ids = generated_ids[:, student_prompt_len:]
            completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        generation_ids = generated_ids[:, student_prompt_len:]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        teacher_full_ids = torch.cat([inputs["teacher_prompts"], generation_ids], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0
        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        labels = generated_ids.clone()
        for row_idx in range(labels.shape[0]):
            actual_prompt_len = int(inputs["student_prompt_lengths_per_example"][row_idx].item())
            labels[row_idx, :actual_prompt_len] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100
        inputs["labels"] = labels

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        if self._self_imitation_is_active():
            outcome_correct = batch_verify_answer(completion_texts, inputs.get("solutions", []))
            self._pending_outcome_correct = torch.tensor(
                outcome_correct,
                dtype=torch.bool,
                device=self.accelerator.device,
            )
            verified_rate = (
                float(self._pending_outcome_correct.float().mean().detach().item())
                if self._pending_outcome_correct.numel() > 0
                else 0.0
            )
            self._record_metric("student_rollout_verified_rate", verified_rate)
        else:
            self._pending_outcome_correct = None

        loss = super().training_step(model, inputs, num_items_in_batch)
        self._pending_outcome_correct = None

        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += loss_scalar
        self._on_policy_step_equiv += 1.0 / grad_accum
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            on_sum, off_sum, on_eq, off_eq = vec.tolist()
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
            and wandb is not None
            and self.args.report_to
            and "wandb" in self.args.report_to
            and wandb.run is not None
        ):
            import pandas as pd

            table = {
                "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                "prompt": self._textual_logs["prompt"],
                "completion": self._textual_logs["completion"],
            }
            df = pd.DataFrame(table)
            if self.wandb_log_unique_prompts:
                df = df.drop_duplicates(subset=["prompt"])
            if self.num_completions_to_print and len(df) > 0:
                df = df.sample(n=self.num_completions_to_print, random_state=42)
            wandb.log({"completions": wandb.Table(dataframe=df)})
