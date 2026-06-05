from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object
from trl.extras.profiling import profiling_decorator
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from .audit import compute_sequence_audit_rewards
from .lookahead import is_lookahead_safe, temporary_sgd_step, tensor_list_norm, trainable_parameters
from .losses import (
    OutcomeCalibratedLossConfig,
    compute_group_advantages,
    compute_outcome_calibrated_opsd_loss,
    compute_student_entropy,
    compute_teacher_entropy,
    compute_token_kl,
)
from .mrosi_compat import MROSITrainer


@dataclass
class LossOutput:
    loss: torch.Tensor
    student_logits: torch.Tensor
    teacher_logits: torch.Tensor
    sampled_token_ids: torch.Tensor
    token_mask: torch.Tensor


class AOMPOPSDTrainer(MROSITrainer):
    """AOMP-style OPSD trainer.

    The original MRO-SI implementation is imported read-only. This subclass
    replaces only the training step with an A-OMP two-oracle shell:
    cheap OPSD hint h_t, temporary lookahead w_t, then audited OPSD direction g_t.
    """

    _tag_names = ["trl", "aomp-opsd"]
    _name = "AOMP-OPSD"

    def __init__(
        self,
        *args,
        aomp_opsd_enabled: bool = True,
        variant: str = "full_aomp_opsd",
        step_size: float = 1.0,
        lookahead_lr: float = 1.0e-5,
        group_size: int = 4,
        audit_source: str = "auto",
        audit_temperature: float = 1.0,
        positive_path_weight: float = 1.0,
        negative_path_weight: float = 1.0,
        teacher_kl_weight: float = 1.0,
        self_distill_weight: float = 1.0,
        token_weight_normalization: str = "sequence_sum",
        use_teacher_reliability: bool = True,
        use_student_uncertainty: bool = True,
        use_prefix_reliability: bool = True,
        prefix_decay_lambda: float = 1.0,
        max_token_weight: float = 5.0,
        min_token_weight: float = 0.0,
        audit_budget: int = 0,
        audit_start_step: int = 0,
        audit_every_n_steps: int = 1,
        vllm_approx_lookahead: bool = False,
        log_diagnostics: bool = True,
        advantage_clip_range: float = 5.0,
        compute_grad_cosine: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aomp_opsd_enabled = bool(aomp_opsd_enabled)
        self.variant = str(variant)
        self.step_size = float(step_size)
        self.lookahead_lr = float(lookahead_lr)
        self.group_size = max(1, int(group_size))
        self.audit_source = str(audit_source)
        self.audit_temperature = float(audit_temperature)
        self.positive_path_weight = float(positive_path_weight)
        self.negative_path_weight = float(negative_path_weight)
        self.teacher_kl_weight = float(teacher_kl_weight)
        self.self_distill_weight = float(self_distill_weight)
        self.token_weight_normalization = str(token_weight_normalization)
        self.use_teacher_reliability = bool(use_teacher_reliability)
        self.use_student_uncertainty = bool(use_student_uncertainty)
        self.use_prefix_reliability = bool(use_prefix_reliability)
        self.prefix_decay_lambda = float(prefix_decay_lambda)
        self.max_token_weight = float(max_token_weight)
        self.min_token_weight = float(min_token_weight)
        self.audit_budget = max(0, int(audit_budget))
        self.audit_start_step = max(0, int(audit_start_step))
        self.audit_every_n_steps = max(1, int(audit_every_n_steps))
        self.vllm_approx_lookahead = bool(vllm_approx_lookahead)
        self.log_diagnostics = bool(log_diagnostics)
        self.advantage_clip_range = float(advantage_clip_range)
        self.compute_grad_cosine = bool(compute_grad_cosine)

        valid_variants = {
            "vanilla_opsd",
            "outcome_weighted_opsd",
            "aomp_uniform",
            "full_aomp_opsd",
        }
        if self.variant not in valid_variants:
            raise ValueError(f"variant must be one of {sorted(valid_variants)}")
        if self.lookahead_lr < 0.0:
            raise ValueError("lookahead_lr must be non-negative.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if self.min_token_weight > self.max_token_weight:
            raise ValueError("min_token_weight must be <= max_token_weight.")

        print("\n" + "=" * 80)
        print("AOMP-OPSD experimental mode enabled")
        print(f"variant: {self.variant}")
        print(f"enabled: {self.aomp_opsd_enabled}")
        print(f"lookahead_lr: {self.lookahead_lr}")
        print(f"group_size: {self.group_size}")
        print(f"audit_source: {self.audit_source}")
        print(f"audit_start_step: {self.audit_start_step}")
        print(f"audit_every_n_steps: {self.audit_every_n_steps}")
        print(f"vllm_approx_lookahead: {self.vllm_approx_lookahead}")
        print("=" * 80 + "\n")

    def _audit_is_scheduled(self) -> bool:
        step = int(self.state.global_step)
        return step >= self.audit_start_step and (step % self.audit_every_n_steps) == 0

    def _loss_config(self, uniform_token_weights: bool) -> OutcomeCalibratedLossConfig:
        return OutcomeCalibratedLossConfig(
            audit_temperature=self.audit_temperature,
            positive_path_weight=self.positive_path_weight,
            negative_path_weight=self.negative_path_weight,
            teacher_kl_weight=self.teacher_kl_weight,
            self_distill_weight=self.self_distill_weight,
            token_weight_normalization=self.token_weight_normalization,
            use_teacher_reliability=self.use_teacher_reliability,
            use_student_uncertainty=self.use_student_uncertainty,
            use_prefix_reliability=self.use_prefix_reliability,
            prefix_decay_lambda=self.prefix_decay_lambda,
            max_token_weight=self.max_token_weight,
            min_token_weight=self.min_token_weight,
            uniform_token_weights=uniform_token_weights,
            temperature=self.temperature,
            top_k_loss=self.top_k_loss,
        )

    def _expand_for_group(self, inputs: dict[str, Any], group_size: int) -> tuple[dict[str, Any], torch.Tensor]:
        if group_size <= 1:
            batch_size = int(inputs["student_prompts"].shape[0])
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.accelerator.device)
            return dict(inputs), group_ids

        batch_size = int(inputs["student_prompts"].shape[0])
        expanded: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                expanded[key] = value.repeat_interleave(group_size, dim=0)
            elif isinstance(value, list) and len(value) == batch_size:
                expanded[key] = [item for item in value for _ in range(group_size)]
            else:
                expanded[key] = value
        group_ids = torch.arange(batch_size, dtype=torch.long, device=self.accelerator.device).repeat_interleave(
            group_size
        )
        return expanded, group_ids

    def _slice_batch(
        self,
        inputs: dict[str, Any],
        limit: int,
        group_ids: torch.Tensor,
        prompt_texts: list[str],
        completion_texts: list[str],
    ) -> tuple[dict[str, Any], torch.Tensor, list[str], list[str]]:
        if limit <= 0 or limit >= int(group_ids.numel()):
            return inputs, group_ids, prompt_texts, completion_texts

        sliced: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == group_ids.numel():
                sliced[key] = value[:limit]
            elif isinstance(value, list) and len(value) == group_ids.numel():
                sliced[key] = value[:limit]
            else:
                sliced[key] = value
        return sliced, group_ids[:limit], prompt_texts[:limit], completion_texts[:limit]

    @torch.no_grad()
    def _generate_rollout_inputs(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        group_size: int = 1,
        force_transformers: bool = False,
    ) -> tuple[dict[str, Any], list[str], list[str], torch.Tensor]:
        rollout_inputs, group_ids = self._expand_for_group(inputs, group_size)

        if getattr(self, "use_vllm", False) and not force_transformers:
            self._wake_vllm_if_needed()
            (
                generated_ids,
                generated_attention_mask,
                _,
                prompt_texts,
                completion_texts,
                student_prompt_len,
                prompt_lengths,
            ) = self._generate_on_policy_outputs_vllm(
                rollout_inputs,
                self.generation_config,
                self.processing_class.pad_token_id,
            )
            rollout_inputs["student_prompt_length"] = student_prompt_len
            rollout_inputs["student_prompt_lengths_per_example"] = prompt_lengths
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generated_ids, generated_attention_mask, _ = self.generate_on_policy_outputs(
                    unwrapped_model,
                    rollout_inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            student_prompt_len = rollout_inputs["student_prompt_length"]
            prompt_texts = self.processing_class.batch_decode(
                rollout_inputs["student_prompts"],
                skip_special_tokens=False,
            )
            completion_ids = generated_ids[:, student_prompt_len:]
            completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        generation_ids = generated_ids[:, student_prompt_len:]
        rollout_inputs["student_input_ids"] = generated_ids
        rollout_inputs["student_attention_mask"] = generated_attention_mask

        teacher_full_ids = torch.cat([rollout_inputs["teacher_prompts"], generation_ids], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0
        rollout_inputs["teacher_input_ids"] = teacher_full_ids
        rollout_inputs["teacher_attention_mask"] = teacher_attention_mask

        labels = generated_ids.clone()
        labels[:, :student_prompt_len] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100
        rollout_inputs["labels"] = labels

        return rollout_inputs, prompt_texts, completion_texts, group_ids

    @staticmethod
    def _generalized_jsd_token_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        beta: float,
        temperature: float,
        top_k: int | None,
        token_clip: float | None,
    ) -> torch.Tensor:
        temperature = max(float(temperature), 1e-8)
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits.detach() / temperature
        if top_k is not None and top_k > 0:
            k = min(int(top_k), teacher_logits.size(-1))
            _, top_k_indices = torch.topk(teacher_logits, k=k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        if beta == 0:
            token_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
        elif beta == 1:
            token_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
        else:
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta_tensor),
                        teacher_log_probs + torch.log(beta_tensor),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
            kl_student = F.kl_div(mixture, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
            token_loss = beta_tensor * kl_teacher + (1.0 - beta_tensor) * kl_student

        if token_clip is not None:
            token_loss = token_loss.clamp(max=float(token_clip))
        return token_loss

    def _compute_proxy_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        sequence_weights: torch.Tensor | None = None,
    ) -> LossOutput:
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        token_mask = shifted_labels != -100

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
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :].detach()
            del outputs_teacher
            empty_cache()

        if sequence_weights is None:
            loss = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
        else:
            token_loss = self._generalized_jsd_token_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            weights = token_mask.to(dtype=token_loss.dtype) * sequence_weights.to(
                device=token_loss.device, dtype=token_loss.dtype
            ).view(-1, 1)
            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

        return LossOutput(
            loss=loss,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            sampled_token_ids=sampled_token_ids,
            token_mask=token_mask,
        )

    def _compute_audited_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        uniform_token_weights: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        proxy = self._compute_proxy_loss(model, inputs)
        loss, diagnostics = compute_outcome_calibrated_opsd_loss(
            student_logits=proxy.student_logits,
            teacher_logits=proxy.teacher_logits.detach(),
            sampled_token_ids=proxy.sampled_token_ids,
            mask=proxy.token_mask,
            advantages=advantages,
            rewards=rewards,
            config=self._loss_config(uniform_token_weights=uniform_token_weights),
        )
        diagnostics["proxy_loss"] = proxy.loss.detach()
        del proxy
        empty_cache()
        return loss, diagnostics

    def _log_proxy_diagnostics(self, proxy: LossOutput) -> None:
        if not self.log_diagnostics:
            return
        mask = proxy.token_mask
        token_kl = compute_token_kl(
            proxy.teacher_logits.detach(),
            proxy.student_logits,
            mask=mask,
            temperature=self.temperature,
            top_k=self.top_k_loss,
        )
        student_entropy = compute_student_entropy(
            proxy.student_logits,
            temperature=self.temperature,
            top_k=self.top_k_loss,
        )
        teacher_entropy = compute_teacher_entropy(
            proxy.teacher_logits.detach(),
            temperature=self.temperature,
            top_k=self.top_k_loss,
        )
        mask_f = mask.to(dtype=token_kl.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        self._record_metric("proxy_opsd_loss", float(proxy.loss.detach().item()))
        self._record_metric("proxy_token_kl", float((token_kl.detach() * mask_f).sum().item() / denom.item()))
        self._record_metric(
            "proxy_student_entropy",
            float((student_entropy.detach() * mask_f).sum().item() / denom.item()),
        )
        self._record_metric(
            "proxy_teacher_entropy",
            float((teacher_entropy.detach() * mask_f).sum().item() / denom.item()),
        )

    def _log_audit_diagnostics(
        self,
        diagnostics: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        audit_budget_used: int,
        audit_rate: float,
        exact_lookahead: bool,
    ) -> None:
        if not self.log_diagnostics:
            return
        prefix_map = {
            "audit_reward_mean": "audit_reward_mean",
            "audit_reward_std": "audit_reward_std",
            "advantage_mean": "audit_advantage_mean",
            "advantage_std": "audit_advantage_std",
            "loss_positive": "positive_path_loss",
            "loss_negative": "negative_path_loss",
            "token_kl_mean": "audited_token_kl",
            "token_weight_mean": "audited_token_weight_mean",
            "token_weight_max": "audited_token_weight_max",
        }
        for key, metric_name in prefix_map.items():
            if key in diagnostics:
                self._record_metric(metric_name, float(diagnostics[key].detach().float().item()))
        self._record_metric("audit_budget_used", float(audit_budget_used))
        self._record_metric("audit_rate", float(audit_rate))
        self._record_metric("aomp_exact_lookahead_update", 1.0 if exact_lookahead else 0.0)
        self._record_metric(
            "correct_rollout_retention_rate",
            float(((advantages > 0).float() * rewards.float()).mean().detach().item()) if advantages.numel() else 0.0,
        )
        self._record_metric(
            "failed_rollout_correction_rate",
            float((advantages < 0).float().mean().detach().item()) if advantages.numel() else 0.0,
        )

    def _backward_loss(self, loss: torch.Tensor) -> None:
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        if getattr(self, "is_deepspeed_enabled", False):
            self.accelerator.backward(loss)
        else:
            self.accelerator.backward(loss / grad_accum)

    def _current_grad_norm(self) -> float:
        grads = [param.grad for param in trainable_parameters(self.model)]
        return tensor_list_norm(grads)

    def _update_loss_totals(self, loss: torch.Tensor) -> None:
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += float(loss.detach().float().item())
        self._on_policy_step_equiv += 1.0 / grad_accum

    def _audit_rollouts(
        self,
        rollout_inputs: dict[str, Any],
        completion_texts: list[str],
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        result = compute_sequence_audit_rewards(
            completions=completion_texts,
            reference_solutions=list(rollout_inputs.get("solutions", [])),
            audit_source=self.audit_source,
            device=self.accelerator.device,
        )
        advantages = compute_group_advantages(
            result.rewards,
            group_ids=group_ids.to(device=result.rewards.device),
            clip_range=self.advantage_clip_range,
        )
        return result.rewards, advantages, result.source

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if (not self.aomp_opsd_enabled) or self.variant == "vanilla_opsd":
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if not self._audit_is_scheduled():
            rollout_inputs, _, _, _ = self._generate_rollout_inputs(model, inputs, group_size=1)
            proxy = self._compute_proxy_loss(model, rollout_inputs)
            self._log_proxy_diagnostics(proxy)
            self._record_metric("audit_rate", 0.0)
            self._backward_loss(proxy.loss)
            self._update_loss_totals(proxy.loss)
            return proxy.loss.detach()

        if self.variant == "outcome_weighted_opsd":
            rollout_inputs, prompt_texts, completion_texts, group_ids = self._generate_rollout_inputs(
                model,
                inputs,
                group_size=self.group_size,
            )
            rollout_inputs, group_ids, prompt_texts, completion_texts = self._slice_batch(
                rollout_inputs,
                self.audit_budget,
                group_ids,
                prompt_texts,
                completion_texts,
            )
            rewards, advantages, _ = self._audit_rollouts(rollout_inputs, completion_texts, group_ids)
            sequence_weights = torch.sigmoid(advantages / max(self.audit_temperature, 1e-8)).detach()
            proxy = self._compute_proxy_loss(model, rollout_inputs, sequence_weights=sequence_weights)
            self._log_proxy_diagnostics(proxy)
            diagnostics = {
                "audit_reward_mean": rewards.mean().detach(),
                "audit_reward_std": rewards.std(unbiased=False).detach(),
                "advantage_mean": advantages.mean().detach(),
                "advantage_std": advantages.std(unbiased=False).detach(),
                "loss_positive": proxy.loss.detach(),
                "loss_negative": proxy.loss.detach() * 0.0,
                "token_kl_mean": proxy.loss.detach() * 0.0,
                "token_weight_mean": sequence_weights.mean().detach(),
                "token_weight_max": sequence_weights.max().detach(),
            }
            self._log_audit_diagnostics(
                diagnostics,
                rewards,
                advantages,
                audit_budget_used=int(rewards.numel()),
                audit_rate=1.0,
                exact_lookahead=False,
            )
            self._backward_loss(proxy.loss)
            self._update_loss_totals(proxy.loss)
            return proxy.loss.detach()

        proxy_inputs, _, _, _ = self._generate_rollout_inputs(model, inputs, group_size=1)
        proxy = self._compute_proxy_loss(model, proxy_inputs)
        self._log_proxy_diagnostics(proxy)
        params = trainable_parameters(model)
        hint_grads = list(torch.autograd.grad(proxy.loss, params, allow_unused=True))
        hint_grad_norm = tensor_list_norm(hint_grads)
        self._record_metric("hint_grad_norm", float(hint_grad_norm))
        self._record_metric("hint_lookahead_cosine", 0.0)  # TODO: compute behind compute_grad_cosine when needed.

        exact_lookahead = is_lookahead_safe(self)
        if getattr(self, "use_vllm", False) and self.vllm_approx_lookahead:
            exact_lookahead = False
        uniform_weights = self.variant == "aomp_uniform"

        if exact_lookahead:
            with temporary_sgd_step(model, hint_grads, step_size=self.lookahead_lr) as lookahead_stats:
                rollout_inputs, prompt_texts, completion_texts, group_ids = self._generate_rollout_inputs(
                    model,
                    inputs,
                    group_size=self.group_size,
                    force_transformers=getattr(self, "use_vllm", False),
                )
                rollout_inputs, group_ids, prompt_texts, completion_texts = self._slice_batch(
                    rollout_inputs,
                    self.audit_budget,
                    group_ids,
                    prompt_texts,
                    completion_texts,
                )
                rewards, advantages, _ = self._audit_rollouts(rollout_inputs, completion_texts, group_ids)
                audit_loss, diagnostics = self._compute_audited_loss(
                    model,
                    rollout_inputs,
                    rewards=rewards,
                    advantages=advantages,
                    uniform_token_weights=uniform_weights,
                )
                self._backward_loss(audit_loss)
                lookahead_grad_norm = self._current_grad_norm()
                loss_for_return = audit_loss.detach()
                self._record_metric("lookahead_update_norm", float(lookahead_stats.update_norm))
        else:
            # Approximation: ZeRO-3/FSDP cannot safely swap parameter shards in v1.
            # We still sample from the current policy and recompute the audited loss on z_t.
            rollout_inputs, prompt_texts, completion_texts, group_ids = self._generate_rollout_inputs(
                model,
                inputs,
                group_size=self.group_size,
            )
            rollout_inputs, group_ids, prompt_texts, completion_texts = self._slice_batch(
                rollout_inputs,
                self.audit_budget,
                group_ids,
                prompt_texts,
                completion_texts,
            )
            rewards, advantages, _ = self._audit_rollouts(rollout_inputs, completion_texts, group_ids)
            audit_loss, diagnostics = self._compute_audited_loss(
                model,
                rollout_inputs,
                rewards=rewards,
                advantages=advantages,
                uniform_token_weights=uniform_weights,
            )
            self._backward_loss(audit_loss)
            lookahead_grad_norm = self._current_grad_norm()
            loss_for_return = audit_loss.detach()
            self._record_metric("lookahead_update_norm", 0.0)

        self._record_metric("lookahead_grad_norm", float(lookahead_grad_norm))
        self._log_audit_diagnostics(
            diagnostics,
            rewards,
            advantages,
            audit_budget_used=int(rewards.numel()),
            audit_rate=1.0,
            exact_lookahead=exact_lookahead,
        )
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        self._update_loss_totals(loss_for_return)
        return loss_for_return
