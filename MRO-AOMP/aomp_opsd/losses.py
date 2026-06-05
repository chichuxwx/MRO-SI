from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OutcomeCalibratedLossConfig:
    audit_temperature: float = 1.0
    positive_path_weight: float = 1.0
    negative_path_weight: float = 1.0
    teacher_kl_weight: float = 1.0
    self_distill_weight: float = 1.0
    token_weight_normalization: str = "sequence_sum"
    use_teacher_reliability: bool = True
    use_student_uncertainty: bool = True
    use_prefix_reliability: bool = True
    prefix_decay_lambda: float = 1.0
    max_token_weight: float = 5.0
    min_token_weight: float = 0.0
    uniform_token_weights: bool = False
    temperature: float = 1.0
    top_k_loss: int | None = 200


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
    clip_range: float | None = 5.0,
) -> torch.Tensor:
    """Normalize rewards relative to other samples from the same prompt group."""

    rewards = rewards.to(dtype=torch.float32)
    group_ids = group_ids.to(device=rewards.device)
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids):
        idx = group_ids == group_id
        group_rewards = rewards[idx]
        if group_rewards.numel() <= 1:
            centered = group_rewards - group_rewards.mean()
            scaled = centered
        else:
            centered = group_rewards - group_rewards.mean()
            std = group_rewards.std(unbiased=False)
            scaled = centered / std.clamp_min(eps)
        advantages[idx] = scaled

    if clip_range is not None and clip_range > 0:
        advantages = advantages.clamp(min=-float(clip_range), max=float(clip_range))
    return advantages


def _topk_logits(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    k = min(int(top_k), logits.size(-1))
    values, _ = torch.topk(logits, k=k, dim=-1)
    return values


def compute_student_entropy(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = _topk_logits(logits, top_k)
    scaled = logits / max(float(temperature), 1e-8)
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def compute_teacher_entropy(
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    return compute_student_entropy(teacher_logits.detach(), temperature=temperature, top_k=top_k)


def compute_token_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Per-token forward KL, KL(teacher || student)."""

    temperature = max(float(temperature), 1e-8)
    teacher_logits = teacher_logits.detach()
    if top_k is not None and top_k > 0 and top_k < teacher_logits.size(-1):
        k = min(int(top_k), teacher_logits.size(-1))
        teacher_logits, topk_indices = torch.topk(teacher_logits, k=k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if mask is not None:
        token_kl = token_kl * mask.to(dtype=token_kl.dtype)
    return token_kl


def compute_prefix_reliability(mask: torch.Tensor, lambda_decay: float = 1.0) -> torch.Tensor:
    """Simple prefix reliability fallback: early response tokens receive larger weight."""

    mask_bool = mask.to(dtype=torch.bool)
    if mask_bool.numel() == 0:
        return mask.to(dtype=torch.float32)

    active_positions = mask_bool.to(dtype=torch.float32).cumsum(dim=1) - 1.0
    lengths = mask_bool.to(dtype=torch.float32).sum(dim=1, keepdim=True).clamp_min(1.0)
    relative_pos = active_positions.clamp_min(0.0) / lengths
    reliability = torch.exp(-float(lambda_decay) * relative_pos)
    return reliability * mask_bool.to(dtype=reliability.dtype)


def normalize_token_weights(
    weights: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "sequence_sum",
    min_weight: float | None = None,
    max_weight: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize token weights without producing NaNs on all-zero rows."""

    mask_f = mask.to(dtype=weights.dtype)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) * mask_f
    if min_weight is not None or max_weight is not None:
        lo = float(min_weight) if min_weight is not None else None
        hi = float(max_weight) if max_weight is not None else None
        weights = weights.clamp(min=lo, max=hi)
        weights = weights * mask_f

    if mode in {"none", "raw", None}:
        return weights

    if mode == "sequence_mean":
        denom = weights.sum(dim=1, keepdim=True)
        active_count = mask_f.sum(dim=1, keepdim=True)
        fallback = mask_f
        normalized = torch.where(
            denom > eps,
            weights * active_count / denom.clamp_min(eps),
            fallback,
        )
        return normalized * mask_f

    if mode == "batch_sum":
        denom = weights.sum()
        fallback = mask_f / mask_f.sum().clamp_min(1.0)
        normalized = torch.where(denom > eps, weights / denom.clamp_min(eps), fallback)
        return normalized * mask_f

    if mode != "sequence_sum":
        raise ValueError(f"Unsupported token weight normalization mode: {mode}")

    denom = weights.sum(dim=1, keepdim=True)
    active_count = mask_f.sum(dim=1, keepdim=True)
    uniform = mask_f / active_count.clamp_min(1.0)
    normalized = torch.where(denom > eps, weights / denom.clamp_min(eps), uniform)
    return normalized * mask_f


def compute_outcome_calibrated_opsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor | None = None,
    config: OutcomeCalibratedLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Outcome-calibrated OPSD loss.

    Sequence-level outcomes route trajectories. They are deliberately not
    divided into token rewards.
    """

    if config is None:
        config = OutcomeCalibratedLossConfig()

    mask_bool = mask.to(dtype=torch.bool)
    mask_f = mask_bool.to(dtype=student_logits.dtype)
    if student_logits.numel() == 0 or not bool(mask_bool.any().item()):
        zero = student_logits.sum() * 0.0
        diagnostics = {
            "audit_reward_mean": zero.detach(),
            "audit_reward_std": zero.detach(),
            "advantage_mean": zero.detach(),
            "advantage_std": zero.detach(),
            "positive_fraction": zero.detach(),
            "negative_fraction": zero.detach(),
            "loss_positive": zero.detach(),
            "loss_negative": zero.detach(),
            "token_kl_mean": zero.detach(),
            "student_entropy_mean": zero.detach(),
            "teacher_entropy_mean": zero.detach(),
            "prefix_reliability_mean": zero.detach(),
            "token_weight_mean": zero.detach(),
            "token_weight_max": zero.detach(),
            "lookahead_loss": zero.detach(),
            "proxy_loss": zero.detach(),
        }
        return zero, diagnostics

    temperature = max(float(config.temperature), 1e-8)
    safe_ids = sampled_token_ids.masked_fill(~mask_bool, 0)
    scaled_student_logits = student_logits / temperature
    sampled_logits = torch.gather(scaled_student_logits, dim=-1, index=safe_ids.unsqueeze(-1)).squeeze(-1)
    nll = torch.logsumexp(scaled_student_logits, dim=-1) - sampled_logits

    token_kl = compute_token_kl(
        teacher_logits=teacher_logits.detach(),
        student_logits=student_logits,
        mask=mask_bool,
        temperature=temperature,
        top_k=config.top_k_loss,
    )
    student_entropy = compute_student_entropy(
        student_logits,
        temperature=temperature,
        top_k=config.top_k_loss,
    ) * mask_f
    teacher_entropy = compute_teacher_entropy(
        teacher_logits.detach(),
        temperature=temperature,
        top_k=config.top_k_loss,
    ) * mask_f
    prefix_reliability = compute_prefix_reliability(mask_bool, lambda_decay=config.prefix_decay_lambda)

    if config.uniform_token_weights:
        pos_raw = mask_f
        neg_raw = mask_f
    else:
        pos_raw = mask_f
        if config.use_student_uncertainty:
            # TODO: replace the uniform importance fallback with a learned or verifier-derived token importance signal.
            pos_raw = pos_raw * student_entropy.detach()

        neg_raw = token_kl.detach().clamp_min(0.0) * mask_f
        if config.use_teacher_reliability:
            neg_raw = neg_raw * torch.exp(-teacher_entropy.detach()).to(dtype=neg_raw.dtype)
        if config.use_prefix_reliability:
            neg_raw = neg_raw * prefix_reliability.detach().to(dtype=neg_raw.dtype)

    pos_weights = normalize_token_weights(
        pos_raw,
        mask_bool,
        mode=config.token_weight_normalization,
        min_weight=config.min_token_weight,
        max_weight=config.max_token_weight,
    )
    neg_weights = normalize_token_weights(
        neg_raw,
        mask_bool,
        mode=config.token_weight_normalization,
        min_weight=config.min_token_weight,
        max_weight=config.max_token_weight,
    )

    advantages = advantages.to(device=student_logits.device, dtype=student_logits.dtype)
    tau = max(float(config.audit_temperature), 1e-8)
    positive_route = (advantages > 0).to(dtype=student_logits.dtype)
    negative_route = (advantages < 0).to(dtype=student_logits.dtype)
    positive_gate = torch.sigmoid(advantages / tau) * positive_route
    negative_gate = torch.sigmoid(-advantages / tau) * negative_route

    pos_seq_loss = (nll * pos_weights).sum(dim=1)
    neg_seq_loss = (token_kl * neg_weights).sum(dim=1)

    pos_scale = float(config.positive_path_weight) * float(config.self_distill_weight)
    neg_scale = float(config.negative_path_weight) * float(config.teacher_kl_weight)
    pos_terms = pos_scale * positive_gate * pos_seq_loss
    neg_terms = neg_scale * negative_gate * neg_seq_loss
    denom = (pos_scale * positive_gate + neg_scale * negative_gate).sum().clamp_min(1.0)
    loss = (pos_terms.sum() + neg_terms.sum()) / denom

    active_weights = torch.cat([pos_weights[mask_bool], neg_weights[mask_bool]], dim=0)
    reward_values = rewards.to(device=student_logits.device, dtype=student_logits.dtype) if rewards is not None else None
    zero = loss.detach() * 0.0
    diagnostics = {
        "audit_reward_mean": reward_values.mean().detach() if reward_values is not None and reward_values.numel() else zero,
        "audit_reward_std": reward_values.std(unbiased=False).detach() if reward_values is not None and reward_values.numel() else zero,
        "advantage_mean": advantages.detach().mean() if advantages.numel() else zero,
        "advantage_std": advantages.detach().std(unbiased=False) if advantages.numel() else zero,
        "positive_fraction": positive_route.detach().mean() if positive_route.numel() else zero,
        "negative_fraction": negative_route.detach().mean() if negative_route.numel() else zero,
        "loss_positive": (pos_terms.sum() / positive_gate.sum().clamp_min(1.0)).detach(),
        "loss_negative": (neg_terms.sum() / negative_gate.sum().clamp_min(1.0)).detach(),
        "token_kl_mean": _masked_mean(token_kl.detach(), mask_bool),
        "student_entropy_mean": _masked_mean(student_entropy.detach(), mask_bool),
        "teacher_entropy_mean": _masked_mean(teacher_entropy.detach(), mask_bool),
        "prefix_reliability_mean": _masked_mean(prefix_reliability.detach(), mask_bool),
        "token_weight_mean": active_weights.detach().mean() if active_weights.numel() else zero,
        "token_weight_max": active_weights.detach().max() if active_weights.numel() else zero,
        "lookahead_loss": loss.detach(),
        "proxy_loss": zero,
    }
    return loss, diagnostics
