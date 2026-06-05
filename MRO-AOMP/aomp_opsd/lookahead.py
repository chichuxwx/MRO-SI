from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


@dataclass
class LookaheadStats:
    grad_norm: float = 0.0
    update_norm: float = 0.0
    updated_tensors: int = 0


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def tensor_list_norm(tensors: list[torch.Tensor | None]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        value = tensor.detach().float().norm().item()
        total += value * value
    return total ** 0.5


def is_lookahead_safe(trainer) -> bool:
    """Return whether in-place temporary trainable-parameter swaps are safe enough for v1."""

    if getattr(trainer, "is_fsdp_enabled", False):
        return False
    deepspeed_plugin = getattr(getattr(trainer, "accelerator", None), "state", None)
    deepspeed_plugin = getattr(deepspeed_plugin, "deepspeed_plugin", None)
    if deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", 0) == 3:
        return False
    return True


@contextmanager
def temporary_sgd_step(
    model: nn.Module,
    gradients: list[torch.Tensor | None],
    step_size: float,
) -> Iterator[LookaheadStats]:
    """Temporarily apply p <- p - step_size * grad and restore on exit."""

    params = trainable_parameters(model)
    if len(params) != len(gradients):
        raise ValueError("Gradient list must align with trainable model parameters.")

    saved: list[torch.Tensor | None] = []
    update_sq = 0.0
    updated = 0
    with torch.no_grad():
        for param, grad in zip(params, gradients):
            if grad is None:
                saved.append(None)
                continue
            saved.append(param.detach().clone())
            update = grad.detach().to(device=param.device, dtype=param.dtype) * float(step_size)
            param.add_(update, alpha=-1.0)
            update_norm = update.float().norm().item()
            update_sq += update_norm * update_norm
            updated += 1

    stats = LookaheadStats(
        grad_norm=tensor_list_norm(gradients),
        update_norm=update_sq ** 0.5,
        updated_tensors=updated,
    )
    try:
        yield stats
    finally:
        with torch.no_grad():
            for param, snapshot in zip(params, saved):
                if snapshot is not None:
                    param.copy_(snapshot)
