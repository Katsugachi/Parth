"""Validation-guarded direct weight editing utilities.

The transformations are intentionally conservative: finite-value repair, norm
rescaling, tiny-magnitude sparsification, and checkpoint interpolation. Every edit
must improve or match a held-out loss before being accepted.
"""

import copy
import json
from pathlib import Path

import torch


def repair_nonfinite(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    repaired = {}
    count = 0
    for name, tensor in state.items():
        value = tensor.clone()
        mask = ~torch.isfinite(value)
        count += int(mask.sum())
        value[mask] = 0
        repaired[name] = value
    return repaired, count


def rescale_to_rms(state: dict[str, torch.Tensor], target_rms: float = 0.02, include_suffixes=(".weight",)) -> dict[str, torch.Tensor]:
    edited = {}
    for name, tensor in state.items():
        value = tensor.clone()
        if tensor.ndim >= 2 and name.endswith(include_suffixes):
            rms = value.pow(2).mean().sqrt().clamp_min(1e-8)
            value *= (target_rms / rms).clamp(0.75, 1.33)
        edited[name] = value
    return edited


def sparsify_smallest(state: dict[str, torch.Tensor], fraction: float = 0.001) -> tuple[dict[str, torch.Tensor], int]:
    if not 0 <= fraction < 0.05:
        raise ValueError("fraction must be in [0, 0.05)")
    weights = [x.flatten().abs() for n, x in state.items() if x.ndim >= 2 and "embed" not in n]
    threshold = torch.quantile(torch.cat(weights).float(), fraction)
    edited = {}
    count = 0
    for name, tensor in state.items():
        value = tensor.clone()
        if tensor.ndim >= 2 and "embed" not in name:
            mask = value.abs() < threshold
            count += int(mask.sum())
            value[mask] = 0
        edited[name] = value
    return edited, count


def interpolate(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    if set(a) != set(b):
        raise ValueError("state dict keys differ")
    return {name: a[name] * (1 - alpha) + b[name] * alpha for name in a}


def load_state(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path))


def evaluate_accepted_edit(model, candidate: dict[str, torch.Tensor], validation_loss: float, loss_fn, tolerance: float = 0.0) -> tuple[bool, float]:
    original = copy.deepcopy(model.state_dict())
    model.load_state_dict(candidate, strict=False)
    loss = float(loss_fn(model))
    model.load_state_dict(original, strict=False)
    return loss <= validation_loss + tolerance, loss
