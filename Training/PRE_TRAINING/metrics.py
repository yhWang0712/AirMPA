import math
from typing import Dict, List, Tuple

import numpy as np
import torch

import config
from lat_weights import get_latitude_weights


METRIC_KEYS = ("r2", "rmse", "mae", "nmb", "nme")


def get_label_channel_names(include_group: bool = False, include_time: bool = False) -> List[str]:
    group_lookup: Dict[str, str] = {}
    for group, vars_ in config.LABEL_GROUPS.items():
        for var in vars_:
            group_lookup[var] = group

    names: List[str] = []
    for var in config.LABEL_VARS:
        label = f"{var}_{config.LABEL_TIME}" if include_time else var
        if include_group and var in group_lookup:
            names.append(f"{group_lookup[var]}/{label}")
        else:
            names.append(label)
    return names


def init_metric_sums(num_channels: int, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
    zeros = torch.zeros(num_channels, device=device, dtype=torch.float64)
    return {
        "sum_w": zeros.clone(),
        "sum_y": zeros.clone(),
        "sum_y_sq": zeros.clone(),
        "sum_diff": zeros.clone(),
        "sum_abs_diff": zeros.clone(),
        "sum_sq_diff": zeros.clone(),
    }


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and target.shape[1] != 1:
        mask = mask.expand(-1, target.shape[1], -1, -1)
    return mask


def update_metric_sums(
    stats: Dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    preds = _ensure_4d(preds)
    targets = _ensure_4d(targets)
    if preds.numel() == 0:
        return

    valid = torch.isfinite(preds) & torch.isfinite(targets)
    if not valid.any():
        return

    weights = valid.to(preds.dtype)
    if mask is not None:
        mask = _expand_mask(mask, preds).to(device=preds.device, dtype=preds.dtype)
        weights = weights * mask

    lat_weights = get_latitude_weights(
        preds.device, preds.dtype, preds.shape[-2], preds.shape[-1]
    )
    if lat_weights is not None:
        weights = weights * lat_weights

    weights = weights.to(dtype=torch.float64)
    preds_f = preds.to(dtype=torch.float64)
    targets_f = targets.to(dtype=torch.float64)
    diff = preds_f - targets_f

    dims = (0, 2, 3)
    stats["sum_w"] += weights.sum(dim=dims)
    stats["sum_y"] += (targets_f * weights).sum(dim=dims)
    stats["sum_y_sq"] += (targets_f * targets_f * weights).sum(dim=dims)
    stats["sum_diff"] += (diff * weights).sum(dim=dims)
    stats["sum_abs_diff"] += (diff.abs() * weights).sum(dim=dims)
    stats["sum_sq_diff"] += (diff * diff * weights).sum(dim=dims)


def _metrics_from_sums(
    sum_w: float,
    sum_y: float,
    sum_y_sq: float,
    sum_diff: float,
    sum_abs_diff: float,
    sum_sq_diff: float,
) -> Dict[str, float]:
    if sum_w <= 0:
        return {k: float("nan") for k in METRIC_KEYS}

    mae = sum_abs_diff / sum_w
    rmse = math.sqrt(sum_sq_diff / sum_w)
    ss_tot = sum_y_sq - (sum_y * sum_y) / sum_w
    r2 = 1 - (sum_sq_diff / ss_tot) if ss_tot != 0 else float("nan")

    if sum_y == 0:
        nmb = float("nan")
        nme = float("nan")
    else:
        nmb = sum_diff / sum_y
        nme = sum_abs_diff / sum_y

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "nmb": nmb,
        "nme": nme,
    }


def finalize_metric_sums(
    stats: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    channel_metrics = {k: [] for k in METRIC_KEYS}
    num_channels = stats["sum_w"].numel()

    for i in range(num_channels):
        m = _metrics_from_sums(
            float(stats["sum_w"][i].item()),
            float(stats["sum_y"][i].item()),
            float(stats["sum_y_sq"][i].item()),
            float(stats["sum_diff"][i].item()),
            float(stats["sum_abs_diff"][i].item()),
            float(stats["sum_sq_diff"][i].item()),
        )
        for k in channel_metrics.keys():
            channel_metrics[k].append(m[k])

    overall = _metrics_from_sums(
        float(stats["sum_w"].sum().item()),
        float(stats["sum_y"].sum().item()),
        float(stats["sum_y_sq"].sum().item()),
        float(stats["sum_diff"].sum().item()),
        float(stats["sum_abs_diff"].sum().item()),
        float(stats["sum_sq_diff"].sum().item()),
    )

    return channel_metrics, overall


def compute_metrics(
    preds: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    preds_t = torch.as_tensor(preds)
    targets_t = torch.as_tensor(targets)
    mask_t = torch.as_tensor(mask) if mask is not None else None

    preds_t = _ensure_4d(preds_t)
    targets_t = _ensure_4d(targets_t)

    stats = init_metric_sums(preds_t.shape[1], device=preds_t.device)
    update_metric_sums(stats, preds_t, targets_t, mask=mask_t)
    return finalize_metric_sums(stats)
