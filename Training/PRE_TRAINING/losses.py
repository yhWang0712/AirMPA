import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import config
from lat_weights import get_latitude_weights

try:
    from pytorch_msssim import ms_ssim, ssim
except ImportError as exc:
    raise ImportError("pytorch_msssim is required for SSIM/MS-SSIM loss") from exc


_LOSS_CHANNEL_WEIGHTS: np.ndarray | None = None
_DEFAULT_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _data_range_from_target(target: torch.Tensor) -> float:
    return max(target.max().item() - target.min().item(), 1e-8)


def _ms_ssim_weights_for_shape(height: int, width: int, win_size: int) -> List[float] | None:
    min_hw = min(height, width)
    if min_hw < win_size:
        return None
    max_levels = int(np.floor(np.log2(min_hw / win_size))) + 1
    if max_levels < 2:
        return None
    num_levels = min(len(_DEFAULT_MS_SSIM_WEIGHTS), max_levels)
    weights = np.array(_DEFAULT_MS_SSIM_WEIGHTS[:num_levels], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    return weights.tolist()


def _ms_ssim_loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    win_size: int,
) -> torch.Tensor:
    weights = _ms_ssim_weights_for_shape(pred.shape[-2], pred.shape[-1], win_size)
    if weights is None:
        ssim_val = ssim(pred, target, data_range=data_range, size_average=True, win_size=win_size)
    else:
        ssim_val = ms_ssim(
            pred,
            target,
            data_range=data_range,
            size_average=True,
            weights=weights,
            win_size=win_size,
        )
    return 1 - ssim_val

def _load_stats_for_loss() -> Dict[str, Dict[str, float]]:
    stats_path: Path = config.STATS_FILE
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    return {}


def _label_keys_for_loss() -> List[str]:
    return [f"{var}_{config.LABEL_TIME}" for var in config.LABEL_VARS]


def _get_loss_channel_weights_np() -> np.ndarray | None:
    global _LOSS_CHANNEL_WEIGHTS
    if _LOSS_CHANNEL_WEIGHTS is not None:
        return _LOSS_CHANNEL_WEIGHTS

    method = config.LOSS_CHANNEL_WEIGHT_METHOD
    if method is None or str(method).lower() in {"none", "off"}:
        return None

    label_keys = _label_keys_for_loss()
    weights = np.ones(len(label_keys), dtype=np.float32)
    method_lower = str(method).lower()

    if method_lower == "manual":
        for i, key in enumerate(label_keys):
            base = key.rsplit("_", 1)[0]
            if key in config.LOSS_CHANNEL_WEIGHTS:
                weights[i] = float(config.LOSS_CHANNEL_WEIGHTS[key])
            elif base in config.LOSS_CHANNEL_WEIGHTS:
                weights[i] = float(config.LOSS_CHANNEL_WEIGHTS[base])
    else:
        stats = _load_stats_for_loss()
        for i, key in enumerate(label_keys):
            s = stats.get(key)
            if not s:
                continue
            if method_lower == "inverse_std":
                denom = float(s.get("std", 1.0))
            elif method_lower == "inverse_mean":
                denom = abs(float(s.get("mean", 1.0)))
            elif method_lower == "inverse_range":
                denom = float(s.get("max", 1.0)) - float(s.get("min", 0.0))
            else:
                continue
            denom = max(denom, config.LOSS_CHANNEL_WEIGHT_EPS)
            weights[i] = 1.0 / denom

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    if config.LOSS_CHANNEL_WEIGHT_NORMALIZE:
        weights = weights / (weights.mean() + config.LOSS_CHANNEL_WEIGHT_EPS)
    clip_min = config.LOSS_CHANNEL_WEIGHT_CLIP_MIN
    clip_max = config.LOSS_CHANNEL_WEIGHT_CLIP_MAX
    if clip_min is not None or clip_max is not None:
        weights = np.clip(
            weights,
            clip_min if clip_min is not None else -np.inf,
            clip_max if clip_max is not None else np.inf,
        )

    _LOSS_CHANNEL_WEIGHTS = weights
    return _LOSS_CHANNEL_WEIGHTS


def get_loss_channel_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    weights = _get_loss_channel_weights_np()
    if weights is None:
        return None
    return torch.from_numpy(weights).to(device=device, dtype=dtype).view(1, -1, 1, 1)


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    channel_weights: torch.Tensor | None = None,
    pixel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    if channel_weights is not None:
        loss = loss * channel_weights
    if pixel_weights is None:
        return torch.mean(loss)
    weighted = loss * pixel_weights
    denom = pixel_weights.sum() * loss.shape[1]
    return weighted.sum() / (denom + 1e-8)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.abs().sum() < 1e-8 or target.abs().sum() < 1e-8:
        return torch.tensor(0.0, device=pred.device)
    win_size = int(getattr(config, "SSIM_WIN_SIZE", 11))
    if channel_weights is None:
        if data_range is None:
            data_range = _data_range_from_target(target)
        return _ms_ssim_loss_value(pred, target, data_range, win_size)

    weights = channel_weights.view(-1)
    num_channels = pred.shape[1]
    if weights.numel() != num_channels:
        if data_range is None:
            data_range = _data_range_from_target(target)
        return _ms_ssim_loss_value(pred, target, data_range, win_size)

    losses = []
    for c in range(num_channels):
        if data_range is None:
            ch_range = _data_range_from_target(target[:, c : c + 1])
        else:
            ch_range = data_range
        losses.append(
            _ms_ssim_loss_value(pred[:, c : c + 1], target[:, c : c + 1], ch_range, win_size)
        )
    loss_vec = torch.stack(losses)
    weight_vec = weights / (weights.sum() + 1e-8)
    return (loss_vec * weight_vec).sum()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        mask = mask.unsqueeze(1)
        pred_masked = pred * mask
        target_masked = target * mask
    else:
        pred_masked = pred
        target_masked = target

    channel_weights = get_loss_channel_weights(pred.device, pred.dtype)
    if channel_weights is not None and channel_weights.shape[1] != pred.shape[1]:
        channel_weights = None

    lat_weights = get_latitude_weights(pred.device, pred.dtype, pred.shape[-2], pred.shape[-1])
    pixel_weights = mask if mask is not None else None
    if lat_weights is not None:
        pixel_weights = lat_weights if pixel_weights is None else pixel_weights * lat_weights

    charbonnier = charbonnier_loss(
        pred,
        target,
        epsilon=config.CHARBONNIER_EPSILON,
        channel_weights=channel_weights,
        pixel_weights=pixel_weights,
    )

    if config.SSIM_DATA_RANGE == "auto":
        data_range = None if channel_weights is not None else max(
            target_masked.max().item() - target_masked.min().item(), 1e-8
        )
    else:
        data_range = float(config.SSIM_DATA_RANGE)

    if config.LAT_WEIGHT_APPLY_TO_SSIM and lat_weights is not None:
        ssim_pred = pred_masked * lat_weights
        ssim_target = target_masked * lat_weights
    else:
        ssim_pred = pred_masked
        ssim_target = target_masked

    ssim_l = ssim_loss(ssim_pred, ssim_target, data_range=data_range, channel_weights=channel_weights)
    return config.LOSS_ALPHA * charbonnier + config.LOSS_BETA * ssim_l
