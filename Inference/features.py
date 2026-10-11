import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import config

_STATS: Dict[str, Dict[str, float]] | None = None
ALLOWED_BASE_HOURS = config.ALLOWED_BASE_HOURS


def apply_emission_preprocess(arr: np.ndarray) -> np.ndarray:
    if config.EMISSION_PREPROCESS.get("method") == "log1p":
        clip_min = config.EMISSION_PREPROCESS.get("clip_min", None)
        if clip_min is not None:
            arr = np.maximum(arr, clip_min)
        return np.log1p(arr)
    return arr


def _get_known_time_labels() -> List[str]:
    labels = set()
    for label_list in config.FEATURE_TIME_LABELS.values():
        labels.update(label_list)
    labels.add(config.LABEL_TIME)
    return sorted(labels, key=len, reverse=True)


_KNOWN_TIME_LABELS = _get_known_time_labels()


def _split_key(key: str) -> Tuple[str, str]:
    for label in _KNOWN_TIME_LABELS:
        suffix = f"_{label}"
        if key.endswith(suffix):
            return key[: -len(suffix)], label
    return key, ""


def is_allowed_base_hour(hour: int) -> bool:
    return hour in ALLOWED_BASE_HOURS


def is_emis_key(key: str) -> bool:
    base, _ = _split_key(key)
    return base in config.EMIS_VARS


def apply_value_clip(arr: np.ndarray, key: str) -> np.ndarray:
    clip_min = config.CLIP_MIN_VALUE
    if clip_min is None:
        return arr
    base, _ = _split_key(key)
    if base in config.CLIP_MIN_BASE_VARS:
        return np.maximum(arr, clip_min)
    return arr


def apply_feature_preprocess(arr: np.ndarray, key: str) -> np.ndarray:
    if is_emis_key(key):
        arr = apply_emission_preprocess(arr)
    return apply_value_clip(arr, key)


def iter_feature_keys():
    for group, vars_ in config.FEATURE_GROUPS.items():
        labels = config.FEATURE_TIME_LABELS.get(group, [])
        for var in vars_:
            for label in labels:
                yield f"{var}_{label}"


def get_feature_keys() -> List[str]:
    return list(iter_feature_keys())


def get_label_keys() -> List[str]:
    keys: List[str] = []
    for group, vars_ in config.LABEL_GROUPS.items():
        _ = group
        for var in vars_:
            keys.append(f"{var}_{config.LABEL_TIME}")
    return keys


def _load_stats() -> Dict[str, Dict[str, float]]:
    global _STATS
    if _STATS is not None:
        return _STATS
    stats_path: Path = config.STATS_FILE
    if stats_path.exists():
        with open(stats_path, "r") as f:
            _STATS = json.load(f)
    else:
        _STATS = {}
    return _STATS


def normalize(arr: np.ndarray, method: str | None = None, key: str | None = None) -> np.ndarray:
    method = method or (config.NORMALIZE_METHOD if config.NORMALIZE else None)
    if method is None:
        return arr.astype(np.float32)

    nan_mask = np.isnan(arr)

    if config.NORMALIZE and config.NORMALIZE_SCOPE == "dataset" and key is not None:
        stats = _load_stats()
        if key in stats:
            s = stats[key]
            if config.NORMALIZE_METHOD == "zscore":
                mean = float(s.get("mean", 0.0))
                std = float(s.get("std", 1.0)) + 1e-6
                normalized = (arr - mean) / std
            elif config.NORMALIZE_METHOD == "minmax":
                mn = float(s.get("min", 0.0))
                mx = float(s.get("max", 1.0))
                normalized = ((arr - mn) / (mx - mn + 1e-6)) * 2 - 1
            else:
                normalized = arr

            normalized = np.where(nan_mask, 0.0, normalized)
            return normalized.astype(np.float32)

    if method == "zscore":
        mean = np.nanmean(arr)
        std = np.nanstd(arr) + 1e-6
        normalized = (arr - mean) / std
    elif method == "minmax":
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        normalized = ((arr - mn) / (mx - mn + 1e-6)) * 2 - 1
    else:
        normalized = arr

    normalized = np.where(nan_mask, 0.0, normalized)
    return normalized.astype(np.float32)


def normalize_label(arr: np.ndarray, key: str) -> np.ndarray:
    if not config.LABEL_NORMALIZE:
        arr = np.where(np.isnan(arr), 0.0, arr)
        return arr.astype(np.float32)

    method = config.LABEL_NORMALIZE_METHOD
    nan_mask = np.isnan(arr)

    if config.LABEL_NORMALIZE_SCOPE == "dataset":
        stats = _load_stats()
        if key in stats:
            s = stats[key]
            if method == "zscore":
                mean = float(s.get("mean", 0.0))
                std = float(s.get("std", 1.0)) + 1e-6
                normalized = (arr - mean) / std
            elif method == "minmax":
                mn = float(s.get("min", 0.0))
                mx = float(s.get("max", 1.0))
                normalized = ((arr - mn) / (mx - mn + 1e-6)) * 2 - 1
            else:
                normalized = arr
            normalized = np.where(nan_mask, 0.0, normalized)
            return normalized.astype(np.float32)

    if method == "zscore":
        mean = np.nanmean(arr)
        std = np.nanstd(arr) + 1e-6
        normalized = (arr - mean) / std
    elif method == "minmax":
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        normalized = ((arr - mn) / (mx - mn + 1e-6)) * 2 - 1
    else:
        normalized = arr

    normalized = np.where(nan_mask, 0.0, normalized)
    return normalized.astype(np.float32)


def denormalize_label(arr: np.ndarray, key: str) -> np.ndarray:
    if not config.LABEL_NORMALIZE:
        return arr.astype(np.float32)

    method = config.LABEL_NORMALIZE_METHOD
    stats = _load_stats()
    if key not in stats:
        return arr.astype(np.float32)

    s = stats[key]
    if method == "zscore":
        mean = float(s.get("mean", 0.0))
        std = float(s.get("std", 1.0)) + 1e-6
        return (arr * std + mean).astype(np.float32)
    if method == "minmax":
        mn = float(s.get("min", 0.0))
        mx = float(s.get("max", 1.0))
        return (((arr + 1) / 2) * (mx - mn + 1e-6) + mn).astype(np.float32)
    return arr.astype(np.float32)


def stack_channels(sample: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, List[str]]:
    h, w = config.HEIGHT, config.WIDTH
    channels: List[np.ndarray] = []
    names: List[str] = []

    for group, vars_ in config.FEATURE_GROUPS.items():
        labels = config.FEATURE_TIME_LABELS.get(group, [])
        for var in vars_:
            for label in labels:
                key = f"{var}_{label}"
                if key not in sample:
                    continue
                arr = sample[key]
                if group == "temporal":
                    arr = arr.astype(np.float32)
                else:
                    arr = apply_feature_preprocess(arr, key)
                    arr = normalize(arr, key=key)
                channels.append(arr)
                names.append(key)

    if not channels:
        raise ValueError("No input channels found; check config.FEATURE_GROUPS")

    x = np.stack(channels, axis=0)
    assert x.shape[1:] == (h, w), f"Expected spatial {(h, w)}, got {x.shape[1:]}"
    return torch.from_numpy(x), names


def stack_channels_no_norm(sample: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, List[str]]:
    h, w = config.HEIGHT, config.WIDTH
    channels: List[np.ndarray] = []
    names: List[str] = []

    for group, vars_ in config.FEATURE_GROUPS.items():
        labels = config.FEATURE_TIME_LABELS.get(group, [])
        for var in vars_:
            for label in labels:
                key = f"{var}_{label}"
                if key in sample:
                    channels.append(sample[key].astype(np.float32))
                    names.append(key)

    if not channels:
        raise ValueError("No input channels found; check config.FEATURE_GROUPS")

    x = np.stack(channels, axis=0)
    assert x.shape[1:] == (h, w), f"Expected spatial {(h, w)}, got {x.shape[1:]}"
    return torch.from_numpy(x), names


def build_label(sample: Dict[str, np.ndarray]) -> torch.Tensor:
    targets = []
    for var in config.LABEL_VARS:
        key = f"{var}_{config.LABEL_TIME}"
        if key not in sample:
            raise KeyError(f"Missing label {key} in sample")
        arr = sample[key].astype(np.float32)
        arr = apply_value_clip(arr, key)
        arr = normalize_label(arr, key)
        targets.append(arr)
    y = np.stack(targets, axis=0)
    return torch.from_numpy(y)


def get_eval_masks(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    masks = {}
    for m in config.EVAL_MASKS:
        if m in sample:
            masks[m] = torch.from_numpy(sample[m].astype(np.float32))
    return masks
