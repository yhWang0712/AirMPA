import math
import os

import numpy as np
import torch

import config


_LATITUDE_WEIGHTS: np.ndarray | None = None


def _open_dataset(path: str):
    import xarray as xr

    if os.path.isdir(path):
        try:
            return xr.open_zarr(path, consolidated=True)
        except Exception:
            return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


def _read_lat_from_sample(sample_path: str) -> np.ndarray | None:
    lat_names = ["latitude", "lat", "LAT", "LATITUDE"]
    ds = None
    try:
        ds = _open_dataset(sample_path)
        for name in lat_names:
            if name in ds.coords or name in ds.variables:
                return ds[name].values
    except Exception:
        return None
    finally:
        if ds is not None:
            ds.close()
    return None


def _default_lat_grid(height: int) -> np.ndarray:
    step = 180.0 / height
    return (90.0 - step / 2.0) - np.arange(height, dtype=np.float32) * step


def _coerce_lat_grid(lat: np.ndarray | None, height: int, width: int) -> np.ndarray | None:
    if lat is None:
        return None
    lat = np.asarray(lat)
    if lat.ndim == 1:
        if lat.shape[0] != height:
            return None
        return np.repeat(lat.reshape(height, 1), width, axis=1)
    if lat.ndim == 2:
        if lat.shape != (height, width):
            return None
        return lat
    return None


def _build_latitude_weights_np(
    sample_path: str | None,
    height: int,
    width: int,
) -> np.ndarray | None:
    if not config.LAT_WEIGHTING:
        return None

    lat_grid = None
    if sample_path:
        lat_grid = _coerce_lat_grid(_read_lat_from_sample(sample_path), height, width)
    if lat_grid is None:
        lat_grid = _coerce_lat_grid(_default_lat_grid(height), height, width)
        if lat_grid is None:
            return None

    abs_max = float(np.nanmax(np.abs(lat_grid)))
    if abs_max <= math.pi + 1e-3:
        lat_rad = lat_grid
    else:
        lat_rad = np.deg2rad(lat_grid)

    weights = np.cos(lat_rad)
    weights = np.clip(weights, 0.0, None)
    if config.LAT_WEIGHT_POWER != 1.0:
        weights = weights ** float(config.LAT_WEIGHT_POWER)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if config.LAT_WEIGHT_NORMALIZE:
        weights = weights / (weights.mean() + config.LAT_WEIGHT_EPS)

    return weights.astype(np.float32)


def init_latitude_weights(sample_path: str | None, height: int, width: int) -> None:
    global _LATITUDE_WEIGHTS
    _LATITUDE_WEIGHTS = _build_latitude_weights_np(sample_path, height, width)


def get_latitude_weights(
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor | None:
    global _LATITUDE_WEIGHTS
    if _LATITUDE_WEIGHTS is None:
        _LATITUDE_WEIGHTS = _build_latitude_weights_np(None, height, width)
    if _LATITUDE_WEIGHTS is None:
        return None
    return torch.from_numpy(_LATITUDE_WEIGHTS).to(device=device, dtype=dtype).view(1, 1, height, width)
