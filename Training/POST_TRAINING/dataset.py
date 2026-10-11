import argparse
import calendar
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from features import (
    apply_feature_preprocess,
    build_label,
    get_eval_masks,
    get_feature_keys,
    get_label_keys,
    stack_channels,
)


def _label_to_offset(label: str) -> int:
    if label == "t0":
        return 0
    m = re.match(r"t_minus_(\d+)", label)
    if m:
        return -int(m.group(1))
    m = re.match(r"t_plus_(\d+)", label)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unrecognized time label: {label}")


def _get_known_time_labels() -> List[str]:
    labels = set()
    for label_list in config.FEATURE_TIME_LABELS.values():
        labels.update(label_list)
    labels.add(config.LABEL_TIME)
    return sorted(labels, key=len, reverse=True)


_KNOWN_TIME_LABELS = _get_known_time_labels()


def _split_key_time_label(key: str) -> Tuple[str, str]:
    for label in _KNOWN_TIME_LABELS:
        suffix = f"_{label}"
        if key.endswith(suffix):
            return key[: -len(suffix)], label
    return key, ""


def _np64_to_datetime(value: np.datetime64) -> datetime:
    ts = value.astype("datetime64[s]").astype(int)
    return datetime.utcfromtimestamp(ts)


def _compute_lon_offsets(lon: np.ndarray) -> np.ndarray:
    offsets = np.floor((lon + 7.5) / 15.0)
    offsets = np.mod(offsets, 24)
    return offsets.astype(np.float32)


def _temporal_features_for_time(
    dt: datetime, lon_offsets: np.ndarray, height: int, width: int
) -> Dict[str, np.ndarray]:
    year_days = 366 if calendar.isleap(dt.year) else 365
    doy = dt.timetuple().tm_yday
    doy_angle = 2.0 * np.pi * ((doy - 1) / year_days)
    day_sin = np.sin(doy_angle).astype(np.float32)
    day_cos = np.cos(doy_angle).astype(np.float32)

    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    local_hour = (utc_hour + lon_offsets) % 24.0
    hour_angle = 2.0 * np.pi * (local_hour / 24.0)
    hour_sin = np.sin(hour_angle).astype(np.float32)
    hour_cos = np.cos(hour_angle).astype(np.float32)

    day_sin_arr = np.full((height, width), day_sin, dtype=np.float32)
    day_cos_arr = np.full((height, width), day_cos, dtype=np.float32)
    hour_sin_arr = np.broadcast_to(hour_sin, (height, width)).astype(np.float32)
    hour_cos_arr = np.broadcast_to(hour_cos, (height, width)).astype(np.float32)

    return {
        "day_of_year_sin": day_sin_arr,
        "day_of_year_cos": day_cos_arr,
        "hour_of_day_sin": hour_sin_arr,
        "hour_of_day_cos": hour_cos_arr,
    }


def _resolve_channel_name(base_var: str, channel_to_index: Dict[str, int]) -> str:
    if base_var in channel_to_index:
        return base_var
    for group, vars_ in config.FEATURE_GROUPS.items():
        if group == "temporal":
            continue
        if base_var in vars_:
            candidate = f"{group}_{base_var}"
            if candidate in channel_to_index:
                return candidate
    raise KeyError(f"Channel not found for variable: {base_var}")



class ZarrTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        years: List[int] | None = None,
        stack_fn: Callable[[Dict[str, np.ndarray]], Tuple[torch.Tensor, List[str]]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        allowed_hours: set[int] | None = None,
        invalid_time_indices: Iterable[int] | None = None,
    ):
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.allowed_hours = allowed_hours or config.ALLOWED_BASE_HOURS
        self.years = years

        if stack_fn is None:
            self.stack_fn = stack_channels
        else:
            self.stack_fn = stack_fn

        self.feature_keys = get_feature_keys()
        self.label_keys = get_label_keys()
        self.mask_keys = list(config.EVAL_MASKS)
        self.required_keys = list(dict.fromkeys(self.feature_keys + self.label_keys))

        self.temporal_vars = set(config.FEATURE_GROUPS.get("temporal", []))
        self.temporal_labels = config.FEATURE_TIME_LABELS.get("temporal", [])
        self.label_offsets = {label: _label_to_offset(label) for label in _KNOWN_TIME_LABELS}

        self._ds: xr.Dataset | None = None
        self._open_dataset()
        if "data" not in self._ds.data_vars:
            raise ValueError("Zarr dataset missing 'data' variable")
        if "channel" not in self._ds.coords or "time" not in self._ds.coords:
            raise ValueError("Zarr dataset must contain 'channel' and 'time' coordinates")

        self.channel_names = [str(v) for v in self._ds["channel"].values.tolist()]
        self.time_values = self._ds["time"].values.astype("datetime64[ns]")
        lon_name = "lon" if "lon" in self._ds.coords else "longitude"
        lat_name = "lat" if "lat" in self._ds.coords else "latitude"
        if lon_name not in self._ds.coords or lat_name not in self._ds.coords:
            raise ValueError("Zarr dataset missing lat/lon coordinates")
        self.lon = self._ds[lon_name].values.astype(np.float32)
        self.lat = self._ds[lat_name].values.astype(np.float32)
        self.height = int(self.lat.size)
        self.width = int(self.lon.size)
        if (self.height, self.width) != (config.HEIGHT, config.WIDTH):
            raise ValueError(
                f"Zarr spatial size {(self.height, self.width)} != {(config.HEIGHT, config.WIDTH)}"
            )

        self.lon_offsets = _compute_lon_offsets(self.lon)
        self.channel_to_index = {name: idx for idx, name in enumerate(self.channel_names)}
        self.channel_cache: Dict[str, str] = {}
        self.time_datetimes = [_np64_to_datetime(t) for t in self.time_values]
        self.base_time_indices = self._build_base_indices()

        _ = invalid_time_indices
        self._close_dataset()

        if not self.base_time_indices:
            raise FileNotFoundError("No valid time steps found for the given filters")

    def _open_dataset(self) -> None:
        if self._ds is not None:
            return
        try:
            self._ds = xr.open_zarr(self.data_dir, consolidated=True)
        except Exception:
            self._ds = xr.open_zarr(self.data_dir, consolidated=False)

    def _close_dataset(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ds"] = None
        return state

    def _resolve_channel_name(self, base_var: str) -> str:
        if base_var in self.channel_cache:
            return self.channel_cache[base_var]
        resolved = _resolve_channel_name(base_var, self.channel_to_index)
        self.channel_cache[base_var] = resolved
        return resolved

    def _passes_filters(self, dt: datetime) -> bool:
        if self.years is not None and dt.year not in self.years:
            return False
        if self.start_date:
            if dt.date() < datetime.strptime(self.start_date, "%Y-%m-%d").date():
                return False
        if self.end_date:
            if dt.date() > datetime.strptime(self.end_date, "%Y-%m-%d").date():
                return False
        if self.allowed_hours is not None and dt.hour not in self.allowed_hours:
            return False
        return True

    def _build_base_indices(self) -> List[int]:
        offsets = set()
        for labels in config.FEATURE_TIME_LABELS.values():
            offsets.update(_label_to_offset(label) for label in labels)
        offsets.add(_label_to_offset(config.LABEL_TIME))
        min_off = min(offsets)
        max_off = max(offsets)

        indices: List[int] = []
        for i, dt in enumerate(self.time_datetimes):
            if not self._passes_filters(dt):
                continue
            if i + min_off < 0 or i + max_off >= len(self.time_datetimes):
                continue
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.base_time_indices)

    def __getitem__(self, idx: int):
        base_idx = self.base_time_indices[idx]
        self._open_dataset()
        data = self._ds["data"]

        sample: Dict[str, np.ndarray] = {}
        temporal_cache: Dict[int, Dict[str, np.ndarray]] = {}

        for key in self.required_keys:
            base_var, label = _split_key_time_label(key)
            if label == "":
                continue
            offset = self.label_offsets[label]
            time_idx = base_idx + offset
            if base_var in self.temporal_vars:
                if time_idx not in temporal_cache:
                    dt = self.time_datetimes[time_idx]
                    temporal_cache[time_idx] = _temporal_features_for_time(
                        dt, self.lon_offsets, self.height, self.width
                    )
                sample[key] = temporal_cache[time_idx][base_var]
                continue

            channel_name = self._resolve_channel_name(base_var)
            channel_idx = self.channel_to_index[channel_name]
            arr = data.isel(channel=channel_idx, time=time_idx).values.astype(np.float32)
            sample[key] = arr

        x, channel_names = self.stack_fn(sample)
        y = build_label(sample)
        masks = get_eval_masks(sample)

        base_dt = self.time_datetimes[base_idx]
        date_str = base_dt.strftime("%Y-%m-%d")
        datetime_str = base_dt.strftime("%Y%m%d%H")

        if getattr(config, "LON_ROLL", 0):
            shift = int(config.LON_ROLL)
            x = torch.roll(x, shifts=shift, dims=-1)
            y = torch.roll(y, shifts=shift, dims=-1)
            for k in masks:
                masks[k] = torch.roll(masks[k], shifts=shift, dims=-1)

        out = {
            "x": x,
            "y": y,
            "channel_names": channel_names,
            "masks": masks,
            "date": date_str,
            "datetime": datetime_str,
        }
        return out


def _open_zarr_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def _filter_time_indices(
    time_datetimes: List[datetime],
    years: List[int] | None,
    start_date: str | None,
    end_date: str | None,
    allowed_hours: set[int] | None,
) -> List[int]:
    start_d = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_d = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    indices: List[int] = []
    for i, dt in enumerate(time_datetimes):
        if years is not None and dt.year not in years:
            continue
        if start_d is not None and dt.date() < start_d:
            continue
        if end_d is not None and dt.date() > end_d:
            continue
        if allowed_hours is not None and dt.hour not in allowed_hours:
            continue
        indices.append(i)
    return indices


def compute_dataset_stats(
    data_dir: str,
    stats_out: Path,
    years: List[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    n_workers: int | None = None,
    invalid_time_indices: Iterable[int] | None = None,
):
    _ = n_workers
    ds = _open_zarr_dataset(data_dir)
    if "data" not in ds.data_vars:
        raise ValueError("Zarr dataset missing 'data' variable")
    if "channel" not in ds.coords or "time" not in ds.coords:
        raise ValueError("Zarr dataset must contain 'channel' and 'time' coordinates")

    channel_names = [str(v) for v in ds["channel"].values.tolist()]
    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}
    time_values = ds["time"].values.astype("datetime64[ns]")
    time_datetimes = [_np64_to_datetime(t) for t in time_values]
    time_indices = _filter_time_indices(
        time_datetimes, years, start_date, end_date, config.ALLOWED_BASE_HOURS
    )
    if not time_indices:
        ds.close()
        raise FileNotFoundError("No time steps found for stats computation")

    required_keys = list(dict.fromkeys(get_feature_keys() + get_label_keys()))
    temporal_vars = set(config.FEATURE_GROUPS.get("temporal", []))

    _ = invalid_time_indices
    base_to_keys: Dict[str, List[str]] = {}
    for key in required_keys:
        base_var, _ = _split_key_time_label(key)
        if base_var in temporal_vars:
            continue
        base_to_keys.setdefault(base_var, []).append(key)

    data = ds["data"]
    base_infos: Dict[str, Tuple[int, str]] = {}
    running_stats: Dict[str, Dict[str, float]] = {}
    for base_var, keys in base_to_keys.items():
        channel_name = _resolve_channel_name(base_var, channel_to_index)
        channel_idx = channel_to_index[channel_name]
        base_infos[base_var] = (channel_idx, keys[0])
        running_stats[base_var] = {
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
        }

    spatial_dims = [dim for dim in data.dims if dim not in ("time", "channel")]
    if not spatial_dims:
        ds.close()
        raise ValueError(f"Zarr dataset missing spatial dims in {data.dims}")

    print(f"Computing stats for {len(base_to_keys)} channels over {len(time_indices)} times...")
    for time_idx in tqdm(time_indices, desc="Times"):
        slice_arr = (
            data.isel(time=time_idx)
            .transpose("channel", *spatial_dims)
            .values.astype(np.float32)
        )
        for base_var, (channel_idx, preprocess_key) in base_infos.items():
            arr = apply_feature_preprocess(slice_arr[channel_idx], preprocess_key)
            valid = np.isfinite(arr)
            vals = arr[valid].reshape(-1)
            if vals.size == 0:
                continue
            running = running_stats[base_var]
            running["sum"] += float(vals.sum())
            running["sumsq"] += float((vals * vals).sum())
            running["count"] += float(vals.size)
            running["min"] = min(running["min"], float(vals.min()))
            running["max"] = max(running["max"], float(vals.max()))

    base_stats: Dict[str, Dict[str, float]] = {}
    for base_var, running in running_stats.items():
        if running["count"] <= 0:
            continue
        mean = running["sum"] / running["count"]
        var = max(0.0, (running["sumsq"] / running["count"]) - mean * mean)
        std = (var ** 0.5) + 1e-6
        base_stats[base_var] = {"mean": mean, "std": std, "min": running["min"], "max": running["max"]}

    out: Dict[str, Dict[str, float]] = {}
    for base_var, keys in base_to_keys.items():
        if base_var not in base_stats:
            continue
        for key in keys:
            out[key] = base_stats[base_var]

    ds.close()
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved stats to {stats_out} with {len(out)} keys")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-stats", action="store_true", help="Compute dataset-level stats")
    parser.add_argument("--start-date", type=str, default=config.TRAIN_START_DATE)
    parser.add_argument("--end-date", type=str, default=config.TRAIN_END_DATE)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--out", type=str, default=str(config.STATS_FILE))
    args = parser.parse_args()

    if args.compute_stats:
        compute_dataset_stats(
            data_dir=str(config.DATA_DIR),
            stats_out=Path(args.out),
            start_date=args.start_date,
            end_date=args.end_date,
            n_workers=args.n_workers,
        )
