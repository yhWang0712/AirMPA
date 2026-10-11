import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

import config
from dataset import ZarrTimeSeriesDataset
from features import denormalize_label
from lat_weights import init_latitude_weights
from metrics import finalize_metric_sums, get_label_channel_names, init_metric_sums, update_metric_sums
from models.swin import SwinTiny2D


def _open_dataset(path: str) -> xr.Dataset:
    if os.path.isdir(path):
        try:
            return xr.open_zarr(path, consolidated=True)
        except Exception:
            return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


def get_spatial_coords_from_zarr(data_dir: str) -> Dict[str, np.ndarray]:
    try:
        ds = _open_dataset(data_dir)

        lat_names = ["latitude", "lat", "LAT", "LATITUDE"]
        lon_names = ["longitude", "lon", "LON", "LONGITUDE"]

        lat = None
        lon = None

        for name in lat_names:
            if name in ds.coords or name in ds.variables:
                lat = ds[name].values
                break

        for name in lon_names:
            if name in ds.coords or name in ds.variables:
                lon = ds[name].values
                break

        if lat is None or lon is None:
            raise ValueError("Latitude or longitude not found in sample dataset")

        ds.close()
        return {"lat": lat, "lon": lon}
    except Exception as exc:
        raise RuntimeError(f"Failed to read spatial coords from {data_dir}: {exc}")




def _apply_masks(
    pred: torch.Tensor,
    target: torch.Tensor,
    masks: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = None
    for mname in config.EVAL_MASKS:
        if mname in masks:
            m = masks[mname].to(pred.device)
            mask = m if mask is None else (mask * m)

    if mask is None:
        return pred, target

    while mask.dim() < target.dim():
        mask = mask.unsqueeze(1)

    return pred * mask, target * mask


def _build_eval_mask(masks: Dict[str, torch.Tensor]) -> torch.Tensor | None:
    mask = None
    for mname in config.EVAL_MASKS:
        if mname in masks:
            m = masks[mname]
            mask = m if mask is None else (mask * m)
    return mask


def _maybe_denormalize(
    preds: torch.Tensor,
    targets: torch.Tensor,
    enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not enabled or not config.LABEL_NORMALIZE or config.LABEL_NORMALIZE_SCOPE != "dataset":
        return preds, targets

    preds_np = preds.numpy()
    targets_np = targets.numpy()
    for i, var in enumerate(config.LABEL_VARS):
        key = f"{var}_{config.LABEL_TIME}"
        preds_np[:, i, ...] = denormalize_label(preds_np[:, i, ...], key)
        targets_np[:, i, ...] = denormalize_label(targets_np[:, i, ...], key)
    return torch.from_numpy(preds_np), torch.from_numpy(targets_np)




def _build_time_coords(dates: List[str] | None, length: int) -> np.ndarray:
    if not dates:
        return np.arange(length)
    try:
        return np.array(dates, dtype="datetime64[ns]")
    except Exception:
        return np.array(dates)


def _safe_tag(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return cleaned or "sample"


def _unique_tag(base: str, seen: set[str]) -> str:
    if base not in seen:
        seen.add(base)
        return base
    idx = 1
    while f"{base}_{idx}" in seen:
        idx += 1
    tag = f"{base}_{idx}"
    seen.add(tag)
    return tag


def _datetime_tag(
    date_value: str | None,
    datetime_value: str | None,
    sample_idx: int,
    seen: set[str],
) -> str:
    candidate = None
    if datetime_value:
        digits = re.sub(r"\D", "", datetime_value)
        if len(digits) >= 10:
            candidate = digits[:10]
        elif len(digits) >= 8:
            candidate = f"{digits[:8]}00"
    if candidate is None and date_value:
        digits = re.sub(r"\D", "", date_value)
        if len(digits) >= 8:
            candidate = f"{digits[:8]}00"
    if candidate is None:
        candidate = f"sample_{sample_idx:06d}"
    return _unique_tag(candidate, seen)


def _range_filename(start_date: str | None, end_date: str | None) -> str:
    if start_date and end_date:
        return f"{_safe_tag(start_date)}_{_safe_tag(end_date)}.nc"
    if start_date:
        return f"{_safe_tag(start_date)}_end.nc"
    if end_date:
        return f"start_{_safe_tag(end_date)}.nc"
    return "predictions.nc"


def _print_metrics(
    channel_metrics: Dict[str, List[float]],
    overall_metrics: Dict[str, float],
    channel_names: List[str],
) -> None:
    num_channels = len(channel_metrics["r2"])

    print("")
    print("========== Overall Metrics ==========")
    print(f"  R2   = {overall_metrics['r2']:.6f}")
    print(f"  RMSE = {overall_metrics['rmse']:.6f}")
    print(f"  MAE  = {overall_metrics['mae']:.6f}")
    print(f"  NMB  = {overall_metrics['nmb']:.6f}")
    print(f"  NME  = {overall_metrics['nme']:.6f}")
    print("")

    print("========== Per-Channel Metrics ==========")
    for c in range(num_channels):
        name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
        print(f"{name}:")
        print(f"  R2   = {channel_metrics['r2'][c]:.6f}")
        print(f"  RMSE = {channel_metrics['rmse'][c]:.6f}")
        print(f"  MAE  = {channel_metrics['mae'][c]:.6f}")
        print(f"  NMB  = {channel_metrics['nmb'][c]:.6f}")
        print(f"  NME  = {channel_metrics['nme'][c]:.6f}")
        print("")


def save_as_netcdf(
    predictions: np.ndarray,
    labels: np.ndarray,
    diff: np.ndarray,
    dates: List[str] | None,
    output_dir: str,
    spatial_coords: Dict[str, np.ndarray],
    channel_names: List[str],
    filename: str = "predictions.nc",
) -> str:
    n_samples, n_channels, height, width = predictions.shape

    lat = spatial_coords["lat"]
    lon = spatial_coords["lon"]

    if lat.ndim == 1:
        if len(lat) != height:
            raise ValueError("Latitude dimension mismatch")
    elif lat.ndim == 2:
        if lat.shape != (height, width):
            raise ValueError("Latitude shape mismatch")
    else:
        raise ValueError("Unsupported latitude dimensions")

    if lon.ndim == 1:
        if len(lon) != width:
            raise ValueError("Longitude dimension mismatch")
    elif lon.ndim == 2:
        if lon.shape != (height, width):
            raise ValueError("Longitude shape mismatch")
    else:
        raise ValueError("Unsupported longitude dimensions")

    time_coords = _build_time_coords(dates, n_samples)

    if lat.ndim == 1 and lon.ndim == 1:
        coords = {
            "time": time_coords,
            "channel": channel_names,
            "lat": lat,
            "lon": lon,
        }
        data_vars = {
            "prediction": (["time", "channel", "lat", "lon"], predictions),
            "label": (["time", "channel", "lat", "lon"], labels),
            "diff": (["time", "channel", "lat", "lon"], diff),
        }
    else:
        y_dim = np.arange(height)
        x_dim = np.arange(width)
        coords = {
            "time": time_coords,
            "channel": channel_names,
            "y": y_dim,
            "x": x_dim,
        }
        data_vars = {
            "prediction": (["time", "channel", "y", "x"], predictions),
            "label": (["time", "channel", "y", "x"], labels),
            "diff": (["time", "channel", "y", "x"], diff),
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
        }

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["description"] = "Model predictions and comparisons"
    ds.attrs["created_at"] = datetime.now().isoformat()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    ds.to_netcdf(output_path)
    return output_path


def predict_date_range(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    spatial_coords: Dict[str, np.ndarray],
    denorm: bool,
    save_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> Dict[str, np.ndarray | List[str] | None]:
    model.eval()

    save_per_sample = save_mode == "per-sample"
    channel_names = [f"{var}_{config.LABEL_TIME}" for var in config.LABEL_VARS]

    all_preds = [] if not save_per_sample else None
    all_labels = [] if not save_per_sample else None
    all_dates: List[str] = []
    saved_files: List[str] = []
    used_tags: set[str] = set()
    sample_idx = 0
    stats = None

    with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", ncols=100) as pbar:
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            mask = _build_eval_mask(batch.get("masks", {}))

            pred_metric = pred.detach().cpu()
            y_metric = y.detach().cpu()
            pred_metric, y_metric = _maybe_denormalize(pred_metric, y_metric, denorm)

            pred, y = _apply_masks(pred, y, batch.get("masks", {}))
            pred = pred.detach().cpu()
            y = y.detach().cpu()
            pred, y = _maybe_denormalize(pred, y, denorm)

            if stats is None:
                stats = init_metric_sums(pred_metric.shape[1], device=pred_metric.device)
            mask_cpu = mask.detach().cpu() if mask is not None else None
            update_metric_sums(stats, pred_metric, y_metric, mask=mask_cpu)

            dates = batch.get("date")
            datetimes = batch.get("datetime")
            batch_size = pred.shape[0]
            date_list: List[str | None]
            datetime_list: List[str | None]
            if dates is None:
                date_list = [None] * batch_size
            elif isinstance(dates, (list, tuple)):
                date_list = [str(d) for d in dates]
            else:
                date_list = [str(dates)] * batch_size
            if datetimes is None:
                datetime_list = [None] * batch_size
            elif isinstance(datetimes, (list, tuple)):
                datetime_list = [str(d) for d in datetimes]
            else:
                datetime_list = [str(datetimes)] * batch_size

            if len(date_list) != batch_size:
                date_list = [None] * batch_size
            if len(datetime_list) != batch_size:
                datetime_list = [None] * batch_size

            if save_per_sample:
                for i in range(batch_size):
                    pred_i = pred[i].numpy()
                    label_i = y[i].numpy()
                    diff_i = pred_i - label_i
                    date_i = date_list[i]
                    datetime_i = datetime_list[i]
                    tag = _datetime_tag(date_i, datetime_i, sample_idx, used_tags)
                    filename = f"prediction_{tag}.nc"
                    output_path = save_as_netcdf(
                        np.expand_dims(pred_i, axis=0),
                        np.expand_dims(label_i, axis=0),
                        np.expand_dims(diff_i, axis=0),
                        [date_i] if date_i else None,
                        output_dir,
                        spatial_coords,
                        channel_names,
                        filename=filename,
                    )
                    saved_files.append(output_path)
                    sample_idx += 1
            else:
                all_preds.append(pred)
                all_labels.append(y)
                all_dates.extend([d for d in date_list if d is not None])

    if save_per_sample:
        predictions = None
        labels_np = None
        diff = None
    else:
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        predictions = preds.numpy()
        labels_np = labels.numpy()
        diff = predictions - labels_np

        if all_dates and len(all_dates) != predictions.shape[0]:
            all_dates = []

    if stats is None:
        empty = {
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "nmb": float("nan"),
            "nme": float("nan"),
        }
        channel_metrics, overall_metrics = {k: [] for k in empty.keys()}, empty
    else:
        channel_metrics, overall_metrics = finalize_metric_sums(stats)

    display_names = get_label_channel_names(include_group=False, include_time=False)
    _print_metrics(channel_metrics, overall_metrics, display_names)

    if save_per_sample:
        print(f"[Info] Saved {len(saved_files)} predictions to: {output_dir}")
    else:
        output_path = save_as_netcdf(
            predictions,
            labels_np,
            diff,
            all_dates if all_dates else None,
            output_dir,
            spatial_coords,
            channel_names,
            filename=_range_filename(start_date, end_date),
        )
        print(f"[Info] Saved predictions to: {output_path}")

    return {
        "predictions": predictions,
        "labels": labels_np,
        "diff": diff,
        "dates": all_dates if all_dates else None,
        "saved_files": saved_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict over a date range")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(config.CKPT_DIR / "best_swin_t0.pth"),
        help="Checkpoint path",
    )
    parser.add_argument("--start-date", type=str, default=config.TEST_START_DATE)
    parser.add_argument("--end-date", type=str, default=config.TEST_END_DATE)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.RESULT_DIR / "predictions"),
    )

    parser.add_argument("--device", type=str, default=str(config.DEVICE))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=("per-sample", "all"),
        default="per-sample",
        help="Save one NetCDF per sample or a single combined file.",
    )
    parser.add_argument(
        "--no-denorm",
        action="store_true",
        help="Skip label denormalization even if configured",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")
    print(f"[Info] Predicting from {args.start_date} to {args.end_date}")

    data_dir = str(config.DATA_DIR)
    dataset = ZarrTimeSeriesDataset(
        data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        allowed_hours=config.ALLOWED_BASE_HOURS,
    )
    print(f"[Info] Found {len(dataset)} samples")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR

    loader = DataLoader(dataset, **loader_kwargs)

    spatial_coords = get_spatial_coords_from_zarr(data_dir)
    if config.LAT_WEIGHTING:
        lat_source = config.LAT_WEIGHT_SOURCE or data_dir
        init_latitude_weights(lat_source, config.HEIGHT, config.WIDTH)

    sample = dataset[0]
    in_chans = sample["x"].shape[0]
    num_classes = sample["y"].shape[0]
    print(f"[Info] in_chans={in_chans}, num_classes={num_classes}")

    model = SwinTiny2D(in_chans=in_chans, num_classes=num_classes)
    ckpt = torch.load(args.ckpt, map_location=device,weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Key 'model_state_dict' not found in checkpoint.")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    predict_date_range(
        model,
        loader,
        device,
        args.output_dir,
        spatial_coords,
        denorm=not args.no_denorm,
        save_mode=args.save_mode,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
