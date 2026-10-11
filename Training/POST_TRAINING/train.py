import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import config
from dataset import ZarrTimeSeriesDataset
from features import get_feature_keys
from lat_weights import init_latitude_weights
from losses import combined_loss
from metrics import finalize_metric_sums, get_label_channel_names, init_metric_sums, update_metric_sums
from models.swin import SwinTiny2D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metric(value, decimals: int = 6) -> str:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _normalize_date_ranges(date_ranges: Any) -> list[tuple[str, str]] | None:
    if date_ranges in (None, ""):
        return None
    if not isinstance(date_ranges, (list, tuple)):
        raise ValueError("DATE_RANGES must be a list/tuple of (start_date, end_date).")

    normalized: list[tuple[str, str]] = []
    for item in date_ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each DATE_RANGES item must be a tuple/list with 2 elements.")
        start_s = str(item[0]).strip()
        end_s = str(item[1]).strip()
        start_d = datetime.strptime(start_s, "%Y-%m-%d").date()
        end_d = datetime.strptime(end_s, "%Y-%m-%d").date()
        if start_d > end_d:
            raise ValueError(f"Invalid date range: ({start_s}, {end_s})")
        normalized.append((start_s, end_s))

    return normalized if normalized else None


def _build_dataset_with_optional_ranges(
    data_dir: str,
    start_date: str | None,
    end_date: str | None,
    date_ranges: list[tuple[str, str]] | None,
    allowed_hours: set[int] | None,
) -> ZarrTimeSeriesDataset | ConcatDataset:
    if date_ranges:
        datasets = [
            ZarrTimeSeriesDataset(
                data_dir,
                start_date=seg_start,
                end_date=seg_end,
                allowed_hours=allowed_hours,
            )
            for seg_start, seg_end in date_ranges
        ]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    return ZarrTimeSeriesDataset(
        data_dir,
        start_date=start_date,
        end_date=end_date,
        allowed_hours=allowed_hours,
    )


def _load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept weights_only.
        return torch.load(path, map_location=device)


def _extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint format is invalid: expected a dict-like object.")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]

    # Some checkpoints are saved as a raw state_dict directly.
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise KeyError("Cannot find model state dict in checkpoint.")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def load_model_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as err:
        if not state_dict:
            raise err

    first_key = next(iter(state_dict.keys()))
    has_module_prefix = first_key.startswith("module.")
    model_is_parallel = isinstance(model, torch.nn.DataParallel)

    if model_is_parallel and not has_module_prefix:
        adapted = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(adapted)
        print("Loaded checkpoint by adding 'module.' prefix for DataParallel model.")
        return

    if not model_is_parallel and has_module_prefix:
        adapted = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(adapted)
        print("Loaded checkpoint by removing 'module.' prefix for single-GPU model.")
        return

    model.load_state_dict(state_dict)


def build_label_denorm_params(device: torch.device):
    if not config.LABEL_NORMALIZE or config.LABEL_NORMALIZE_SCOPE != "dataset":
        return None

    stats_path = Path(config.STATS_FILE)
    if not stats_path.exists():
        print(f"Warning: stats file not found at {stats_path}; metrics remain normalized")
        return None

    with open(stats_path, "r") as f:
        stats = json.load(f)

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for var in config.LABEL_VARS:
        key = f"{var}_{config.LABEL_TIME}"
        if key not in stats:
            print(f"Warning: stats missing for {key}; metrics remain normalized")
            return None
        s = stats[key]
        means.append(float(s.get("mean", 0.0)))
        stds.append(float(s.get("std", 1.0)) + 1e-6)
        mins.append(float(s.get("min", 0.0)))
        maxs.append(float(s.get("max", 1.0)))

    mean_t = torch.tensor(means, device=device).view(-1, 1, 1)
    std_t = torch.tensor(stds, device=device).view(-1, 1, 1)
    min_t = torch.tensor(mins, device=device).view(-1, 1, 1)
    max_t = torch.tensor(maxs, device=device).view(-1, 1, 1)

    return {
        "method": config.LABEL_NORMALIZE_METHOD,
        "mean": mean_t,
        "std": std_t,
        "min": min_t,
        "max": max_t,
    }


def denormalize_tensor(tensor: torch.Tensor, params: Dict | None) -> torch.Tensor:
    if params is None:
        return tensor
    method = params["method"]
    if method == "zscore":
        return tensor * params["std"] + params["mean"]
    if method == "minmax":
        return ((tensor + 1) / 2) * (params["max"] - params["min"] + 1e-6) + params["min"]
    return tensor


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    denorm_params,
    scheduler=None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
):
    model.train()
    total_loss = 0.0

    num_channels = None
    metric_stats = None

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    for i, batch in enumerate(pbar):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        train_mask = None
        if config.TRAIN_MASK and config.TRAIN_MASK in batch.get("masks", {}):
            train_mask = batch["masks"][config.TRAIN_MASK].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(x)
            loss = combined_loss(pred, y, train_mask)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            y_metric = denormalize_tensor(y, denorm_params)
            pred_metric = denormalize_tensor(pred.detach(), denorm_params)
            if num_channels is None:
                num_channels = y_metric.shape[1]
                metric_stats = init_metric_sums(num_channels, device=device)
            update_metric_sums(metric_stats, pred_metric, y_metric, mask=train_mask)

        pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

    if metric_stats is None:
        empty = {k: float("nan") for k in ("r2", "rmse", "mae", "nme", "nmb")}
        channel_metrics, overall_metrics = {k: [] for k in empty.keys()}, empty
    else:
        channel_metrics, overall_metrics = finalize_metric_sums(metric_stats)

    valid_r2s = [r for r in channel_metrics["r2"] if not np.isnan(r)]
    valid_rmses = [r for r in channel_metrics["rmse"] if not np.isnan(r)]
    valid_maes = [r for r in channel_metrics["mae"] if not np.isnan(r)]
    valid_nmes = [r for r in channel_metrics["nme"] if not np.isnan(r)]
    valid_nmbs = [r for r in channel_metrics["nmb"] if not np.isnan(r)]

    avg_r2 = np.mean(valid_r2s) if valid_r2s else float("nan")
    avg_rmse = np.mean(valid_rmses) if valid_rmses else float("nan")
    avg_mae = np.mean(valid_maes) if valid_maes else float("nan")
    avg_nme = np.mean(valid_nmes) if valid_nmes else float("nan")
    avg_nmb = np.mean(valid_nmbs) if valid_nmbs else float("nan")

    return {
        "loss": total_loss / max(1, len(loader)),
        "r2": overall_metrics["r2"],
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "nme": overall_metrics["nme"],
        "nmb": overall_metrics["nmb"],
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_nme": avg_nme,
        "avg_nmb": avg_nmb,
        "r2_per_channel": channel_metrics["r2"],
        "rmse_per_channel": channel_metrics["rmse"],
        "mae_per_channel": channel_metrics["mae"],
        "nme_per_channel": channel_metrics["nme"],
        "nmb_per_channel": channel_metrics["nmb"],
    }


def evaluate(model, loader, device, denorm_params, amp_enabled: bool = False, amp_dtype: torch.dtype | None = None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0

    num_channels = None
    metric_stats = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            mask = None
            for mname in config.EVAL_MASKS:
                if mname in batch["masks"]:
                    m = batch["masks"][mname].to(device)
                    mask = m if mask is None else (mask * m)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x)
                loss = combined_loss(pred, y, mask)
            total_loss += loss.item()

            y_metric = denormalize_tensor(y, denorm_params)
            pred_metric = denormalize_tensor(pred, denorm_params)
            if num_channels is None:
                num_channels = y_metric.shape[1]
                metric_stats = init_metric_sums(num_channels, device=device)
            update_metric_sums(metric_stats, pred_metric, y_metric, mask=mask)

    if metric_stats is None:
        empty = {k: float("nan") for k in ("r2", "rmse", "mae", "nme", "nmb")}
        channel_metrics, overall_metrics = {k: [] for k in empty.keys()}, empty
    else:
        channel_metrics, overall_metrics = finalize_metric_sums(metric_stats)

    valid_r2s = [r for r in channel_metrics["r2"] if not np.isnan(r)]
    valid_rmses = [r for r in channel_metrics["rmse"] if not np.isnan(r)]
    valid_maes = [r for r in channel_metrics["mae"] if not np.isnan(r)]
    valid_nmes = [r for r in channel_metrics["nme"] if not np.isnan(r)]
    valid_nmbs = [r for r in channel_metrics["nmb"] if not np.isnan(r)]

    avg_r2 = np.mean(valid_r2s) if valid_r2s else float("nan")
    avg_rmse = np.mean(valid_rmses) if valid_rmses else float("nan")
    avg_mae = np.mean(valid_maes) if valid_maes else float("nan")
    avg_nme = np.mean(valid_nmes) if valid_nmes else float("nan")
    avg_nmb = np.mean(valid_nmbs) if valid_nmbs else float("nan")

    return {
        "loss": total_loss / max(1, len(loader)),
        "r2": overall_metrics["r2"],
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "nme": overall_metrics["nme"],
        "nmb": overall_metrics["nmb"],
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_nme": avg_nme,
        "avg_nmb": avg_nmb,
        "r2_per_channel": channel_metrics["r2"],
        "rmse_per_channel": channel_metrics["rmse"],
        "mae_per_channel": channel_metrics["mae"],
        "nme_per_channel": channel_metrics["nme"],
        "nmb_per_channel": channel_metrics["nmb"],
    }


def save_best_model_log(epoch, train_metrics, val_metrics, log_dir):
    def to_serializable(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value.tolist()
        if isinstance(value, (int, float)) and np.isnan(value):
            return None
        return value

    def to_serializable_list(values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return [to_serializable(v) for v in values]

    log_data = {
        "epoch": epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_metrics": {
            "loss": to_serializable(train_metrics["loss"]),
            "r2": to_serializable(train_metrics["r2"]),
            "rmse": to_serializable(train_metrics["rmse"]),
            "mae": to_serializable(train_metrics["mae"]),
            "nme": to_serializable(train_metrics["nme"]),
            "nmb": to_serializable(train_metrics["nmb"]),
            "r2_per_channel": to_serializable_list(train_metrics["r2_per_channel"]),
            "rmse_per_channel": to_serializable_list(train_metrics["rmse_per_channel"]),
            "mae_per_channel": to_serializable_list(train_metrics["mae_per_channel"]),
            "nme_per_channel": to_serializable_list(train_metrics["nme_per_channel"]),
            "nmb_per_channel": to_serializable_list(train_metrics["nmb_per_channel"]),
            "avg_r2": to_serializable(train_metrics["avg_r2"]),
            "avg_rmse": to_serializable(train_metrics["avg_rmse"]),
            "avg_mae": to_serializable(train_metrics["avg_mae"]),
            "avg_nme": to_serializable(train_metrics["avg_nme"]),
            "avg_nmb": to_serializable(train_metrics["avg_nmb"]),
        },
        "val_metrics": {
            "loss": to_serializable(val_metrics["loss"]),
            "r2": to_serializable(val_metrics["r2"]),
            "rmse": to_serializable(val_metrics["rmse"]),
            "mae": to_serializable(val_metrics["mae"]),
            "nme": to_serializable(val_metrics["nme"]),
            "nmb": to_serializable(val_metrics["nmb"]),
            "r2_per_channel": to_serializable_list(val_metrics["r2_per_channel"]),
            "rmse_per_channel": to_serializable_list(val_metrics["rmse_per_channel"]),
            "mae_per_channel": to_serializable_list(val_metrics["mae_per_channel"]),
            "nme_per_channel": to_serializable_list(val_metrics["nme_per_channel"]),
            "nmb_per_channel": to_serializable_list(val_metrics["nmb_per_channel"]),
            "avg_r2": to_serializable(val_metrics["avg_r2"]),
            "avg_rmse": to_serializable(val_metrics["avg_rmse"]),
            "avg_mae": to_serializable(val_metrics["avg_mae"]),
            "avg_nme": to_serializable(val_metrics["avg_nme"]),
            "avg_nmb": to_serializable(val_metrics["avg_nmb"]),
        },
    }

    log_file = os.path.join(log_dir, f"best_model_log_{config.LABEL_TIME}.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    tqdm.write(f"  Best model log saved to {log_file}")


def main() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = config.DEVICE
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        print(f"Warning: configured device {device} is unavailable; fallback to cuda:0")
        device = torch.device("cuda:0")

    available_gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    enable_multi_gpu = bool(getattr(config, "MULTI_GPU", True))
    raw_gpu_ids = getattr(config, "GPU_IDS", []) or []
    gpu_ids = list(raw_gpu_ids)
    if enable_multi_gpu and device.type == "cuda":
        if gpu_ids:
            parsed_gpu_ids = []
            for gid in gpu_ids:
                try:
                    idx = int(gid)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < available_gpu_count:
                    parsed_gpu_ids.append(idx)
            gpu_ids = parsed_gpu_ids
        else:
            gpu_ids = list(range(available_gpu_count))
    else:
        gpu_ids = []

    use_multi_gpu = enable_multi_gpu and device.type == "cuda" and len(gpu_ids) > 1
    if use_multi_gpu:
        device = torch.device(f"cuda:{gpu_ids[0]}")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA devices available: {available_gpu_count}")
    if use_multi_gpu:
        print(f"Multi-GPU enabled (DataParallel), GPU IDs: {gpu_ids}")
    elif enable_multi_gpu and device.type == "cuda" and available_gpu_count <= 1:
        print("Multi-GPU requested but only one CUDA device is visible; using single GPU.")
    print(
        "Loss function: "
        f"{config.LOSS_ALPHA:.2f} * Charbonnier(eps={config.CHARBONNIER_EPSILON}) + "
        f"{config.LOSS_BETA:.2f} * SSIM(range={config.SSIM_DATA_RANGE})"
    )

    data_dir = str(config.DATA_DIR)
    train_date_ranges = _normalize_date_ranges(getattr(config, "TRAIN_DATE_RANGES", None))
    val_date_ranges = _normalize_date_ranges(getattr(config, "VAL_DATE_RANGES", None))
    allowed_hours_cfg = getattr(config, "ALLOWED_BASE_HOURS", None)
    allowed_hours = set(allowed_hours_cfg) if allowed_hours_cfg is not None else None

    print(
        "Data split: "
        f"train[{config.TRAIN_START_DATE} ~ {config.TRAIN_END_DATE}] "
        f"val[{config.VAL_START_DATE} ~ {config.VAL_END_DATE}] "
        f"hours={sorted(allowed_hours) if allowed_hours is not None else None}"
    )
    if train_date_ranges:
        print(f"Train date ranges override: {train_date_ranges}")
    if val_date_ranges:
        print(f"Val date ranges override: {val_date_ranges}")

    train_ds = _build_dataset_with_optional_ranges(
        data_dir,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        date_ranges=train_date_ranges,
        allowed_hours=allowed_hours,
    )
    val_ds = _build_dataset_with_optional_ranges(
        data_dir,
        start_date=config.VAL_START_DATE,
        end_date=config.VAL_END_DATE,
        date_ranges=val_date_ranges,
        allowed_hours=allowed_hours,
    )

    if config.LAT_WEIGHTING:
        lat_source = config.LAT_WEIGHT_SOURCE
        if lat_source is None:
            if hasattr(train_ds, "files") and train_ds.files:
                lat_source = train_ds.files[0]
            else:
                lat_source = data_dir
        init_latitude_weights(lat_source, config.HEIGHT, config.WIDTH)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        prefetch_factor=config.PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        prefetch_factor=config.PREFETCH_FACTOR,
    )

    in_chans = len(get_feature_keys())
    num_classes = len(config.LABEL_VARS)
    model = SwinTiny2D(in_chans=in_chans, num_classes=num_classes).to(device)
    if use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, config.EPOCHS * steps_per_epoch)
    warmup_steps = int(getattr(config, "WARMUP_STEPS", 0) or 0)
    if warmup_steps <= 0:
        warmup_epochs = float(getattr(config, "WARMUP_EPOCHS", 0))
        warmup_steps = max(0, int(warmup_epochs * steps_per_epoch))

    base_lr = float(config.LEARNING_RATE)
    min_lr = float(config.MIN_LR)
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / float(max(1, warmup_steps))
        if warmup_steps >= total_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp_enabled = bool(getattr(config, "AMP_ENABLED", False)) and device.type == "cuda"
    amp_dtype_name = str(getattr(config, "AMP_DTYPE", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    denorm_params = build_label_denorm_params(device)

    best_loss = float("inf")
    patience_counter = 0

    best_ckpt = os.path.join(config.CKPT_DIR, "best_swin_t0.pth")
    latest_ckpt = os.path.join(config.CKPT_DIR, "latest_swin_t0.pth")

    start_epoch = 1
    if os.path.exists(latest_ckpt):
        #ckpt = torch.load(latest_ckpt, map_location=device)
        ckpt = _load_checkpoint(latest_ckpt, device)
        load_model_state_dict(model, _extract_model_state_dict(ckpt))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            print(f"Warning: failed to load scheduler state ({exc}); using fresh scheduler.")
        if scaler is not None and "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as exc:
                print(f"Warning: failed to load scaler state ({exc}); using fresh scaler.")
        best_loss = float(ckpt.get("best_loss", best_loss))
        patience_counter = int(ckpt.get("patience_counter", patience_counter))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from {latest_ckpt} at epoch {start_epoch}")
    else:
        pretrained_ckpt = getattr(config, "PRETRAINED_CKPT_PATH", None)
        if pretrained_ckpt:
            pretrained_ckpt = str(pretrained_ckpt)
            if os.path.exists(pretrained_ckpt):
                ckpt = _load_checkpoint(pretrained_ckpt, device)
                load_model_state_dict(model, _extract_model_state_dict(ckpt))
                print(f"Initialized model weights from pretrained checkpoint: {pretrained_ckpt}")
            else:
                print(
                    "Warning: PRETRAINED_CKPT_PATH does not exist: "
                    f"{pretrained_ckpt}; training starts from random initialization."
                )

    if start_epoch > config.EPOCHS:
        print(f"Checkpoint already finished training at epoch {start_epoch - 1}.")
        return

    for epoch in range(start_epoch, config.EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            denorm_params,
            scheduler=scheduler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, device, denorm_params, amp_enabled=amp_enabled, amp_dtype=amp_dtype)

        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(f"\n{'='*120}")
        tqdm.write(f"Epoch {epoch}/{config.EPOCHS} - LR: {current_lr:.6f}")
        tqdm.write(f"{'='*120}")

        tqdm.write("\nTraining Metrics (Overall)")
        tqdm.write(f"  Loss: {format_metric(train_metrics['loss'])}")
        tqdm.write(f"  R2:   {format_metric(train_metrics['r2'])}")
        tqdm.write(f"  RMSE: {format_metric(train_metrics['rmse'])}")
        tqdm.write(f"  MAE:  {format_metric(train_metrics['mae'])}")
        tqdm.write(f"  NME:  {format_metric(train_metrics['nme'])}")
        tqdm.write(f"  NMB:  {format_metric(train_metrics['nmb'])}")

        tqdm.write("\nValidation Metrics (Overall)")
        tqdm.write(f"  Loss: {format_metric(val_metrics['loss'])}")
        tqdm.write(f"  R2:   {format_metric(val_metrics['r2'])}")
        tqdm.write(f"  RMSE: {format_metric(val_metrics['rmse'])}")
        tqdm.write(f"  MAE:  {format_metric(val_metrics['mae'])}")
        tqdm.write(f"  NME:  {format_metric(val_metrics['nme'])}")
        tqdm.write(f"  NMB:  {format_metric(val_metrics['nmb'])}")

        tqdm.write("\nValidation Metrics (Per Channel)")
        channel_names = get_label_channel_names(include_group=False, include_time=False)
        tqdm.write("  Channel                    R2         RMSE        MAE         NME         NMB")
        for i in range(len(val_metrics["r2_per_channel"])):
            name = channel_names[i] if i < len(channel_names) else f"channel_{i}"
            tqdm.write(
                f"  {name:<26}"
                f" {format_metric(val_metrics['r2_per_channel'][i], 4):<11}"
                f" {format_metric(val_metrics['rmse_per_channel'][i], 4):<11}"
                f" {format_metric(val_metrics['mae_per_channel'][i], 4):<11}"
                f" {format_metric(val_metrics['nme_per_channel'][i], 4):<11}"
                f" {format_metric(val_metrics['nmb_per_channel'][i], 4):<11}"
            )
        tqdm.write(
            f"  Average    {format_metric(val_metrics['avg_r2'], 4):<11}"
            f" {format_metric(val_metrics['avg_rmse'], 4):<11}"
            f" {format_metric(val_metrics['avg_mae'], 4):<11}"
            f" {format_metric(val_metrics['avg_nme'], 4):<11}"
            f" {format_metric(val_metrics['avg_nmb'], 4):<11}"
        )

        improved = val_metrics["avg_rmse"] < best_loss - config.MIN_DELTA
        if improved:
            best_loss = val_metrics["avg_rmse"]
            patience_counter = 0
            tqdm.write(f"\n  New best model! avg_rmse: {best_loss:.6f}")

            best_state = {
                "epoch": epoch,
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "patience_counter": patience_counter,
            }
            if scaler is not None:
                best_state["scaler_state_dict"] = scaler.state_dict()
            torch.save(best_state, best_ckpt)
            save_best_model_log(epoch, train_metrics, val_metrics, config.LOG_DIR)
        else:
            patience_counter += 1
            tqdm.write(f"\n  Early stopping counter: {patience_counter}/{config.PATIENCE}")

        latest_state = {
            "epoch": epoch,
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        }
        if scaler is not None:
            latest_state["scaler_state_dict"] = scaler.state_dict()
        torch.save(latest_state, latest_ckpt)

        if patience_counter >= config.PATIENCE:
            tqdm.write(f"\n{'='*120}")
            tqdm.write(f"Early stopping triggered after {epoch} epochs")
            tqdm.write(f"Best validation avg_rmse: {best_loss:.6f}")
            tqdm.write(f"{'='*120}")
            break

    print("\n" + "=" * 120)
    print("Training finished.")
    print(f"Best validation avg_rmse: {best_loss:.6f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
