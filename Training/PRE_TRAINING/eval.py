import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ZarrTimeSeriesDataset
from features import denormalize_label
from lat_weights import init_latitude_weights
from metrics import finalize_metric_sums, get_label_channel_names, init_metric_sums, update_metric_sums
from models.swin import SwinTiny2D


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




def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    denorm: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    model.eval()
    stats = None

    with torch.no_grad(), tqdm(loader, desc="Evaluating", unit="batch", ncols=100) as pbar:
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)

            mask = _build_eval_mask(batch.get("masks", {}))
            pred = pred.detach().cpu()
            y = y.detach().cpu()
            pred, y = _maybe_denormalize(pred, y, denorm)

            if stats is None:
                stats = init_metric_sums(pred.shape[1], device=pred.device)
            mask_cpu = mask.detach().cpu() if mask is not None else None
            update_metric_sums(stats, pred, y, mask=mask_cpu)

    if stats is None:
        empty = {
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "nmb": float("nan"),
            "nme": float("nan"),
        }
        return {k: [] for k in empty.keys()}, empty

    return finalize_metric_sums(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin2D on test set")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(config.CKPT_DIR / "best_swin_t0.pth"),
        help="Checkpoint path",
    )
    parser.add_argument("--device", type=str, default=str(config.DEVICE))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--start-date", type=str, default=config.TEST_START_DATE)
    parser.add_argument("--end-date", type=str, default=config.TEST_END_DATE)
    parser.add_argument(
        "--no-denorm",
        action="store_true",
        help="Skip label denormalization even if configured",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_dir = str(config.DATA_DIR)
    test_ds = ZarrTimeSeriesDataset(
        data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        allowed_hours=config.ALLOWED_BASE_HOURS,
    )

    if config.LAT_WEIGHTING:
        lat_source = config.LAT_WEIGHT_SOURCE or data_dir
        init_latitude_weights(lat_source, config.HEIGHT, config.WIDTH)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR

    loader = DataLoader(test_ds, **loader_kwargs)

    sample = test_ds[0]
    in_chans = sample["x"].shape[0]
    num_classes = sample["y"].shape[0]
    print(f"[Info] in_chans={in_chans}, num_classes={num_classes}")

    model = SwinTiny2D(in_chans=in_chans, num_classes=num_classes)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Key 'model_state_dict' not found in checkpoint.")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    channel_metrics, overall_metrics = evaluate(
        model,
        loader,
        device,
        denorm=not args.no_denorm,
    )

    label_names = get_label_channel_names(include_group=False, include_time=False)
    num_channels = len(channel_metrics["r2"])

    print("")
    print("========== Overall Metrics ==========")
    print(f"  R2   = {overall_metrics['r2']:.6f}")
    print(f"  RMSE = {overall_metrics['rmse']:.6f}")
    print(f"  MAE  = {overall_metrics['mae']:.6f}")
    # print(f"  NMB  = {overall_metrics['nmb']:.6f}")
    # print(f"  NME  = {overall_metrics['nme']:.6f}")
    print("")

    print("========== Per-Channel Metrics ==========")
    for c in range(num_channels):
        name = label_names[c] if c < len(label_names) else f"channel_{c}"
        print(f"{name}:")
        print(f"  R2   = {channel_metrics['r2'][c]:.6f}")
        print(f"  RMSE = {channel_metrics['rmse'][c]:.6f}")
        print(f"  MAE  = {channel_metrics['mae'][c]:.6f}")
        # print(f"  NMB  = {channel_metrics['nmb'][c]:.6f}")
        # print(f"  NME  = {channel_metrics['nme'][c]:.6f}")
        print("")


if __name__ == "__main__":
    main()
