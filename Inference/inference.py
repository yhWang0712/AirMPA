import os
import sys
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = SCRIPT_DIR
DEFAULT_ONNX_PATH = BUNDLE_DIR / "checkpoints" / "AirMPA_checkpoint.onnx"
DEFAULT_ZARR_PATH = BUNDLE_DIR / "data" / "cams_fuxi_900_451_v1_3day.zarr"
DEFAULT_OUTPUT_ROOT = BUNDLE_DIR / "output"
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

import config  # noqa: E402
from features import denormalize_label, get_feature_keys, stack_channels  # noqa: E402


KNOWN_TIME_LABELS = sorted(
    {label for labels in config.FEATURE_TIME_LABELS.values() for label in labels} | {config.LABEL_TIME},
    key=len,
    reverse=True,
)


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def split_key_time_label(key: str) -> Tuple[str, str]:
    for label in KNOWN_TIME_LABELS:
        suffix = f"_{label}"
        if key.endswith(suffix):
            return key[: -len(suffix)], label
    return key, ""


def compute_lon_offsets(lon: np.ndarray) -> np.ndarray:
    offsets = np.floor((lon + 7.5) / 15.0)
    offsets = np.mod(offsets, 24)
    return offsets.astype(np.float32)


def temporal_features_for_time(dt: datetime, lon_offsets: np.ndarray, height: int, width: int) -> Dict[str, np.ndarray]:
    year_days = 366 if (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0)) else 365
    doy = dt.timetuple().tm_yday
    doy_angle = 2.0 * np.pi * ((doy - 1) / year_days)

    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    local_hour = (utc_hour + lon_offsets) % 24.0
    hour_angle = 2.0 * np.pi * (local_hour / 24.0)

    day_sin = np.full((height, width), np.sin(doy_angle).astype(np.float32), dtype=np.float32)
    day_cos = np.full((height, width), np.cos(doy_angle).astype(np.float32), dtype=np.float32)
    hour_sin = np.broadcast_to(np.sin(hour_angle).astype(np.float32), (height, width)).astype(np.float32)
    hour_cos = np.broadcast_to(np.cos(hour_angle).astype(np.float32), (height, width)).astype(np.float32)

    return {
        "day_of_year_sin": day_sin,
        "day_of_year_cos": day_cos,
        "hour_of_day_sin": hour_sin,
        "hour_of_day_cos": hour_cos,
    }


def generate_lat_weights(lat_values):
    lat_values = np.asarray(lat_values, dtype=np.float32)
    return np.cos(np.deg2rad(np.abs(lat_values))).reshape(-1, 1)


def simple_weighted_rmse_from_error(error, lat_weights):
    err = np.asarray(error, dtype=np.float64)
    w = np.asarray(lat_weights, dtype=np.float64)
    weighted_err2 = (err * err) * w
    if not np.any(np.isfinite(weighted_err2)):
        return np.nan
    return float(np.sqrt(np.nanmean(weighted_err2)))


def strict_weighted_rmse_from_error(error, lat_weights):
    err = np.asarray(error, dtype=np.float64)
    w = np.asarray(lat_weights, dtype=np.float64)
    mask = np.isfinite(err)
    if not np.any(mask):
        return np.nan

    weighted_err2 = np.where(mask, (err * err) * w, 0.0)
    weighted_w = np.where(mask, w, 0.0)
    denom = float(np.sum(weighted_w))
    if denom <= 0.0:
        return np.nan
    return float(np.sqrt(np.sum(weighted_err2) / denom))


def weighted_rmse_from_error(error, lat_weights, use_strict_weighted_avg=False):
    if use_strict_weighted_avg:
        return strict_weighted_rmse_from_error(error, lat_weights)
    return simple_weighted_rmse_from_error(error, lat_weights)


def create_lock_file(dir_path):
    lock_file = dir_path / ".lock"
    if lock_file.exists():
        logger.warning(f"Lock file exists; {dir_path} may be running or was interrupted previously")
        return False

    with open(lock_file, "w", encoding="utf-8") as f:
        f.write(f"pid: {os.getpid()}\n")
        f.write(f"start_time: {pd.Timestamp.now()}\n")

    logger.info(f"Created lock file: {lock_file}")
    return True


def create_complete_file(dir_path, success=True, message=""):
    complete_file = dir_path / ".complete"
    with open(complete_file, "w", encoding="utf-8") as f:
        f.write(f"success: {success}\n")
        f.write(f"end_time: {pd.Timestamp.now()}\n")
        if message:
            f.write(f"message: {message}\n")

    logger.info(f"Created completion file: {complete_file}")


def remove_lock_file(dir_path):
    lock_file = dir_path / ".lock"
    if lock_file.exists():
        lock_file.unlink()
        logger.info(f"Removed lock file: {lock_file}")


class OnnxInferencer:
    def __init__(self, onnx_path, lat_values, lon_values, device="cuda"):
        self.onnx_path = str(onnx_path)
        self.requested_device = str(device).lower()
        self.lat_weights = generate_lat_weights(lat_values)
        self.feature_keys = get_feature_keys()
        self.label_vars = list(config.LABEL_VARS)
        self.temporal_vars = set(config.FEATURE_GROUPS.get("temporal", []))
        self.height = len(lat_values)
        self.width = len(lon_values)
        self.lon_offsets = compute_lon_offsets(np.asarray(lon_values, dtype=np.float32))

        self.ort_session = None
        self.ort_input_name = None
        self.ort_output_name = None
        self._init_onnx_session()

    def _init_onnx_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Please install `onnxruntime` or `onnxruntime-gpu`."
            ) from exc

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model was not found: {self.onnx_path}")

        available_providers = ort.get_available_providers()
        if self.requested_device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            if self.requested_device == "cuda":
                logger.warning(
                    "CUDAExecutionProvider is unavailable in current onnxruntime build; fallback to CPUExecutionProvider"
                )
            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available_providers else available_providers

        if not providers:
            raise RuntimeError("No available ONNX Runtime execution providers found.")

        logger.info(f"Loading ONNX model: {self.onnx_path}")
        self.ort_session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        self.ort_output_name = self.ort_session.get_outputs()[0].name
        logger.success(
            "ONNX model loaded successfully, providers="
            f"{self.ort_session.get_providers()}"
        )

    @staticmethod
    def _extract_channel_frame(data, time_val, lead_val, channels: List[str]) -> Dict[str, np.ndarray]:
        arr = (
            data.sel(time=time_val, channel=channels)
            .isel(lead_time=lead_val)
            .transpose("channel", "lat", "lon")
            .values.astype(np.float32)
        )
        return {c: arr[i] for i, c in enumerate(channels)}

    def _extract_noncams_future_series(
        self,
        data,
        init_time,
        infer_steps: int,
        noncams_vars: List[str],
    ) -> List[Dict[str, np.ndarray]]:
        arr = (
            data.sel(time=init_time, channel=noncams_vars)
            .isel(lead_time=slice(0, infer_steps))
            .transpose("lead_time", "channel", "lat", "lon")
            .values.astype(np.float32)
        )
        if arr.shape[0] < infer_steps:
            raise ValueError(f"External fields for init_time {init_time} have fewer than {infer_steps} lead_time steps")
        return [{c: arr[s, i] for i, c in enumerate(noncams_vars)} for s in range(infer_steps)]

    def _build_step_sample(
        self,
        cams_hist: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        noncams_minus1: Dict[str, np.ndarray],
        noncams_t0: Dict[str, np.ndarray],
        step_time: pd.Timestamp,
    ) -> Dict[str, np.ndarray]:
        sample: Dict[str, np.ndarray] = {}
        temporal_map = temporal_features_for_time(step_time.to_pydatetime(), self.lon_offsets, self.height, self.width)

        for key in self.feature_keys:
            base_var, label = split_key_time_label(key)
            if base_var in self.temporal_vars:
                sample[key] = temporal_map[base_var]
                continue

            if base_var in self.label_vars:
                if label == "t_minus_2":
                    sample[key] = cams_hist[0][base_var]
                elif label == "t_minus_1":
                    sample[key] = cams_hist[1][base_var]
                else:
                    raise KeyError(f"Unexpected time label for CAMS variable: {key}")
            else:
                if label == "t_minus_1":
                    sample[key] = noncams_minus1[base_var]
                elif label == "t0":
                    sample[key] = noncams_t0[base_var]
                else:
                    raise KeyError(f"Unexpected time label for external-field variable: {key}")

        return sample

    def _predict_one_step(self, sample: Dict[str, np.ndarray]) -> np.ndarray:
        x, _ = stack_channels(sample)
        x_np = x.detach().cpu().numpy().astype(np.float32)
        if getattr(config, "LON_ROLL", 0):
            x_np = np.roll(x_np, shift=int(config.LON_ROLL), axis=-1)

        x_np = np.expand_dims(x_np, axis=0)
        pred_norm = self.ort_session.run([self.ort_output_name], {self.ort_input_name: x_np})[0]
        pred_norm_np = np.asarray(pred_norm[0], dtype=np.float32)

        if getattr(config, "LON_ROLL", 0):
            pred_norm_np = np.roll(pred_norm_np, shift=-int(config.LON_ROLL), axis=-1)

        pred_raw_np = np.empty_like(pred_norm_np)
        for i, var in enumerate(self.label_vars):
            pred_raw_np[i] = denormalize_label(pred_norm_np[i], f"{var}_{config.LABEL_TIME}")
        pred_raw_np = np.maximum(pred_raw_np, 0.0)

        return pred_raw_np

    def infer_single_init_time(
        self,
        init_time,
        data,
        infer_steps=10,
        output_root="./output",
        pred_vars=None,
        noncams_vars=None,
        use_strict_weighted_avg=False,
    ):
        pred_vars = pred_vars or self.label_vars
        noncams_vars = noncams_vars or []

        time_str = init_time.strftime("%Y%m%d_%H")
        for output_dir in (Path(output_root) / time_str, Path(output_root) / f"{time_str}_steps{infer_steps}"):
            complete_file = output_dir / ".complete"
            rmse_file = output_dir / "rmse_results.csv"
            if not (complete_file.exists() and rmse_file.exists()):
                break
            previous = pd.read_csv(rmse_file)
            if (
                complete_file.read_text(encoding="utf-8").startswith("success: True\n")
                and "step" in previous
                and previous["step"].tolist() == list(range(infer_steps))
                and all((output_dir / f"output_{step}.npy").exists() for step in range(infer_steps))
            ):
                logger.info(f"Init time {time_str} already has {infer_steps} completed steps; skipping")
                return previous

        output_dir.mkdir(parents=True, exist_ok=True)

        if not create_lock_file(output_dir):
            return None

        try:
            logger.info(f"Processing init time: {time_str}; outputs will be saved to: {output_dir}")
            prev_time = init_time - pd.Timedelta("12H")
            curr_time = init_time

            cams_prev = self._extract_channel_frame(data, prev_time, 0, pred_vars)
            cams_hist = self._extract_channel_frame(data, curr_time, 0, pred_vars)
            noncams_future = self._extract_noncams_future_series(
                data=data,
                init_time=init_time,
                infer_steps=infer_steps + 1,
                noncams_vars=noncams_vars,
            )

            cams_queue = (cams_prev, cams_hist)
            step_rmse_results = []

            for step in tqdm(range(infer_steps), desc=f"Inference {time_str} step"):
                noncams_minus1 = noncams_future[step]
                noncams_t0 = noncams_future[step + 1]
                step_base_time = init_time + pd.Timedelta(f"{(step + 1) * 12}H")

                sample = self._build_step_sample(
                    cams_hist=cams_queue,
                    noncams_minus1=noncams_minus1,
                    noncams_t0=noncams_t0,
                    step_time=step_base_time,
                )
                pred_raw = self._predict_one_step(sample)

                output = pred_raw[None, None, :, :, :]
                np.save(output_dir / f"output_{step}.npy", output)

                step_rmse = {"step": step, "init_time": time_str}
                target_time = step_base_time
                for idx, c in enumerate(pred_vars):
                    try:
                        target = data.sel(time=target_time, channel=c).isel(lead_time=0).values
                        error = pred_raw[idx] - target
                        rmse = weighted_rmse_from_error(
                            error,
                            self.lat_weights,
                            use_strict_weighted_avg=use_strict_weighted_avg,
                        )
                        step_rmse[c] = np.round(rmse, 4)
                    except Exception as exc:
                        logger.warning(f"Failed to compute RMSE for {c} at step={step}: {exc}")
                        step_rmse[c] = np.nan

                step_rmse_results.append(step_rmse)

                next_cams = {c: pred_raw[i] for i, c in enumerate(pred_vars)}
                cams_queue = (cams_queue[1], next_cams)

            if not step_rmse_results:
                raise RuntimeError("No valid inference results")

            step_rmse_df = pd.DataFrame(step_rmse_results)
            step_rmse_df.to_csv(rmse_file, index=False)
            create_complete_file(output_dir, success=True, message=f"Successfully finished {len(step_rmse_results)} inference steps")
            remove_lock_file(output_dir)
            logger.info(f"RMSE results saved for init time {time_str}")
            return step_rmse_df

        except Exception as exc:
            logger.error(f"Execution failed for init time {time_str}: {exc}")
            create_complete_file(output_dir, success=False, message=f"Execution failed: {str(exc)}")
            remove_lock_file(output_dir)
            return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch ONNX inference for FuXi Zarr data "
            "(CAMS[t-12]/CAMS[t] + non-CAMS lead0/lead1 -> CAMS[t+12])"
        )
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2022-06-01 12:00:00",
        help="Forecast init time t (first-step output is CAMS[t+12], format: YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="2022-06-05 12:00:00",
        help="End of forecast init time range t (format: YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=10,
        help="Number of autoregressive inference steps; 10 steps means 5 days for 12-hour interval data",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=str(DEFAULT_ONNX_PATH),
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=str(DEFAULT_ZARR_PATH),
        help="Input Zarr path",
    )
    parser.add_argument("--data_var", type=str, default="data", help="Main variable name in Zarr")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for outputs (.npy and .csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device preference",
    )
    parser.add_argument(
        "--use_strict_weighted_avg",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
        help="Use strict weighted RMSE (true: sqrt(sum(w*e^2)/sum(w)); false: sqrt(mean(w*e^2)))",
    )
    args = parser.parse_args()

    logger.add(str(BUNDLE_DIR / "inference_log.log"), rotation="500MB", level="INFO")

    pred_vars = list(config.LABEL_VARS)
    noncams_vars: List[str] = []
    for group, vars_ in config.FEATURE_GROUPS.items():
        if group == "temporal":
            continue
        for var in vars_:
            if var not in pred_vars and var not in noncams_vars:
                noncams_vars.append(var)

    all_needed_channels = pred_vars + noncams_vars

    logger.info("Loading Zarr data...")
    try:
        ds = xr.open_zarr(args.zarr_path, consolidated=True, mask_and_scale=False)
    except Exception:
        ds = xr.open_zarr(args.zarr_path, consolidated=False, mask_and_scale=False)
    if args.data_var not in ds:
        raise KeyError(f"Variable {args.data_var} was not found in Zarr. Available variables: {list(ds.data_vars)}")

    data = ds[args.data_var].sel(channel=all_needed_channels)
    data = data.transpose("time", "lead_time", "channel", "lat", "lon").sel(channel=all_needed_channels)

    n_lead_times = len(data["lead_time"].values)
    if n_lead_times < args.infer_steps + 1:
        raise ValueError(
            f"infer_steps={args.infer_steps} requires at least {args.infer_steps + 1} lead_time entries,"
            f"but dataset has only {n_lead_times}"
        )

    start = pd.to_datetime(args.start_time)
    end = pd.to_datetime(args.end_time)
    all_times = pd.DatetimeIndex(data["time"].values)
    valid_time_set = set(t for t in all_times if pd.notna(t))

    init_times = [
        t
        for t in pd.date_range(start=start, end=end, freq="12H")
        if (t in valid_time_set and (t - pd.Timedelta("12H")) in valid_time_set)
    ]
    logger.info(
        f"Generated init time list: {len(init_times)} slots ({start} to {end}, 00/12 only, with guaranteed t-12 context)"
    )

    inferencer = OnnxInferencer(
        onnx_path=args.onnx_path,
        lat_values=data["lat"].values,
        lon_values=data["lon"].values,
        device=args.device,
    )

    all_rmse_results = []
    for init_time in init_times:
        step_rmse_df = inferencer.infer_single_init_time(
            init_time=init_time,
            data=data,
            infer_steps=args.infer_steps,
            output_root=args.output_root,
            pred_vars=pred_vars,
            noncams_vars=noncams_vars,
            use_strict_weighted_avg=args.use_strict_weighted_avg,
        )
        if step_rmse_df is not None:
            all_rmse_results.append(step_rmse_df)

    if not all_rmse_results:
        logger.warning("No valid inference results; aggregated CSV not generated")
        return

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    combined_df = pd.concat(all_rmse_results, ignore_index=True)
    agg_df = combined_df.groupby("step").agg({c: "mean" for c in pred_vars}).round(4)
    agg_df.reset_index(inplace=True)

    agg_output_path = output_root / "aggregated_rmse_results.csv"
    agg_df.to_csv(agg_output_path, index=False)
    rmse_mode = "strict weighted RMSE" if args.use_strict_weighted_avg else "simple weighted RMSE"
    logger.success(f"Aggregated RMSE across all init times saved to: {agg_output_path} (RMSE mode: {rmse_mode})")

    raw_combined_path = output_root / "raw_rmse_results.csv"
    combined_df.to_csv(raw_combined_path, index=False)
    logger.info(f"Raw RMSE results across all init times saved to: {raw_combined_path}")

    logger.info("\n===== RMSE Aggregated by Step =====")
    print(agg_df)


if __name__ == "__main__":
    main()
