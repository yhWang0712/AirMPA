#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_TIME="${START_TIME:-2022-06-01 12:00:00}"
END_TIME="${END_TIME:-2022-06-02 12:00:00}"
INFER_STEPS="${INFER_STEPS:-4}"
DEVICE="${DEVICE:-cuda}"
STRICT_WEIGHTED_AVG="${STRICT_WEIGHTED_AVG:-true}"

ZARR_PATH="${ZARR_PATH:-${BUNDLE_ROOT}/data/cams_fuxi_900_451_v1_3day.zarr}"
ONNX_PATH="${ONNX_PATH:-${BUNDLE_ROOT}/checkpoints/AirMPA_checkpoint.onnx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BUNDLE_ROOT}/output}"

cd "${BUNDLE_ROOT}"

"${PYTHON_BIN}" ./inference.py \
  --zarr_path "${ZARR_PATH}" \
  --start_time "${START_TIME}" \
  --end_time "${END_TIME}" \
  --infer_steps "${INFER_STEPS}" \
  --onnx_path "${ONNX_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --use_strict_weighted_avg "${STRICT_WEIGHTED_AVG}"
