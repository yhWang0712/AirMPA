# AirMPA

English | [简体中文](README_CN.md)

AirMPA provides code for global air-pollution forecasting: pretraining on CAMS reanalysis (RA), post-training on CAMS forecast (FC), and standalone autoregressive ONNX inference. The training stages use pollutant fields from the previous two 12-hour time steps together with meteorological and emission features to predict the next step.

## Repository layout

| Path | Contents |
| --- | --- |
| `Training/PRE_TRAINING/` | RA pretraining, evaluation, and prediction |
| `Training/POST_TRAINING/` | FC post-training, evaluation, and prediction |
| `Training/DATA/` | Five-day RA/FC examples and normalization statistics; **FuXi is not used for training** |
| `Inference/` | ONNX inference code, FuXi example data, and local outputs |

The training example Zarr stores cover June 1-5, 2022, at 12-hour intervals: 10 time steps, 78 channels, and a 451 x 900 grid. The statistics in `Training/DATA/Normalization/` were calculated from the full datasets, not the five-day slices. The FuXi example resides in `Inference/data/` and is used only for inference.

`.gitignore` excludes Zarr data and model weights. After cloning, download and place the required files using the instructions for [training data](Training/DATA/README.md), [inference data](Inference/data/README.md), and the [ONNX model](Inference/checkpoints/README.md).

## Environment

Use Python 3.10 or newer. Install the shared dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The requirements install CPU ONNX Runtime by default. For CUDA inference, install a compatible `onnxruntime-gpu` in place of `onnxruntime`; GPU training likewise requires a PyTorch build compatible with your CUDA installation. Zarr is constrained to version 2.x.

## Two-stage training

The training configurations are `Training/PRE_TRAINING/config.py` and `Training/POST_TRAINING/config.py`. Their default data and statistics paths are relative to `Training/`. The five-day examples are suitable for checking the data layout, but they do not cover the configured training and validation periods. Supply complete RA/FC datasets and review the date ranges before a full training run.

```bash
python Training/PRE_TRAINING/train.py
python Training/PRE_TRAINING/eval.py
python Training/POST_TRAINING/train.py
python Training/POST_TRAINING/eval.py
```

To regenerate statistics for new data, first run `dataset.py --compute-stats` in the corresponding stage. Before post-training, check that `PRETRAINED_CKPT_PATH` points to the checkpoint produced by pretraining; weights are not included in this repository. Checkpoints, logs, and results are written under each stage directory. Both stages also provide `predict.py`.

## ONNX inference

`Inference/run.sh` resolves its default Zarr, ONNX model, and output paths from the script location, so it can be called from any working directory:

```bash
bash Inference/run.sh
```

The default input is `Inference/data/cams_fuxi_900_451_v1_3day.zarr`, the model is `Inference/checkpoints/AirMPA_checkpoint.onnx`, and the run window is June 1, 2022, 12:00 through June 2, 2022, 12:00 with `INFER_STEPS=4`. Results are written to `Inference/output/`. Override defaults with `ZARR_PATH`, `ONNX_PATH`, `OUTPUT_ROOT`, `START_TIME`, `END_TIME`, `INFER_STEPS`, `DEVICE`, or `PYTHON_BIN`. For example:

```bash
DEVICE=cpu INFER_STEPS=2 bash Inference/run.sh
```

Default file paths are derived from script locations; additional data and weights can be specified in configuration or on the command line.
