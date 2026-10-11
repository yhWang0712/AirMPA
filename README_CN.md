# AirMPA

[English](README_EN.md) | 简体中文

全球空气污染物预报代码，包含 CAMS 再分析数据预训练、CAMS 预报数据后训练，以及独立的 ONNX 自回归推理程序。训练使用过去两个 12 小时时次的污染物场，结合气象和排放特征预测下一时次。

## 目录

| 路径 | 内容 |
| --- | --- |
| `Training/PRE_TRAINING/` | RA 预训练、评估和预测 |
| `Training/POST_TRAINING/` | FC 后训练、评估和预测 |
| `Training/DATA/` | RA/FC 五天示例和归一化统计；**训练不使用 FuXi 数据** |
| `Inference/` | ONNX 推理脚本、FuXi 推理示例和本地输出目录 |

训练示例 Zarr 均覆盖 2022-06-01 至 2022-06-05，每 12 小时一个时次，共 10 个时次、78 个通道，空间网格为 451 x 900。统计文件位于 `Training/DATA/Normalization/`，来自完整数据集，不是用这五天重新计算的。推理用的 FuXi 数据单独位于 `Inference/data/`，不属于训练输入。

当前 `.gitignore` 排除了 Zarr 数据和模型权重；克隆代码后需自行放置所需文件。下载及放置方法见 [训练数据](Training/DATA/README.md)、[推理数据](Inference/data/README.md) 和 [ONNX 权重](Inference/checkpoints/README.md)。

## 环境

使用 Python 3.10 或更新版本，从仓库根目录安装统一依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 默认安装 CPU 版 ONNX Runtime。需要 CUDA 推理时，按本机 CUDA 环境安装兼容的 `onnxruntime-gpu` 替代 `onnxruntime`；GPU 训练同样需要兼容 CUDA 的 PyTorch。Zarr 依赖限定为 2.x。

## 两阶段训练

两个训练配置分别在 `Training/PRE_TRAINING/config.py` 和 `Training/POST_TRAINING/config.py`。默认数据路径与统计文件路径以 `Training/` 为基准。五天示例可用于检查数据格式，但不覆盖配置中的完整训练和验证日期；正式训练前须提供完整 RA/FC 数据并核对日期范围。

```bash
python Training/PRE_TRAINING/train.py
python Training/PRE_TRAINING/eval.py
python Training/POST_TRAINING/train.py
python Training/POST_TRAINING/eval.py
```

需要针对新数据重新生成统计量时，先运行相应目录的 `dataset.py --compute-stats`。后训练前请确认 `PRETRAINED_CKPT_PATH` 指向预训练产生的权重；权重文件不随仓库提供。训练产物写入各阶段目录下的 `checkpoints/`、`logs/` 和 `result/`。两个阶段还提供 `predict.py`。

## ONNX 推理

`Inference/run.sh` 以脚本所在目录定位默认 Zarr、ONNX 权重和输出目录，可从任意工作目录调用：

```bash
bash Inference/run.sh
```

默认输入为 `Inference/data/cams_fuxi_900_451_v1_3day.zarr`，模型为 `Inference/checkpoints/AirMPA_checkpoint.onnx`，运行窗口为 2022-06-01 12:00 至 2022-06-02 12:00，`INFER_STEPS=4`。结果写入 `Inference/output/`。可通过 `ZARR_PATH`、`ONNX_PATH`、`OUTPUT_ROOT`、`START_TIME`、`END_TIME`、`INFER_STEPS`、`DEVICE` 和 `PYTHON_BIN` 覆盖默认值。例如：

```bash
DEVICE=cpu INFER_STEPS=2 bash Inference/run.sh
```

所有代码中的默认文件位置均由脚本位置推导；额外数据和权重可在配置或命令行中指定。
