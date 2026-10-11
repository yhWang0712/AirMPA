# 训练数据

## 下载示例

RA/FC 示例数据发布在 [AirMPA_Example_Datasets](https://huggingface.co/datasets/JamesWang0712/AirMPA_Example_Datasets/tree/main)。下载其中的 `cams_ra_900_451_v1.zarr.tar` 和 `cams_fc_900_451_v1.zarr.tar`，在本目录解包：

```bash
cd Training/DATA
tar -xf /path/to/cams_ra_900_451_v1.zarr.tar
tar -xf /path/to/cams_fc_900_451_v1.zarr.tar
```

解包后应分别存在 `Training/DATA/cams_ra_900_451_v1.zarr/` 和 `Training/DATA/cams_fc_900_451_v1.zarr/`。如果 Hugging Face 上的归档文件名不同，请以其文件列表为准；保持解包后的 Zarr 目录名与训练配置一致。同一数据仓库中的 FuXi 示例只供 `Inference/` 使用。

本目录只提供 CAMS 再分析（RA）和 CAMS 预报（FC）数据的五天示例，**不包含 FuXi 训练数据**：

| 文件 | 用途 |
| --- | --- |
| `cams_ra_900_451_v1.zarr/` | `PRE_TRAINING` 的 RA 示例 |
| `cams_fc_900_451_v1.zarr/` | `POST_TRAINING` 的 FC 示例 |
| `Normalization/stats_ra_900_451.json` | RA 归一化统计 |
| `Normalization/stats_fc_900_451.json` | FC 归一化统计 |

两个 Zarr 均覆盖 2022-06-01 至 2022-06-05，每 12 小时一个时次，包含全部 78 个通道和 451 x 900 空间网格。统计文件来自完整数据集，并非由示例切片计算。

五天示例不覆盖训练配置中的完整日期范围。正式训练时需提供完整 RA/FC 数据，并核对 `PRE_TRAINING/config.py` 与 `POST_TRAINING/config.py` 中的数据路径、统计文件和日期范围。FuXi 样本仅用于独立 ONNX 推理，放在仓库根目录的 `Inference/data/`。
