# 推理数据下载

FuXi 推理示例发布在 [AirMPA_Example_Datasets](https://huggingface.co/datasets/JamesWang0712/AirMPA_Example_Datasets/tree/main)。下载 `cams_fuxi_900_451_v1_3day.zarr.tar`，在本目录解包：

```bash
cd Inference/data
tar -xf /path/to/cams_fuxi_900_451_v1_3day.zarr.tar
```

解包后应存在 `Inference/data/cams_fuxi_900_451_v1_3day.zarr/`。`run.sh` 默认读取这个目录，并使用同目录下的 `stats_fc_900_451_2025.json`。FuXi 示例不用于 `Training/` 中的 RA/FC 训练。若远端归档文件名有变化，请以 Hugging Face 的文件列表为准，并保持解包后的目录名不变。

ONNX 权重从 [AirMPA_ONNX_Model](https://huggingface.co/JamesWang0712/AirMPA_ONNX_Model/tree/main) 下载为 `Inference/checkpoints/AirMPA_checkpoint.onnx`，见 [权重下载说明](../checkpoints/README.md)。
