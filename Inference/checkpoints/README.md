# ONNX 权重下载

从 [AirMPA_ONNX_Model](https://huggingface.co/JamesWang0712/AirMPA_ONNX_Model/tree/main) 下载 `AirMPA_checkpoint.onnx`，放在本目录：

```text
Inference/checkpoints/AirMPA_checkpoint.onnx
```

`Inference/run.sh` 默认读取此文件。若下载后文件名不同，请重命名为 `AirMPA_checkpoint.onnx`，或运行时通过 `ONNX_PATH=/path/to/model.onnx bash Inference/run.sh` 指定。模型权重不纳入代码仓库。
