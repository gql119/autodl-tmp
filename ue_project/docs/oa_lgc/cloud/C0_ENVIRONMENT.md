# C0 Environment

## Runtime

- OS：Windows 11 10.0.26200。
- Python：3.12.13。
- Python executable：`F:/autodl-tmp/ue_project/.venv/Scripts/python.exe`。
- PyTorch：2.11.0+cu128。
- CUDA runtime：12.8。
- Ultralytics：8.4.90。
- GPU：NVIDIA GeForce RTX 2070，8,589,606,912 bytes。
- C0 definitive run 后空闲显存：7,361,003,520 bytes。
- C0 峰值分配显存：98,153,984 bytes。
- F 盘空闲：10,233,798,656 bytes（约 9.53 GiB）。

## Paths

- Workspace：`F:/autodl-tmp - 副本/ue_project`。
- Runtime/dependency workspace：`F:/autodl-tmp/ue_project`。
- Dataset：`F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset`。
- Surrogate checkpoint：`F:/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt`。
- Model config：`configs/voc_yolov8n_20cls.yaml`。
- Definitive C0 artifact：`artifacts/oa_lgc/cloud/20260714_141729_C0_0/`。

当前含中文的 workspace 路径通过 PowerShell 管道传给外部解释器时会发生编码损坏，因此运行时输入数据与 checkpoint 使用 SHA 一致的纯 ASCII 原工作区副本。新代码仍从当前分支加载；`PYTHONPATH` 指向当前 workspace。这个边界必须在 C1-C3 命令中继续显式记录。
