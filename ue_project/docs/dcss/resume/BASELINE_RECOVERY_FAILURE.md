# Clean baseline 恢复失败记录

没有 baseline 指标失败，Gate 为 pass。

唯一异常是 2026-07-12 Windows 控制台编码环境下，Ultralytics logger 在输出 per-class 表时出现 `OSError: [Errno 22] Invalid argument`。分类为 environment warning，不是 evaluation failure：200 张验证图已全部处理，per-class AP、`metrics.json`、`baseline_comparison.csv` 均完整写出。未修改 checkpoint、数据或 E0–E4。
