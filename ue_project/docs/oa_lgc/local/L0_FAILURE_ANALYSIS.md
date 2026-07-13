# L0 失败分析

## 环境发现

- 日期时间：2026-07-13（Asia/Hong_Kong）。
- branch：`codex/oa-lgc-local-chain`。
- commit：`b72672a1505a6ea76acbbedca4f404b38ab4b021`。
- 当前副本 Python：`.venv` Python 3.12.13。
- CUDA 环境：当前副本 `.venv` 无 torch，无法直接检查；同机只读解释器为 PyTorch 2.11.0+cu128、CUDA available、RTX 2070。
- 配置/数据/checkpoint：mini VOC 800/200；`checkpoints/voc20_surrogate.pt`。
- 失败阶段：L0 environment audit。
- 预期行为：当前副本解释器可运行 PyTorch/Ultralytics。
- 实际行为：当前副本 `.venv` 缺两项依赖；WindowsApps `python.exe` 也不可用。
- traceback：无 Python traceback；解释器命令退出 1。
- 指标证据：外部同机解释器成功导入，历史测试 `39 passed in 4.80s`。
- 初步原因：副本 `.venv` 未安装 ML 依赖，历史项目一直显式复用同机原工作区解释器。
- 已检查项：Python launcher、两处 `.venv`、torch/CUDA/Ultralytics 版本、数据与 checkpoint 存在性。
- 修复内容：不修改环境；后续命令显式使用只读同机解释器。
- 修复后结果：历史测试全部通过。
- 是否影响历史代码：否。
- 是否影响历史实验：否。
- 是否需要云端验证：完整 YOLO functional meta-update 需要后续云端验证；本地工程 proxy 不需要。
- 分类：environment failure（已绕过，非 blocking）。

No blocking failure was triggered.
