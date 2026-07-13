# L6 Smoke 失败分析

No blocking failure was triggered.

- 日期：2026-07-13；branch：`codex/oa-lgc-local-chain`；起始 commit：`161e529`。
- 环境：Python 3.12.13、PyTorch 2.11.0+cu128；smoke 使用 CPU detector proxy。
- 配置：`configs/oa_lgc/local/smoke.yaml`。
- 数据：`F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset`；checkpoint 未用于 proxy virtual update。
- 未验证项：真实 YOLO TAL/DFL 和 full-model virtual update，分类为 `blocked by local resources` 的云端 handoff，不影响本地 proxy engineering Gate。
- 指标证据：`artifacts/oa_lgc/local/20260713_222449_255485_L6_seed0/`。
- 历史代码/实验影响：无；所有输出使用唯一目录。

