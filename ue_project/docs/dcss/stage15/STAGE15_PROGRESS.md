# DCSS Stage 1.5 进度

最后更新：2026-07-12

当前结论边界：Stage 1 fixed-Q weighted-loss implementation failed；DCSS hypothesis remains unresolved。禁止进入 Stage 2。

| 工作项 | 状态 | 证据路径 | 结论 |
| --- | --- | --- | --- |
| 工作区恢复 | completed | `docs/dcss/stage15/STAGE15_PROGRESS.md` | 分支 `codex/dcss-stage0-stage1`，起点 `ef656a9`；历史脏文件保留 |
| checkpoint审计 | completed | `artifacts/dcss/stage15/audit_20260712_v1/` | metadata insufficient；无法排除 evaluation leakage risk |
| 数据划分审计 | completed | `artifacts/dcss/stage15/audit_20260712_v1/` | mini train/val overlap=0 |
| clean victim baseline | running | `artifacts/dcss/stage15/clean_victim_20260712_v1/` | 固定 scratch 初始化，100 epoch |
| 等预算机制实验 | pending |  |  |
| 梯度几何诊断 | pending |  |  |
| constrained update | pending |  | 仅显著冲突时启用 |
| carrier消融 | pending |  |  |
| victim正式决策实验 | pending |  | 仅前置 Gate 通过时启用 |
| 最终Gate | pending |  |  |

## 恢复记录

- 历史 tracked 修改：`ue_framework/launch_one.py`、`paths.py`、`runtime.py`、`stages/aggregate.py`、`evaluate.py`、`train_victim.py`。
- 历史未跟踪目录包括 `.venv/`、`artifacts/`、`configs/dcss/`、既有 Stage 0/1 docs/scripts、runs 与 figures。
- 不 reset、不 clean、不删除；Stage 1.5 新输出只写入 `artifacts/dcss/stage15/` 与 `docs/dcss/stage15/`。
