# DCSS Stage 1.5 进度

最后更新：2026-07-12

当前结论边界：Stage 1 fixed-Q weighted-loss implementation failed；DCSS hypothesis remains unresolved。禁止进入 Stage 2。

| 工作项 | 状态 | 证据路径 | 结论 |
| --- | --- | --- | --- |
| 工作区恢复 | completed | `docs/dcss/stage15/STAGE15_PROGRESS.md` | 分支 `codex/dcss-stage0-stage1`，起点 `ef656a9`；历史脏文件保留 |
| checkpoint审计 | completed | `artifacts/dcss/stage15/audit_20260712_v1/` | metadata insufficient；无法排除 evaluation leakage risk |
| 数据划分审计 | completed | `artifacts/dcss/stage15/audit_20260712_v1/` | mini train/val overlap=0 |
| clean victim baseline | completed-fail | `artifacts/dcss/stage15/clean_victim_20260712_v1/` | 稳定但non-target=0.2485<0.70 |
| 等预算机制实验 | completed-fail | `artifacts/dcss/stage15/equal_budget_gate.json` | M3 leakage高于M0×1.10 |
| 梯度几何诊断 | completed | `artifacts/dcss/stage15/gradient_geometry_20260712_v1/` | 显著梯度冲突 |
| constrained update | completed-fail | `artifacts/dcss/stage15/diagnostic_M4_noPt_legacy_constrained_20260712/` | leakage下降但target retention不足 |
| carrier消融 | completed-pass | `artifacts/dcss/stage15/diagnostic_M5_random_object_weighted_20260712/` | Legacy carrier coupling显著 |
| victim正式决策实验 | not-run | `docs/dcss/stage15/VICTIM_DECISION_REPORT.md` | 前置Gate失败 |
| 最终Gate | completed | `docs/dcss/stage15/STAGE15_FINAL_REPORT.md` | formal decision incomplete |

## 恢复记录

- 历史 tracked 修改：`ue_framework/launch_one.py`、`paths.py`、`runtime.py`、`stages/aggregate.py`、`evaluate.py`、`train_victim.py`。
- 历史未跟踪目录包括 `.venv/`、`artifacts/`、`configs/dcss/`、既有 Stage 0/1 docs/scripts、runs 与 figures。
- 不 reset、不 clean、不删除；Stage 1.5 新输出只写入 `artifacts/dcss/stage15/` 与 `docs/dcss/stage15/`。
