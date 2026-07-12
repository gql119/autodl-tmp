# DCSS Stage 1R 续作进度

最后更新：2026-07-12

边界：只执行 Stage 1R-A 至 Stage 1R-E；不重跑 E0–E4，不进入或实现 Stage 2，不覆盖历史 artifact。

| 工作项 | 状态 | 证据路径 | 结论 |
| --- | --- | --- | --- |
| 中断状态恢复 | completed | `docs/dcss/resume/DCSS_RESUME_PROGRESS.md` | 分支与起点已核验；历史脏工作区已保留 |
| 已有结果核验 | completed | `docs/dcss/DCSS_CURRENT_WORK_REPORT.md`、`artifacts/dcss/stage1/stage1_gate.json` | 接受 Stage 0 pass、Stage 1 fail，不重新包装 |
| Clean baseline 核验 | completed | `artifacts/dcss/resume/baseline_20260712_recovery_v1/` | non-target mAP50=0.8735，Gate pass |
| Stage 1 失败诊断 | completed | `artifacts/dcss/resume/diagnosis_20260712_offline_v2/` | underfitting 是绝对结论混杂因素；另有独立 mechanism/selectivity/transfer failure |
| no-P_t 消融 | completed | `artifacts/dcss/resume/no_pt_20260712_v2/` | 工程 Gate pass，进入机制筛选 |
| energy/leakage 筛选 | completed | `artifacts/dcss/resume/diagnostic_gate.json` | D1–D6 全部因 leakage 超阈值而 fail |
| 正式复验 | not run | `docs/dcss/resume/STAGE1_REPAIR_REPORT.md` | 机制前置 Gate 未通过 |
| 最终 Gate | completed | `artifacts/dcss/resume/diagnostic_gate.json` | Stage 1 repair fail |

## 恢复状态

- 当前分支：`codex/dcss-stage0-stage1`。
- 起始 commit：`e02a752 将训练噪声batchsize改为32`。
- 起始工作区存在历史未提交修改：6 个 `ue_framework` 文件有 tracked 修改；`.venv/`、`artifacts/`、`configs/dcss/`、`dcss/`、`docs/`、`scripts/`、`tests/` 等为未跟踪内容。它们构成中断前成果或更早工作，不在续作中 reset、删除或覆盖。
- 历史实验覆盖：否。新输出仅写入 `docs/dcss/resume/` 与 `artifacts/dcss/resume/`。
- 当前副本解释器缺少所需运行环境；续用只读的 `F:/autodl-tmp/ue_project/.venv/Scripts/python.exe`，不修改另一仓库代码。

## 已核验历史结论

- Stage 0：pass；推荐 P3 / `model.15` / rank 8，`R_sel=2.3598`、`R_sem=1.0000`、`R_stab=0.5619`。
- Stage 1：fail；E4 target energy 0.8672、NT leakage 0.3411、`R_shift=2.5425`，但未优于 random 或 target-only。
- E0 clean mini victim 在 15 epoch 下严重欠拟合：target/non-target mAP50 为 0.1410/0.0192。
- Stage 0 不重跑；Stage 2 不进入。
