# OA-LGC Real-YOLO / Pilot 进度

最后更新：2026-07-14

起始分支：`codex/oa-lgc-local-chain`

当前分支：`codex/oa-lgc-real-yolo-pilot`

起始 commit：`04448a338239863d71a12198ede2fb08980be3a0`

| 阶段 | 状态 | 起始commit | 结束commit | 证据路径 | 结论 |
| --- | --- | --- | --- | --- | --- |
| C0 云端预检 | pass | `04448a3` | `c713756` | `artifacts/oa_lgc/cloud/20260714_141729_C0_0/` | CUDA、mini VOC、真实 YOLO forward、原生 box/cls/DFL loss 与真实 TAL 诊断均可用 |
| C1 Real YOLO functional adapter | pass | `c713756` | `7cd65bc` | `artifacts/oa_lgc/cloud/20260714_143121_C1_0/` | Mode A/B J=1、Mode C runnability、mixed derivative、cloned buffers 与 base hash 全部通过 |
| C2 TAL/Box/DFL diagnostics | pass | `7cd65bc` | 待本阶段提交 | `artifacts/oa_lgc/cloud/20260714_144529_C2_0/` | coverage median 1.0、low coverage 0、box/DFL 可用、3 个 non-target 类有效 |
| C3 Real YOLO end-to-end smoke | pass | `8145a1a` | `7c0d8ba` | `artifacts/oa_lgc/cloud/20260714_145816_C3_0/` | real-detector engineering chain pass；A-E 与同 seed 复现全部通过 |
| C4 数据协议与 clean baseline | blocked | `7c0d8ba` | `de0ae5d` | `artifacts/oa_lgc/cloud/20260714_151949_C4_0/` | 缺 VOC2012 trainval 与 VOC2007 test，且无法从历史曲线确定 E_pilot |
| C5 Learning-gain pilot | blocked | | | `docs/oa_lgc/cloud/C4_FAILURE_ANALYSIS.md` | C4 Gate 未通过，按协议未启动 |
| C6 决策与云端交接 | pass | `de0ae5d` | 本报告所在最终 commit | `docs/oa_lgc/cloud/OA_LGC_REAL_YOLO_FINAL_REPORT.md` | real YOLO engineering pass, pilot blocked |

状态只使用：`pending`、`running`、`pass`、`partial pass`、`fail`、`blocked`、`interrupted`。

历史 OA-LGC/DCSS/TAUSB 文件、artifact、checkpoint 与 6 个既有 dirty `ue_framework` 文件均纳入保护，不覆盖、不删除。C0 使用唯一 run 目录；失败和被后续审计取代的 run 也原样保留。
