# Stage 1R-C2 最小机制筛选报告

状态：**fail**。1 epoch 结果只用于机制筛选，不用于最终有效性声明。汇总：`artifacts/dcss/resume/diagnostic_summary.csv`；Gate：`diagnostic_gate.json`。

D0 为历史 E4 3-epoch 结果：target energy 0.8672、NT leakage 0.3411、R_shift 2.5425，只引用、不重跑，且不与 1-epoch 候选冒充同阶段比较。

| 实验 | Q 类型 | margin 倍率 | leakage 倍率 | target energy | NT leakage | R_shift | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D1 | 原 Q | 0.5 | 1 | 1.0898 | 0.3948 | 2.7605 | fail |
| D2 | 原 Q | 1 | 2 | 1.3379 | 0.5061 | 2.6433 | fail |
| D3 | 原 Q | 1 | 4 | 1.1981 | 0.5472 | 2.1895 | fail |
| D4 | 原 Q | 0.5 | 2 | 1.1784 | 0.4786 | 2.4624 | fail |
| D5 | no-P_t | 0.5 | 2 | 0.6786 | 0.2113 | 3.2111 | fail |
| D6 | no-P_t | 0.5 | 4 | 0.7950 | 0.2792 | 2.8472 | fail |

全部候选 coverage=0.5595、assignment overlap=1.0、数值有限、最大扰动幅度 0.06274509（不超过 eps），target energy 和 R_shift 也通过。唯一共同失败项是 `NT leakage <= 0.1803`。D5 最接近，但仍超阈值 0.0310；D6 表明把 leakage 权重从 ×2 加到 ×4 没有继续改善，反而升至 0.2792。

结论：no-P_t 明显优于原 Q 候选的 leakage，但最小修复仍未达到预先注册的 random+10% 门槛。根据计划，不扩大网格、不增加模块、不进入正式复验。
