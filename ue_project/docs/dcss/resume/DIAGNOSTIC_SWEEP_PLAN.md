# Stage 1R-C2 最小机制筛选计划

固定 P3/`model.15`、rank 8、forced pseudo fallback Legacy-Best carrier/support、seed 0、800/200 mini split、batch 8、eps 16/255 与 materialization 定义。D0 仅引用 E4，不重跑；D1–D6 各运行 1 epoch protected-data generation，不训练 victim。

| 实验 | Q | energy margin 倍率 | leakage 权重倍率 |
| --- | --- | ---: | ---: |
| D1 | 原 Q | 0.5 | 1 |
| D2 | 原 Q | 1 | 2 |
| D3 | 原 Q | 1 | 4 |
| D4 | 原 Q | 0.5 | 2 |
| D5 | no-P_t Q | 0.5 | 2 |
| D6 | no-P_t Q | 0.5 | 4 |

Gate：coverage >= 0.50；target energy >= 0.3601；NT leakage <= 0.1803；R_shift >= 2.0；数值有限；扰动幅度不超过 E4 的 eps。
