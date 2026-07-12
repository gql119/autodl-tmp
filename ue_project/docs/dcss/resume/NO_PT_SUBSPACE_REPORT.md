# no-P_t 子空间报告

状态：engineering Gate **pass**。使用既有 Stage 0 完整统计构造独立 `no_pt` 子空间，不替代 Stage 0 原结论。证据：`artifacts/dcss/resume/no_pt_20260712_v2/`。

求解式为 `C_t q = lambda (C_nt + mu I) q`，固定 P3/`model.15`、rank 8、`mu=1e-4`，未重新收集 Stage 0 数据。

| R_sel | R_sem | R_stab | random mean ± std | Q^TQ 最大误差 |
| ---: | ---: | ---: | ---: | ---: |
| 3.6779 | 0.6184 | 0.5452 | 0.8318 ± 0.1601 | 6.66e-16 |

- 数值有限：pass。
- 正交误差 <=1e-4：pass。
- `R_sel > random_mean + 2*random_std`：3.6779 > 1.1519，pass。
- `R_stab` 可计算：0.5452，pass。
- R_sem 相对原 Q 从 1.0000 降低 0.3816。
- 与原 Q 的 8 个 principal angles 为 22.99°、24.29°、29.63°、35.52°、45.20°、46.77°、54.21°、61.35°，均值 40.00°。
- person/non-target 梯度投影能量为 0.003632/0.000988。

结论：no-P_t 值得进入有限机制筛选，但这只证明其 Stage 0 选择性与稳定性，不证明 victim 可学习性保护有效。
