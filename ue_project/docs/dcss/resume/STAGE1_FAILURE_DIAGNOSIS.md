# Stage 1 失败诊断

状态：completed。只读取 E2–E4 既有 artifact，没有重跑 E0–E4。数据证据：`artifacts/dcss/resume/diagnosis_20260712_offline_v2/`。

## 统一诊断表

| 指标 | E2 random | E3 target-only | E4 DCSS | E4 相对 E2 |
| --- | ---: | ---: | ---: | ---: |
| target energy | 0.3601 | 0.5646 | 0.8672 | +0.5072 |
| target outside energy | 2.5433 | 2.3836 | 2.4809 | -0.0624 |
| in-subspace ratio | 0.1251 | 0.1893 | 0.2556 | +0.1305 |
| NT leakage mean | 0.1639 | 0.2763 | 0.3411 | +0.1772 |
| NT leakage max（batch p95 最大值） | 1.9200 | 3.2297 | 2.9137 | +0.9938 |
| R_shift | 2.1967 | 2.0437 | 2.5425 | +0.3458 |
| pairwise cosine | 0.0574 | 0.0648 | 0.0565 | -0.0010 |
| effective rank | 14.9349 | 14.9242 | 14.6122 | -0.3226 |

## 逐项回答

1. E4 只在 P3/`model.15` 上定义机制损失，因此不存在 P4/P5 对 Stage 1 target energy 的贡献。最高 batch 是 epoch 0 / step 32，target energy=3.7482；其后较高项为 epoch 2 / step 133（2.4077）、epoch 2 / step 106（1.9137）、epoch 0 / step 17（1.9126）和 epoch 0 / step 48（1.9009）。按 epoch 均值为 0.9937、0.8125、0.8672，没有单调增长。
2. E4 leakage 最高五类为 bottle/class 4（0.7019）、cat/7（0.6581）、bicycle/1（0.4238）、dog/11（0.4161）、horse/12（0.3436）。
3. 这五类在 400 张 person train 图中的共现率分别为 8.75%、1.75%、8.25%、7.00%、9.25%。cat 泄漏第二但共现率最低，故没有证据表明泄漏只集中于高共现类；共享前景方向比简单共现频率更符合数据。
4. outside penalty 只产生很小改善：E4 outside energy 2.4809 比 E2 低 0.0624，但高于 E3 的 2.3836，且仍是 projected energy 0.8672 的 2.86 倍。按 epoch 为 2.7705、2.3938、2.4809，未持续下降。
5. energy margin=1.0。按 `sqrt(target projected energy) < 1` 计算，epoch 0/1/2 未满足比例为 57.1%/72.0%/71.4%，所以 energy loss 在大多数 batch 中持续激活并推动偏移。
6. E0–E4 未记录各 loss 的独立梯度，不能诚实给出直接 gradient cosine。指标代理中 energy/leakage 的逐 batch 相关系数为 0.012、0.120、-0.085，未显示稳定同向或反向梯度；但总体 E4 同时提高 target energy 和 leakage，说明优化结果存在实际 collateral trade-off。`gradient_conflict_summary.csv` 明确标注为 proxy，不伪装成直接梯度证据。
7. P_t 的因果贡献不能仅由 E4 识别。后续 no-P_t 将 R_sem 从 1.0000 降至 0.6184、R_sel 从 2.3598 提高至 3.6779，且与原 Q 平均主角约 40°；这支持“P_t 把 Q 限制在共享语义方向可能降低选择性”的假设，但最终需看 D5/D6 leakage。
8. 原配置 `lambda_energy=1`、`lambda_leakage=1`、`lambda_outside=0.25`、`lambda_logits=0.8`。由于 margin 在多数 batch 激活，且 outside 权重仅 0.25，target-side 驱动长期存在；不过旧日志缺独立梯度，不能定量声称每步由 energy 梯度绝对主导。
9. pairwise cosine 约 0.0565 且 effective rank 14.61，结合 rank 8 Q，说明 shift 既不形成一致单方向，也不局限在 Q 内；低 cosine 不只是允许的实例差异，因为 outside energy 很大且 in-subspace ratio 仅 0.2556。
10. 失败为多因素：E0 是 baseline underfitting，限制绝对 victim 结论；但同预算 E4 仍不优于 E2，并有更高 leakage，因此另有 mechanism failure 与 selectivity failure；surrogate mechanism 改善没有转化为 victim 优势，构成 transfer failure。clean checkpoint 的 0.8735 non-target 证明 mini split 本身不是低 AP 的根因。

结论：evaluation underfitting 不是唯一或主要可修复解释。即使把绝对评估问题隔离，E4 的 non-target leakage 与相对 random 失败仍独立成立。
