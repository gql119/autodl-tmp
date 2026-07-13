# L4 Learning Gain 报告

状态：**pass**。证据：`artifacts/oa_lgc/local/20260713_221448_428352_L4_seed0/`。

synthetic controllable case：

| 指标 | 值 |
| --- | ---: |
| G_t clean | 0.500000 |
| G_t poison | 0.090000 |
| target gain ratio | 0.180000 |
| L_protect | 0.080000 |
| L_carrier | 0.710000 |
| L_auth | 0.150000 |
| outer gradient to delta | -0.200000 |

`G_t^c > G_t^p` 符号方向通过。class 1 的 authorized gap=0；class 3 用负 clean gain 验证符号保留并计算 gap；class 2 因无样本无效且不进入求和。独立 target 反例记录 `invalid_clean_gain`，ratio 为 null。

9/9 L4 tests 通过；历史+L1–L4 合计 `79 passed in 3.37s`。数学符号、invalid 处理、缺失类排除、梯度有限均通过 Gate。

