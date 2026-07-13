# L4 Learning Gain 实现

`oa_lgc/gains.py` 提供：

- `target_learning_gain`：只有 `G_t^c > min_valid_clean_gain` 且 finite 时才计算 ratio；否则明确 `invalid_clean_gain`。
- `authorized_learning_gain`：按 class 独立检查 support/query count、finite 与 `abs(G_k^c)` 门槛；有效项求和，不平均，不填零。
- `carrier_query_loss`：只接受未见 poison query 的 target loss，并拒绝非有限值。

负 clean gain 不被静默丢弃：若绝对值达到门槛，会按定义计算 gap并保留负号日志。

