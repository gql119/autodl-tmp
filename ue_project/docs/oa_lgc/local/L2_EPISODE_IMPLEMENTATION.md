# L2 Disjoint Support–Query 实现

`oa_lgc/episodes.py` 定义不可变 `ImageRecord`、`Episode` 和 `DisjointEpisodeSampler`。sampler 只从含 person 的不同原始记录中抽取，先检查 `support_size + query_size` 数量，再用 seed/episode/worker 的确定性组合洗牌。四个分支显式携带同一记录对象，`Episode.validate` 在返回前检查 ID、配对与 target presence。

逐类统计分别记录 support/query 正实例数；本阶段的 `class_validity` 要求非目标类在 support 和 query 都达到门槛。缺失类保留 false，不补零冒充有效类。

