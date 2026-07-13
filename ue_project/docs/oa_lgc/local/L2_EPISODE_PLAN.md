# L2 Disjoint Support–Query 计划

从 mini VOC 原始 image ID 构造 `S_c/S_p/Q_c/Q_p`。clean/poison 仅表示同一记录的两种像素版本，不能改变 `source_id`；support 与 query 在选择原始记录时一次性切分，禁止数据不足时复用。

Gate：同 seed 可复现、不同 seed 可变化、ID overlap=0、clean/poison pair 对齐、两侧有 person、逐类有效性可统计、数据不足明确失败、augmentation 保持 ID、多 worker 每个 episode 内仍严格互斥。

