# L1 Object-Aligned Carrier 计划

目标是在不修改历史 `dcss/stage15.py` 的前提下，实现唯一可更新 `delta_obj` 到 person 框的标准对象坐标映射。valid support 为 target box soft mask 乘以膨胀 non-target mask 的补集；多 person 累加后按全局 eps 截断。

Gate：单/多 person、无/有/完全 overlap、小目标、越界、非方形框、三种插值、soft edge、面积指标和 delta-only 梯度测试全部通过；artifact 使用唯一 run id 且不覆盖历史目录。

