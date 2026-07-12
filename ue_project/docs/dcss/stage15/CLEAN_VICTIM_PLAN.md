# 收敛 clean victim 计划

- 数据：固定 mini 800 train / 200 clean val，二者 ID overlap=0。
- 初始化：`configs/voc_yolov8n_20cls.yaml` scratch，seed 0；本地没有同架构标准 `yolov8n.pt`，不使用存在 split 泄漏不确定性的 surrogate。
- 协议：640 px、batch 16、SGD、AMP、cosine LR、与 E0–E4 相同模型结构和基础训练入口。
- 先执行 100 epoch（已有独立 50 epoch证据仍明显欠拟合）；若最近20 epoch仍上升则延长至150。
- 每10 epoch统一 clean val，并保存初始化 hash、训练 loss 与 per-class AP50。
