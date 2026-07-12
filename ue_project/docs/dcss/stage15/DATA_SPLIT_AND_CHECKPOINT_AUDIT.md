# 数据划分与 checkpoint 审计

状态：**metadata insufficient**。证据目录：`artifacts/dcss/stage15/audit_20260712_v1/`。

## Checkpoint

- `checkpoints/voc20_surrogate.pt` SHA256：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`。
- metadata 只记录 `/kaggle/working/voc20_clean.yaml`、YOLOv8n、150 epoch、seed 0；该 YAML 本机不存在。
- checkpoint 未内嵌 train/val image ID 或 manifest，故无法重建其训练与验证集合。
- Stage 0：作为 frozen surrogate 参与全部 feature/gradient statistics 与 Q 构造。
- Stage 1：作为 frozen surrogate 参与 poison optimization；不是 E0–E4 scratch victim 初始化。
- Stage 1R：曾仅用于 evaluation，不能代替重新训练的 clean victim。

## Mini split

- mini train：800 IDs；mini val：200 IDs。
- mini train ∩ mini val：0。
- checkpoint train ∩ mini val：unknown。
- checkpoint val ∩ mini val：unknown。

## 结论

审计分类为 `metadata insufficient`，并存在无法排除的 evaluation leakage risk。此前 checkpoint 在 mini val 上的 target/non-target mAP50=0.9342/0.8735 只能说明 checkpoint 对该集合性能高，不能称为独立 clean baseline。Stage 1.5 clean victim 必须从与 mini val 独立的固定初始化重新训练。
