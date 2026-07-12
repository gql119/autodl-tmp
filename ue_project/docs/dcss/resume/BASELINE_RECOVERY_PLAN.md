# Clean baseline 恢复计划

## 目标与 Gate

先建立足够可信的 clean evaluation baseline，再决定是否允许正式 Stage 1R 复验。Gate 优先采用仓库记录的 clean VOC 参考 `mAP50_non_target=0.78`，允许绝对差 0.02；若该参考与可用本地 split 不同，则至少要求 0.70，或给出已收敛但仅适用于相对实验的证据。

## 搜索结果与选择

- 完整正式协议记录：Legacy-Best victim 使用 200 epoch、完整 VOC 路径及 unified clean evaluation，历史 non-target mAP50 为 0.7173；该 checkpoint 是 poisoned-data victim，不作为 clean baseline。
- clean checkpoint：`checkpoints/voc20_surrogate.pt`，checkpoint metadata 显示在 clean VOC20 上训练 150 epoch，类别表与 person id 14 一致。
- 本地完整 Kaggle-ready VOC07+12 目录未发现；可解析 clean split 为 `F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset`，800 train / 200 val。
- checkpoint 两份本地副本 SHA256 一致：`8DE8A0C78C6414AD0BF98052B3BC96C33D8E854A2A2A905D47C8195363975B89`。

因此按优先级 A：复用已训练 clean checkpoint，在 200 张 clean val 上使用与项目相同的 Ultralytics per-class AP50 提取逻辑重新评估，不重新训练。

## 验收

输出目录：`artifacts/dcss/resume/baseline_20260712_recovery_v1/`。

必须包含 `config.yaml`、`command.txt`、`environment.txt`、`git_commit.txt`、`reused_checkpoint.txt`、`evaluation.log`、`metrics.json` 和 `baseline_comparison.csv`。若 Gate 未通过，则停止正式复验，但仍允许完成离线诊断、no-P_t 与一轮机制筛选。
