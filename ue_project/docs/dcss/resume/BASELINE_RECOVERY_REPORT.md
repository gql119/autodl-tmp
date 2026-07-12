# Clean baseline 恢复报告

状态：**pass**。证据目录：`artifacts/dcss/resume/baseline_20260712_recovery_v1/`。

## 协议

- 数据：VOC mini clean split；800 train / 200 val。本机未找到 Legacy-Best 配置所指的完整 Kaggle-ready VOC07+12 目录，因此不声称这是完整 VOC baseline。
- checkpoint：`checkpoints/voc20_surrogate.pt`，SHA256 `8DE8A0C78C6414AD0BF98052B3BC96C33D8E854A2A2A905D47C8195363975B89`。
- 来源：checkpoint metadata 记录 clean VOC20、YOLOv8n、150 epoch、seed 0；20 类名称正确且 person id 为 14。
- 是否重新训练：否；直接复用已完成的 clean checkpoint。
- 是否收敛：作为 checkpoint 的原 clean 训练已完成 150 epoch；本次只做 clean val 核验。它显著超过 mini split 上的最低可信 Gate，但不用于证明 scratch victim 的收敛轨迹。
- 评估：Ultralytics clean val，使用项目 `extract_map50_per_class` 与 `compute_non_target_map` 逻辑。

## 指标与 Gate

| mAP50_target | mAP50_non_target | mAP50_all | Gate |
| ---: | ---: | ---: | --- |
| 0.934158 | 0.873465 | 0.876500 | pass |

仓库没有可验证的同一 800/200 split clean 历史值；配置里的 0.78 只是参考值。因此采用附件规定的第二条 Gate：`mAP50_non_target >= 0.70`。观察值 0.8735 通过。Stage 1 的 E0 0.0192 因而确认为 15 epoch scratch victim 严重欠拟合，而不是数据 split 本身只能达到低 AP。

运行日志出现 Windows 控制台编码导致的 logger flush 警告，但 validator 完成了 200 张评估并写出完整 per-class AP、`metrics.json` 与比较表；指标未受影响。
