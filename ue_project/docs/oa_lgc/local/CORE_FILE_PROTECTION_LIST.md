# OA-LGC 核心文件保护列表

在 L7 清理完成前，以下内容不得删除、重命名或被历史不兼容改写。

## 数据、模型与正式评估

- `ue_framework/data_utils.py`：VOC/YOLO 标注解析、图像读写与数据枚举。
- `configs/voc_yolov8n_20cls.yaml`：VOC20 YOLOv8n 风格模型定义。
- `ue_framework/stages/train_victim.py`：victim 训练入口（历史 dirty 文件）。
- `ue_framework/stages/evaluate.py`：clean evaluation（历史 dirty 文件）。
- `ue_framework/stages/aggregate.py`：统一指标聚合（历史 dirty 文件）。
- `ue_framework/metrics_utils.py`：统一指标实现。
- `ue_framework/launch_one.py`、`ue_framework/runtime.py`、`ue_framework/paths.py`：历史正式运行入口与路径管理（均为历史 dirty 文件）。

## Detector hooks、TAL 与既有 carrier

- `dcss/feature_hooks.py`：FPN hook。
- `dcss/unit_partition.py`：TAL target/non-target unit partition 与 PAG。
- `ue_framework/ultra/hijacked_loss.py`：真实 TAL assignment 拦截。
- `ue_framework/methods/alce_acgt.py`：PAG 和 FPN gate 映射。
- `ue_framework/methods/tausb_universal.py`：TAUSB carrier、poison optimization 与 materialization。
- `ue_framework/support.py`：既有 bbox/mask support。
- `dcss/stage15.py`：Stage 1.5 历史 object-aligned carrier 与约束方向实现。

## OA-LGC 新核心（创建后自动纳入保护）

- `oa_lgc/carrier.py`：正式 object-aligned carrier。
- `oa_lgc/episodes.py`：严格 disjoint support/query sampler。
- `oa_lgc/virtual_update.py`：functional multi-step virtual update。
- `oa_lgc/gains.py`：target 与 per-class authorized learning gain。
- `oa_lgc/objective.py`：OA-LGC core objective、projection 和 checkpoint。
- `oa_lgc/smoke.py`：本地 mini VOC 工程 smoke 入口。
- `configs/oa_lgc/local/` 下所有已使用配置。
- `tests/test_oa_lgc_*.py`：OA-LGC 回归测试。

## 历史研究证据

- `dcss/` 全目录中的 Stage 0/1/1.5 原始代码。
- `configs/dcss/` 下全部配置。
- `docs/dcss/` 下全部报告和恢复文档。
- `artifacts/dcss/` 下全部历史产物。
- `scripts/dcss_*.py` 与 `scripts/dcss_stage15_*.py`。
- `tests/test_dcss.py`、`tests/test_dcss_stage15.py`。
- `checkpoints/`、`runs/`、数据文件和任何 `.pt` 实验权重。
- AGENTS.md 中列明的 TAUSB best 配置及方法实现。

## 删除原则

本任务默认不删除任何文件。只有 L7 清理表逐文件记录、确认无引用且相关测试通过后，才允许一次删除一个明确文件；无法 100% 确认时记录 `retained due to uncertain dependency`。

