# C0 Preflight Audit

## Gate 结论

状态：`pass`。

权威证据目录：`artifacts/oa_lgc/cloud/20260714_141729_C0_0/`。

| Gate | 结果 | 证据 |
| --- | --- | --- |
| PyTorch / Ultralytics 可导入 | pass | PyTorch 2.11.0+cu128；Ultralytics 8.4.90 |
| GPU 可用 | pass | NVIDIA GeForce RTX 2070，8.00 GiB；smoke 后空闲约 6.86 GiB |
| mini VOC 可读 | pass | train 800、val 200；图像与标签计数一致；stem overlap 0 |
| 真实 YOLO forward | pass | 输入 320，输出 `(1, 24, 2100)`，全为有限值 |
| 原生 detection loss | pass | box=2.026055、cls=1.076285、DFL=1.543931，总和=4.646270 |
| 真实 TAL 诊断 | pass | 40 个 foreground units；target score mass=23.360126 |
| 历史文件未覆盖 | pass | 所有输出使用新建唯一目录；既有 dirty/untracked 文件未改写或删除 |
| 最小资源 | pass | F 盘空闲 10,233,798,656 bytes；峰值显存约 93.6 MiB |

## 仓库与分支

- 起始 commit：`04448a338239863d71a12198ede2fb08980be3a0`。
- 当前分支：`codex/oa-lgc-real-yolo-pilot`。
- `git fetch origin` 与 `git pull --ff-only origin codex/oa-lgc-local-chain` 成功，`FETCH_HEAD` 精确等于起始 commit。
- 本地未生成 `origin/codex/oa-lgc-local-chain` remote-tracking ref；这是引用显示问题，不影响按 `FETCH_HEAD` 核验基线或创建新分支。
- 既有 6 个 dirty tracked 文件保持不变：`ue_framework/launch_one.py`、`paths.py`、`runtime.py`、`stages/aggregate.py`、`stages/evaluate.py`、`stages/train_victim.py`。

## 模型与 checkpoint

- 可执行 checkpoint：`F:/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt`。
- 当前副本中的同名 checkpoint 与上述文件 SHA256 均为 `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`。
- checkpoint 大小：6,243,306 bytes；保存版本 8.4.32；训练配置记录 150 epochs、SGD、seed 0、VOC 20 类。
- 模型：`ultralytics.nn.tasks.DetectionModel`；参数 3,014,748；Detect head 是 `model.22`。
- classification branch：严格前缀 `model.22.cv3.`，24 tensors / 373,308 parameters。
- box/distribution branch：严格前缀 `model.22.cv2.`，24 tensors / 381,888 parameters。
- DFL integral module：严格前缀 `model.22.dfl.`，1 tensor / 16 个固定积分权重。
- buffer：171 tensors / 10,457 elements；BatchNorm modules：57。

## 数据与接口边界

- mini VOC 根目录：`F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset`。
- 当前仅有 train/val manifests；没有独立 test manifest。它满足 C0-C3 工程 smoke，但不满足 C4 正式 pilot 协议。
- 原生训练损失由 `DetectionModel.loss` 提供并真实包含 TAL、box、classification、DFL。
- `HijackedV8Loss.last_real_assign` 用于 TAL assignment 诊断；不得把它的手工解码路径冒充原生 DFL loss。
- checkpoint 加载后参数处于冻结状态；真实 virtual update 必须显式开启所选参数的梯度。
- 旧 checkpoint 的运行时 `model.args` 只有少量字段；当前 Ultralytics 需显式合并完整默认配置，实际 box/cls/DFL gains 为 7.5/0.5/1.5。

## 审计范围

已审计 `oa_lgc/`、本地 OA-LGC tests、现有 functional-call proxy、`ue_framework/ultra/hijacked_loss.py`、victim train/evaluate/metrics、DCSS unit partition/feature hooks/stage 1.5、VOC 模型配置、mini VOC 配置与 surrogate checkpoint。没有修改受保护核心训练和评估文件。
