---
spec_id: TAUSB-SDH-E2E-V0-SPARSE-E200-v1
title: SDH V0 稀疏物化 paired E200 完整训练时域验证
status: approved
experiment_type: end_to_end_full_horizon
parent_specs:
  - TAUSB-SDH-E2E-V0-MAP50-v1
  - TAUSB-SDH-E2E-V0-SPARSE-E20-v3
exp_id: TAUSB-SDH-E2E-V0-S0-E200-SPARSE
run_id: SPARSE-E200-S0-R1
csv: issues/TAUSB-SDH-E2E-V0-SPARSE-E200-v1.csv
created: 2026-08-12
approved: 2026-08-12
approval_evidence: user approved with 9-hour wall cap and mandatory preservation/reporting of every terminal outcome
---

# SDH V0 稀疏物化 paired E200 完整训练时域验证

## 1. 问题锚点

- 研究目标：在 Pascal VOC 目标检测中，使目标类 `person` 难以被 clean-test 检测，同时尽量保持其余 19 类。
- 触发证据：`TAUSB-SDH-E2E-V0-S0-E20-SPARSE / SPARSE-E20-S0-R1` 已完成真实 paired E20：
  - person AP50：`0.772072 -> 0.128312`，drop=`0.643759`；
  - non-target macro AP50：`0.591046 -> 0.495146`，drop=`0.095900`；
  - 预注册结论：`inconclusive_tradeoff`；
  - person-free non-target drop=`0.047899`，person-cooccur non-target drop=`0.132122`。
- 本轮可判别问题：当 C0/M1 victim 都从相同随机初始化重新训练到 200 epochs 时，E20 的目标类崩塌是否持续；非目标类是恢复、维持，还是进一步下降？
- “完整实验”的边界：完成 frozen P1 物化、fresh C0/M1 E200、clean VOC20 逐类评估与配对比较；不重新训练 hiding encoder 或重新优化 P1。
- 非目标：不修改 secret、carrier、D-LFC、CICR、CGR、NLA、support、扰动预算、数据 split、victim 优化器或 AP50 计算；不做 EOT、JPEG/blur/gray、迁移模型、多 seed 或 200-image smoke。

## 2. Idea Source 与方案比较

- 来源类型：E20 实验信号与训练时域缺口。
- 主证据：
  - `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20-SPARSE/remote_artifacts/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-SPARSE-COMPARISON-R1/comparison.json`
  - `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20-SPARSE/analysis/result-TAUSB-SDH-E2E-V0-S0-E20-SPARSE.md`
- 为什么现在做：E20 已证明数据流和方向性效果，继续调载体或保护损失会把“训练时长效应”与“方法改动效应”混在一起。

| 方案 | 能回答的问题 | 代价/风险 | 决策 |
|---|---|---|---|
| A. 冻结方法，fresh paired E200 | E20 捷径在充分训练后是否持续，非目标是否恢复 | 约 6–7 小时 GPU | **采用** |
| B. 先改共现保护，再做 E20 | 新保护能否改善 E20 tradeoff | 无法判断原方法的 E200 趋势 | 后续独立消融 |
| C. 从 E20 checkpoint 续训到 E200 | 训练曲线延续 | 不是 fresh paired run，恢复状态和随机性难以对齐 | 不采用 |
| D. 重新训练 hiding/P1 后再 E200 | 新 P1 的完整表现 | 同时改变 Stage-I 与 victim 时域，成本高且因果混淆 | 不采用 |

不运行本实验的代价是：当前只能证明 20-epoch 早期捷径，不能排除 victim 在充分训练后重新学回 person 语义，或非目标 collateral 继续恶化。

## 3. 冻结方法与模型

### 3.1 代理模型

- 架构：YOLOv8n，VOC20 检测器。
- checkpoint：`/root/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt`。
- checkpoint SHA256：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`。
- 状态：冻结；仅用于已经完成的 P1/检测特征/TAL/损失优化，不参与 victim 更新。

### 3.2 受害者模型

- C0/M1 均使用 `configs/voc_yolov8n_20cls.yaml` 构造的 20 类 YOLOv8n-style detector。
- 两臂均从随机初始化 fresh training 开始，不加载 surrogate checkpoint，不继承 E20 checkpoint，不 resume。
- seed=`0`；两臂初始模型 tensor hash 必须一致，并记录 init hash。
- 训练参数冻结：epochs=`200`、imgsz=`640`、batch=`36`、workers=`16`、SGD、cosine LR、close_mosaic=`10`、AMP、`lr0=0.01`、`lrf=0.01`、momentum=`0.937`、weight_decay=`0.0005`。

### 3.3 P1 与扰动协议

- frozen SDH state SHA256：`c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168`。
- P1 state SHA256：`2e102026a9356116de38acb1f5056bf5728afcd453e3447b516d4222f4d70b81`。
- target=`person` / class id `14`；support=`person_gt_bbox`；eps=`16/255`。
- 一个固定高语义 secret；sample-adaptive deep hiding；D-LFC、CICR、CGR、NLA 的状态和权重均不改变。
- 不加入 EOT/JND，不更换 secret，不进行新的 carrier/P1 优化。
- 诚实边界：该 P1 的 hiding/mechanism gate 先前未通过；本实验是 full-horizon empirical effectiveness test，不追认机制门禁为通过。

## 4. 数据与配对协议

- VOC2007+2012 train：16,551 张；其中 6,095 张含 person。
- clean VOC validation：4,952 张；验证图像保持 clean。
- C0：16,551 张训练图像全部直接引用原始 JPEG，`poisoned_count=0`。
- M1：仅重新物化 6,095 张含 person 图像为无损 PNG；其余 10,456 张直接引用原始 JPEG。
- 两臂 ordered stems、label-content manifest、类别空间和 clean-val manifest 必须一致。
- E200 重新物化必须复用精确 P1；若远端 E20 manifest 仍存在，应生成 saved-file hash 对照并要求 E200 与 E20 对应 PNG 逐文件一致。若该对照已不可得，必须记录 validation gap，但 P1/state/config/source hashes、6095 count、Linf 与 support 门禁仍不得放宽。
- saved-reload `Linf <= 16/255 + 1/255`，support 外扰动为 0；报告 PSNR、LPIPS、平均 perturbed area 与 support area。
- Ultralytics 在训练前必须真实解析 mixed list/labels 并 probe 至少一个 batch；缺标签、重复 stem、silent drop 或类别越界均 fail closed。

## 5. Canonical 接入与最小实施

本轮不新增科学方法模块，只参数化已经跑通的 sparse paired controller。

| Step | 文件/入口 | 原子改动 | 必须证据 |
|---|---|---|---|
| 1 | `ue_framework/tools/run_tausb_sdh_sparse_e20.py` | 增加显式 `--victim-epochs {20,200}`；默认值 20，已有 E20 行为不变 | E20 regression、CLI/config sink 测试 |
| 2 | `ue_framework/tools/bind_tausb_sdh_e2e_v0.py` | 增加 `--victim-epochs` 并生成 matched E200 C0/M1 configs；保留旧 `--e20-only` 兼容入口，不修改方法字段 | canonical config diff 只允许 arm、epoch、fresh root 差异 |
| 3 | `ue_framework/stages/train_victim.py` | 记录 fresh init tensor hash；维持 `resume=false` | 两臂 init hash 相同，surrogate hash 与 victim init hash 不同 |
| 4 | sparse materializer/list gate | 在数据盘 fresh root 重建 C0/M1 mixed lists 与 M1 PNG | 16551/6095/10456、hash、Linf、round-trip、dataloader gate |
| 5 | E200 controller/wrapper | C0 E200→evaluate→sanity gate→M1 E200→evaluate→compare→shutdown | stage order、时间/空间/关机单测 |
| 6 | evidence pull/analysis | 拉回最小证据并输出20类 AP50/drop/retention | SHA256 manifest、paired comparison、H→E→N |

参数链必须可追踪为：

```text
CLI --victim-epochs 200
→ bound C0/M1 YAML victim.epochs=200
→ RunContext cfg
→ train_victim.py train_args[epochs]
→ Ultralytics model.train(epochs=200)
→ best.pt / latest.pt
→ clean VOC evaluate
→ paired comparison.json
```

回退路径：`victim_epochs=20` 必须保持已验证 E20 行为；旧 E20 artifacts 只读且不得覆盖。

## 6. 数据盘与成本控制

### 6.1 强制存储布局

`REQUIRED_STORAGE_ROOT=/root/autodl-tmp`，所有增长型路径必须位于数据盘：

```text
/root/autodl-tmp/tausb-sdh/
├── checkouts/<execution-commit>/
├── runs/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1-*/
├── binding/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1/
├── control/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1/
├── logs/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1/
├── comparison/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1/
├── cache/
└── tmp/
```

- checkout、dataset、binding、control、logs、comparison、C0/M1 run roots 必须同时通过“位于挂载根下”和“设备号等于数据盘”的门禁。
- `TMPDIR`、`XDG_CACHE_HOME`、`TORCH_HOME`、`YOLO_CONFIG_DIR` 指向上述数据盘 cache/tmp；不修改 `HOME`。
- 启动前记录 `/` 与 `/root/autodl-tmp` 的设备号、free bytes；每个阶段边界重新记录。
- 系统盘启动时可用空间必须至少 4 GiB；任一阶段系统盘新增占用超过 1 GiB，停止后续阶段并关机。
- 数据盘必须满足 `free >= projected_new_bytes + 8 GiB reserve`；空间不足不删除旧证据，直接停止并关机。
- 禁用周期性 bundle；只保留训练必要 checkpoint、最终 best/latest、日志和一次最小证据包。

### 6.2 时间预算

真实 E20 证据：

- C0 train=`1086.47s`，evaluate=`75.20s`；
- M1 train=`1082.51s`，evaluate=`65.19s`；
- M1 materialize=`385.46s`；paired 总 wall 约 46 分钟。

E200 预算按 full-VOC 每 epoch 实测外推，不放大固定 smoke 开销：

- 预计每臂 train+evaluate：约 3.0–3.3 小时；
- 预计 paired 总 wall：约 6–7 小时；
- C0 train+evaluate hard cap：3.5 小时；
- M1 train+evaluate hard cap：3.5 小时；
- M1 materialize hard cap：20 分钟；
- 整体 GPU wall hard cap：9 小时。

任何异常不在 GPU 模式反复修复：Traceback/OOM/NaN/Inf 立即终止；连续 10 分钟无 epoch/status/log/GPU 有效进度则终止；意外问题的诊断累计达到 20 分钟仍无法恢复则关机。

## 7. 远程运行顺序

```text
PRECHECK_BRANCH_COMMIT_HASHES_MOUNTS_SPACE
→ REUSE_FROZEN_P1
→ SPARSE_MATERIALIZE_C0_M1_ON_DATA_DISK
→ MIXED_LIST_LABEL_DATALOADER_AND_INIT_HASH_GATE
→ C0_FRESH_E200
→ C0_CLEAN_VOC20_EVALUATE
→ C0_SANITY_AND_COST_GATE
→ M1_FRESH_E200
→ M1_CLEAN_VOC20_EVALUATE
→ VOC20_PAIRED_COMPARE
→ MINIMAL_EVIDENCE_MANIFEST
→ AUTO_SHUTDOWN
```

C0 sanity gate：20 类 AP50 全部 finite，person AP50 `>=0.60`，non-target macro AP50 `>=0.50`；未通过则不启动 M1。这是运行完整性门禁，不是方法成功条件。

- ExpID：`TAUSB-SDH-E2E-V0-S0-E200-SPARSE`。
- RunID：`SPARSE-E200-S0-R1`。
- 单一持久 tmux controller；所有成功、失败、超时终态自动关机。
- exact branch/commit、tmux session、host、GPU、environment、artifact roots 与恢复命令在 pre-run review 后冻结。
- 不使用 `--force_resume`；任一臂中断后不得把 partial checkpoint 包装成 fresh E200 结果。是否重新运行需另行审查成本和污染边界。

## 8. Research Contract

### Hypothesis

在 method、P1、数据、victim 初始化协议和训练超参数均冻结时，M1 的 person shortcut 会在 fresh 200-epoch YOLOv8n victim 中持续，使 clean VOC val 的 person AP50 显著低于 matched C0；同时充分训练会使大多数非目标类保持可学，不出现与 person 同量级的下降。

### Success Signal

以下全部满足，记为 `selective_full_horizon_success_single_seed`：

1. `AP50_person(C0)-AP50_person(M1) >= 0.30`；
2. 19 类 non-target macro AP50 drop `<=0.05`；
3. 至少 16/19 个非目标类 AP50 drop `<=0.10`；
4. M1 `poisoned_count=6095`、C0=0，20 类 AP50 finite；
5. saved-reload Linf、support、mixed-list、label、clean-val、fresh-init 和训练协议哈希门禁全部通过；
6. C0/M1 均完成 fresh E200 和独立 clean evaluation。

额外报告但不替代成功判据：person 保持率、non-target 宏保持率、19 类逐类保持率、person-free/cooccur non-target AP，以及相对 E20 的同方向变化。

### Failure Signal

以下任一项成立，记为当前 frozen P1 的 `full_horizon_failure`：

1. person AP50 drop `<0.10`，说明 E20 shortcut 在充分训练后基本消失；
2. non-target macro AP50 drop `>0.15`，说明 collateral damage 过大；
3. 至少 5/19 个非目标类 AP50 drop `>0.20`；
4. C0 sanity gate、fresh init pairing、count/path/label/hash、Linf/support 或 clean-val 任一完整性门禁失败；
5. 任一 victim 未完成 E200，或结果仅来自 resume/partial checkpoint。

未满足 Success 且未触发 Failure，记为 `inconclusive_tradeoff`。不得为了 E200 结果事后改动这些阈值。

### Metric & Split

- 训练：VOC2007+2012 train 16,551；M1 对全部 6,095 张 person images 加噪。
- 验证：clean VOC val 4,952；不加入 robustness transforms。
- Primary：person AP50 drop、19 类 non-target macro AP50 drop。
- Secondary：20 类各自 C0/M1 AP50、drop=`C0-M1`、retention=`M1/C0`；mAP50_all；person-free/cooccur non-target AP。
- Quality：poisoned count、saved-reload Linf max/mean、PSNR、LPIPS、perturbed/support area。
- Provenance：SpecID、ExpID、RunID、branch、commit、seed、surrogate hash、P1 hash、init hash、config/list/label/val hashes。

### Stop Condition

- Spec 未批准、CSV/实现/本地测试/pre-run review 任一未完成：不启动 GPU。
- exact commit、模型/P1/config hash、数据盘挂载、fresh roots、空间投影或 dataloader probe 任一失败：不训练并关机。
- C0 未完成、超过 3.5 小时或 sanity gate 失败：不启动 M1并关机。
- NaN、Inf、CUDA OOM、Traceback、10 分钟无有效进度、20 分钟未解决异常或总 wall 达 9 小时：停止并关机。
- M1/compare 成功、失败或超时后均立即关机；`running_remote` 不算完成。
- Success、Failure 或 Inconclusive 只决定科学分类，不决定证据是否保留；所有终态的可读结果均必须保留、拉回、入账并明确告诉用户，禁止因未达到成功判据而清理结果。操作失败或超时也必须保留已产生的 status、日志、partial metrics 和失败原因。

### Claim Boundary

- 单 seed E200 只能形成 tentative full-horizon empirical evidence，不能声称多 seed 稳定。
- 由于 frozen P1 的 hiding/mechanism gate 未通过，即使 Success Signal 满足，也只说明当前物化数据在该同构 YOLOv8n victim 协议下具有选择性不可学习效果；不能声称机制得到验证。
- surrogate 与 victim 均为 YOLOv8n，属于同构白盒代理设置；不声称跨架构迁移性。
- 不声称 EOT/物理/JPEG/blur 鲁棒性、SOTA、普适类别效果或真实隐私系统安全性。
- 本项目仅涉及公开模型、公开数据集和授权本地/AutoDL 环境，不涉及第三方网络或信息系统攻击。

## 9. Pre-run Review

以下项目在用户批准 Spec 后由执行 CSV 落盘；全部通过前不得启动：

- reviewed branch / exact execution commit；
- E20 regression 与 E200 config/runtime sink；
- surrogate/P1/init/config/list/label/val hashes；
- data-disk mount/device、cache/tmp/output roots 与系统盘增长门禁；
- projected bytes、actual free bytes、8 GiB reserve；
- fresh C0/M1 roots、no-resume、checkpoint policy；
- exact tmux controller、timeouts、shutdown trap、recovery command；
- result：`pending`。

## 10. 最小证据与结果落盘

- Remote controller/status/stage timings：`pending`。
- C0/M1 metrics/status/train summaries：`pending`。
- Paired `comparison.json` 与 `per_class_ap50.csv`：`pending`。
- Quality/provenance/mixed-list manifests：`pending`。
- Pull manifest 与逐文件 SHA256：`pending`。
- Metrics ledger 与 H→E→N：`pending`。
- STATE decision：未经用户批准不替换 Current Best；无论 Success/Failure/Inconclusive 都必须进入实验台账。
