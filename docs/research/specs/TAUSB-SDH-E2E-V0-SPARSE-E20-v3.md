---
spec_id: TAUSB-SDH-E2E-V0-SPARSE-E20-v3
title: SDH V0 稀疏物化 paired E20 有效性实验
status: approved
experiment_type: end_to_end_feasibility
parent_specs:
  - TAUSB-SDH-E2E-V0-MAP50-v1
  - TAUSB-SDH-E2E-V0-ONEBOOT-v2
exp_id: TAUSB-SDH-E2E-V0-S0-E20-SPARSE
created: 2026-08-12
approved: 2026-08-12
approval_evidence: user explicitly approved TAUSB-SDH-E2E-V0-SPARSE-E20-v3 Spec
---

# SDH V0 稀疏物化 paired E20 有效性实验

## 1. 问题锚点

- 当前科学问题不变：验证现有固定语义 secret + D-LFC + CICR + CGR + NLA 是否能在
  fresh YOLOv8n victim 上降低 `person` AP50，并保持其余 19 类。
- R4 已证明 P1 state → person-bbox materialization → fresh victim → clean VOC20 AP50
  数据流贯通，但旧费用门禁把 200-image smoke 的固定开销按
  `16551/200 × 20 epochs` 放大，错误给出 59.29 GPU-hours。
- 旧磁盘门禁同时把两个 smoke 根中的 checkpoint、日志和固定文件按数据量线性放大；
  当前 generation 还会把所有 clean JPEG 重编码为 PNG，造成真实但不必要的空间膨胀。
- 旧 `tausb_mask` 的真实记录显示：完整 generate + 200-epoch victim + evaluate 约
  4.88 小时，其中 20-epoch victim 曲线约 15.96 分钟。因此 59.29 小时不是可信成本。
- 本轮问题：在不改变 P1、方法损失、数据 split、victim 超参数或 AP50 判据的前提下，
  能否通过稀疏物化在 2 GPU-hours 硬上限内取得第一组可解释 paired E20 指标？

非目标：不再调 carrier，不重跑 mechanism，不新增 smoke 版本，不做 200 epochs、EOT、
鲁棒性、多 seed 或项目整理；不删除任何旧 artifacts。

## 2. 方案比较与决策

| 方案 | 空间 | 风险 | 决策 |
|---|---:|---|---|
| C0/M1 全量 PNG | 本地抽样推算每臂约 4.89 GiB，仅图像约 9.78 GiB | 无意义复制，剩余空间紧张 | 不采用 |
| 为 clean 样本建硬链接树 | 图像内容不复制 | 跨文件系统可能失败，仍有 16,551 个链接和目录维护 | 备选但不自动回退 |
| 训练路径清单混合原始 JPEG 与 poisoned PNG | 只新增 6,095 张 poisoned PNG，抽样预计约 1.8 GiB | 需显式验证 Ultralytics 路径/标签映射 | **采用** |

不运行本实验的代价是继续只有机制与 smoke 证据，而没有当前方法的有效性数据。

## 3. 冻结方法与稀疏物化协议

### 3.1 方法保持不变

- 复用已核验 P1 feasibility state：
  `c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168`。
- P1 provenance、r2 hiding、secret、D-LFC、CICR、CGR、NLA、8-step mechanism 结果不变。
- target=`person` / class id 14；support=`person_gt_bbox`；`eps=16/255`；no EOT/JND。
- M1 对全部 6,095 张含 person 的 train images 物化；非 person 图像保持完全 clean。
- C0/M1 均从相同结构、seed0、随机初始化的 YOLOv8n-style victim 开始，禁止 checkpoint
  继承或 resume。

### 3.2 C0 数据

- 不创建 clean 图片副本或 hardlink tree。
- `train-c0.txt` 按冻结的 train manifest 顺序列出全部 16,551 张原始 JPEG 绝对路径。
- 训练标签由原始 `images/train` → `labels/train` 对应关系解析。
- C0 manifest 明确记录 `is_poisoned=0`、原始 image/label hash 与 mixed-list hash。

### 3.3 M1 数据

- 仅对 6,095 张含 person 图像运行 frozen P1 bbox renderer，并以无损 PNG 保存到新的
  `poisoned_images/person/`；只复制对应的小型 label 文件。
- 其余 10,456 张非 person 图像不复制、不转码；`train-m1.txt` 直接引用原始 JPEG。
- `train-m1.txt` 保持与 C0 完全相同的 stem 顺序；只有 6,095 个路径替换为 poisoned PNG。
- 每条 manifest 记录 effective image path、clean source path、label path、是否加噪、support、
  secret/P1 hash、saved-file hash、Linf、PSNR 与 perturbed-area。
- 指标必须基于**保存后重新读取**的 PNG 与原始 clean JPEG 解码结果计算，防止量化/保存误差
  被漏掉；要求 `actual_linf <= 16/255 + 1/255`。
- 随机固定抽取至少 64 个 clean JPEG 做 JPEG decode → PNG save → PNG reload 一致性测试，
  像素必须逐值相等，证明格式改变本身不引入额外像素混杂。

### 3.4 训练清单门禁

- 两份清单各 16,551 个唯一 stem；顺序、stem 集合、label hashes 完全相同。
- C0 全部指向原始 JPEG；M1 精确 6,095 个 PNG + 10,456 个原始 JPEG。
- Ultralytics 必须在本地/无卡阶段通过真实路径解析、label 推导和至少一个 batch 的
  dataloader probe；缺标签、重复 stem、错误类别空间或 silent drop 均 fail closed。
- 不允许在运行时静默回退到全量 PNG、hardlink、旧 poisoned root 或 `tausb_mask`。

## 4. 成本与磁盘修订

### 4.1 时间估计

禁止继续使用：

```text
smoke_total_train_stage × (16551/200) × epochs
```

新的 pre-run 估计只使用两类证据：

1. 同一 VOC/YOLOv8n 协议的历史 full-VOC epoch curve；当前证据为 E20≈15.96 分钟/臂；
2. 稀疏物化的 target-image throughput，仅按 6,095 张 target images 外推，不放大固定初始化、
   full-val、checkpoint 或日志开销。

预算冻结为：

- 预期 paired 总时长：45–90 分钟；
- M1 稀疏物化 hard cap：40 分钟；
- C0 train+evaluate hard cap：40 分钟；
- M1 train+evaluate hard cap：40 分钟；
- 整体 GPU wall hard cap：2 小时；
- 任一阶段连续 10 分钟无日志/状态/GPU有效进度，或 bug 诊断累计 20 分钟，立即关机。

### 4.2 磁盘估计

- 本地 300-image 抽样：6,095 张 PNG 均值投影约 1.80 GiB；这只是预估，remote pre-run
  必须用实际 target subset 抽样重新计算。
- 磁盘投影分项计算：`poisoned PNG + copied labels + manifests + checkpoints + final bundle +
  1 GiB contingency`；不得把整个 smoke root 乘数据倍率。
- C0 不建立数据副本；M1 clean images 不占新增图像空间。
- 启动要求：`free_bytes >= projected_new_bytes + 3 GiB reserve`。
- 禁止靠删除旧证据腾空间；空间不足则自动关机并报告具体分项。
- 禁用周期性 bundle；训练中只保留必要 checkpoint，结束后生成一次最小证据 bundle。

## 5. 实施与验证

| Step | 文件/入口 | 原子改动 | 必需证据 |
|---|---|---|---|
| 1 | `stages/generate.py` / 新 sparse helper | 只写 target PNG，生成 C0/M1 mixed lists 与 provenance manifest | 16,551/6,095/10,456 计数、hash、round-trip、Linf 测试 |
| 2 | `stages/train_victim.py` | V0-only 支持显式 train path list；formal/legacy 目录训练路径不变 | dataloader/label probe、feature-off regression |
| 3 | V0 binder/config | 绑定 sparse protocol、fresh roots、E20 参数；不再绑定全量 PNG | config → runtime → list → victim sink 测试 |
| 4 | cost/disk gate | 分离可变计算与固定开销；按实际新增文件分项 | 历史曲线 replay、边界和 insufficient-space tests |
| 5 | controller | 复用 P1，依次 sparse materialize → C0 E20 → M1 E20 → compare → shutdown | no mechanism/smoke rerun、2h hard cap、自动关机测试 |
| 6 | pre-run review | 独立核对 exact commit、commands、paths、counts、预算与 metric sinks | `pass / allow_run` 后才能启动 GPU |

只允许修改上述稀疏数据路径、V0 训练清单与成本控制。不得修改 carrier、D-LFC、CICR、
CGR、NLA、AP50 计算或正式 200-epoch路径。

## 6. 远程运行

```text
PRECHECK
→ REUSE_P1_AND_SPARSE_MATERIALIZE_M1
→ MIXED_LIST_AND_DATALOADER_GATE
→ E20_C0 (fresh)
→ E20_M1 (fresh)
→ VOC20_COMPARE
→ PULL / INGEST / H→E→N
→ SHUTDOWN
```

- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`。
- ExpID：`TAUSB-SDH-E2E-V0-S0-E20-SPARSE`。
- RunID：`SPARSE-E20-S0-R1`。
- 新 control/log/run roots 必须 fresh；R1–R4 证据只读。
- 不再运行 200-image/1-epoch paired smoke；R4 已提供旧链路数据流证据，新 mixed-list 路径由
  无卡 dataloader gate 和 full-run 启动前 batch gate覆盖。
- 所有 GPU 命令在唯一 tmux controller 中串行运行；成功、失败、超时均自动关机。

## 7. Research Contract

### Hypothesis

相对 matched C0，当前 frozen P1 method 在 seed0、20-epoch fresh YOLOv8n victim 中会使
clean VOC val 的 person AP50 明显下降，同时其余 19 类宏平均不会出现同量级下降。稀疏
物化仅消除冗余复制，不应改变 M1 的有效训练像素、标签或方法语义。

### Success Signal

以下全部满足记为 single-seed directional feasibility PASS：

1. `AP50_person(C0)-AP50_person(M1) >= 0.10`；
2. 19 类 non-target macro AP50 drop `<=0.08`；
3. 至少 15/19 个非目标类 AP50 drop `<=0.15`；
4. M1 `poisoned_count=6095`，C0=0，20 类 AP50 finite；
5. saved-reload `Linf<=16/255+1/255`，support 外扰动为 0；
6. 两臂 stem/label/seed/init/optimizer/clean-val hashes 匹配。

### Failure Signal

以下任一项独立否定当前 V0 的推进价值：

1. person AP50 drop `<0.03`；
2. non-target macro AP50 drop `>0.15`；
3. 至少 5/19 个非目标类 drop `>0.20`；
4. poisoned count、Linf、support、path/label/hash 或 saved-reload 任一违规；
5. 任一 victim 未完成 E20，或 C0 仍处在全零/不可解释区。

介于 Success 与 Failure 之间为 `inconclusive_tradeoff`。

### Metric & Split

- VOC2007+2012 train 16,551；person images 6,095；clean VOC val 4,952。
- YOLOv8n-style fresh victim；seed0；E20；imgsz640；batch36；SGD。
- Primary：person AP50 drop、19类 non-target macro AP50 drop。
- Secondary：19类逐类 AP50/drop/retention、mAP50_all、person-free/cooccur NT AP。
- Quality：poisoned count、saved-reload Linf、PSNR、perturbed area；LPIPS 缺失标 gap。

### Stop Condition

- 本地测试或 pre-run review 不通过，不启动 GPU。
- path-list/label/hash/count/round-trip/disk 任一失败，停止并关机。
- NaN/Inf/OOM/Traceback、连续10分钟无进度、bug诊断20分钟、总 wall 2小时，停止并关机。
- C0 未完成或 AP50 不可解释，不启动 M1；M1/compare 完成或失败后立即关机。

### Claim Boundary

- 结果仅是 single-seed、20-epoch、failed-scientific-gate P1 的方向性证据。
- 不声称 hiding/mechanism 已验证、正式 UE 成功、鲁棒性、迁移性、SOTA 或多 seed 稳定。
- 稀疏物化必须证明有效训练像素/标签语义与原协议等价，才能与 C0 比较。

## 8. Pre-run Review

- reviewed branch/commit：`pending`
- exact controller command：`pending`
- sparse manifest / mixed-list hashes：`pending`
- saved-reload equivalence and Linf：`pending`
- config → sparse path list → dataloader → victim → metrics sink：`pending`
- projected disk / actual free / 2h cap：`pending`
- result：`pending`

## 9. 结果落盘

- Sparse materialization evidence：`pending`
- C0/M1 E20 metrics：`pending`
- VOC20 comparison：`pending`
- Metrics summary / ledger：`pending`
- H→E→N：`pending`
- STATE decision：用户批准前不修改 Current Best。
