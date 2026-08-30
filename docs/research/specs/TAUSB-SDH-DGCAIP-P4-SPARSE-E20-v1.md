---
spec_id: TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1
title: 严格确定性 DG-CAIP P4 候选状态与稀疏成对 E20
status: approved
experiment_type: mechanism_to_effectiveness
parent_specs:
  - TAUSB-SDH-DGCAIP-CGR-E20-v2
  - TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-R2-v1
  - TAUSB-SDH-E2E-V0-SPARSE-E20-v3
created: 2026-08-30
approved: 2026-08-30
---

# 严格确定性 DG-CAIP P4 候选状态与稀疏成对 E20

## 1. 本轮目标

本轮只回答一个问题：修复确定性 resize 后，完整 DG-CAIP P4 载体在 fresh
YOLOv8n victim 上，是否能在保持 person 不可学效果的同时降低共现非目标类损伤。

不再以短程机制阈值替代 AP50，也不再增加新方法模块、EOT、JND、鲁棒性增强、
更多 secret、更多 seed 或 E200。所有终态均保留数据并自动关机。

## 2. 冻结的方法与数据

### 2.1 数据、模型与扰动范围

- 数据：VOC2007+2012，训练集 16,551 张，含 person 图像 6,095 张；验证集保持干净。
- 目标类：`person`，VOC/YOLO class id `14`。
- surrogate：冻结的 VOC20 YOLOv8n checkpoint；victim：fresh YOLOv8n。
- secret：冻结单张 `bg-building-sky-09`，只嵌入 person GT bbox。
- 扰动：`epsilon=16/255`，bbox 外扰动必须严格为 0。
- 第一轮继续禁止 EOT、JPEG、blur、gray、JND 与 pseudo-mask fallback。

### 2.2 方法模块

1. host-conditioned semantic hiding carrier：同一 secret 经不同 person 宿主产生
   sample-wise 扰动；不恢复旧 Fourier/ALCE/PAG/late-repair 路径。
2. D-LFC：聚合 person 实例中 secret 所诱导的隐特征。
3. CICR：约束 P3/P4/P5 上 person 特征残差方向一致，并保留能量下限。
4. target composite objective：抑制 person 的检测学习，同时保留载体捷径和残差结构。
5. NLA：在 clean real-TAL positives 上对齐非目标 assigned-class logit。
6. DG-CAIP：按 clean/poison 非目标实例响应 JS 散度定位高风险共现实例，重分配
   固定保护预算，并保护分类、bbox、TAL alignment 与分布。
7. CGR：将目标梯度投影到逐类非目标保护梯度的零空间，再加入固定比例的显式保护
   梯度；最多五次非线性回溯。

## 3. 已完成的确定性前置证据

本轮绑定下列已完成证据，不重新消耗 GPU 复现两步 audit：

- runtime repair commit：`83f43070f04e2a98401ad17ec098c01d83d96665`；
- task-level repair attestation：
  `docs/research/evidence/TAUSB-P1-DET-RESIZE-REPAIR-PASS-83f4307.md`；
- repair attestation SHA256：
  `f05f5f9ca255083d3697af69ad47127c28f8349219e1cf50530edd632bc91b3b`；
- audit config SHA256：
  `0294f29190b60b168afc54ac25e41eb5509a6103ceddf095bc713281a9480900`；
- G0 三次 forward 与 input-gradient bitwise identical；
- G1 strict-fresh A/B bitwise identical；
- G2 2/2 step accepted，adapter 确实变化，状态可加载、finite、support 合法且
  `Linf=0.06274443864822388`。

上述证据只关闭执行确定性 blocker，不作为 P4 或 AP50 有效性证据。

## 4. P4 production mechanism

### 4.1 冻结运行

- strict deterministic backend 必须在 CUDA 初始化前启用；不得 warn-only。
- 固定 seed 0、batch 4、16 calibration batches、24 held-out batches、8 update steps。
- 四臂共享完全相同的初始 adapter、batch 顺序、D-LFC/CICR bank、target/NLA/DG-CAIP
  calibration：`P1-R / P2-CAIP / P3-DIST / P4-DGCAIP`。
- D0 locator 复用已通过并由 SHA256 绑定的报告；不得在本轮重新筛选样本。
- 历史 P1 指标产生于修复前的非严格数值路径，只作参考，不再作为 bitwise/numeric
  阻断项；本轮必须记录该 claim-boundary 修订。

### 4.2 两层门禁

必须分别输出两个结论，禁止合并：

**A. `state_integrity_gate`（决定能否进入 E20）**

全部满足才允许保存和物化 P4：

1. exact commit/config/input/D0/source-P1/repair-report hashes 全部匹配；
2. strict deterministic backend 已实际启用，无 unsupported operator；
3. 四臂初始 adapter hash 和 batch 序列一致；
4. P4 所有 state tensor、loss、gradient、SVD、候选与 held-out 指标 finite；
5. P4 至少 1/8 step accepted，final adapter hash 不同于 initial；
6. P4 扰动 `Linf <= 16/255 + 1e-6`，person bbox 外严格为 0，state round-trip 可加载；
7. frozen YOLO、hiding trunk、reveal decoder、D-LFC/CICR bank 与所有冻结权重 hash 不变；
8. CGR `max_projected_row_dot <= 1e-5` 且每个有效更新的 null dimension `>0`。

任一失败：不物化数据、不训练 victim，保留证据并自动关机。

**B. `mechanism_scientific_gate`（只形成机制结论）**

继续按父 Spec 报告 Q4 三项改善、P4-vs-P3 排序增益、Q1 non-worse、target
retention、CICR、pattern、保护预算和 backtrack/skip。该门禁无论 PASS、FAIL 或
INCONCLUSIVE，都不得覆盖 A 门禁；A 通过后继续 E20。

### 4.3 候选状态身份

- raw state：`p4_dgcaip_state.pt`，`arm_id=P4-DGCAIP`；
- materialization state：独立 P4 协议，不冒充旧 `P1`/formal state；
- 固定标签：`evidence_scope=diagnostic_candidate_ap50_evaluation`；
- 显式记录 `mechanism_scientific_gate_passed`、`state_integrity_gate_passed=true`、
  exact raw-state/metrics/config/D0/P1/repair hashes；
- materializer 必须显式要求 `source_arm_id=P4-DGCAIP`，旧 E2E V0 P1 路径保持不变。

## 5. 稀疏成对 E20

### 5.1 数据物化

- C0：16,551 条训练清单全部直接指向原始 JPEG，不复制图像。
- M1：只将 6,095 张含 person 的图像以无损 PNG 物化；10,456 张 person-free 图像
  直接指向原始 JPEG。
- C0/M1 stem 顺序、标签、类别范围和总数必须一致；M1 精确 6,095 poisoned images。
- PNG 保存后重读，重新核验 `Linf`、bbox 外零扰动、hash 与解码有效性。
- 数据集、checkout、cache、tmp、runs、control、logs 与 comparison 全部位于 AutoDL
  数据盘；系统盘阶段增长超过 1 GiB 即停止。

### 5.2 victim 协议

- C0/M1 均为 fresh YOLOv8n，seed 0，20 epochs，imgsz 640，batch 36，SGD。
- 禁止 resume，禁止用 surrogate checkpoint 初始化 victim。
- 两臂 `victim_init_tensor_sha256` 必须完全一致；否则结果无效并停止比较。
- C0 必须产生可解释的非零 VOC20 AP50，才允许训练 M1。
- 验证集始终使用未加噪 VOC val。

### 5.3 必须报告

- person C0/M1 AP50、drop、retention；
- 19 个非目标类各自 C0/M1 AP50、drop、retention；
- 非目标宏平均、person-cooccur 与 person-free 非目标指标；
- 6,095/count/support/Linf/PSNR、state 与 fresh-init provenance；
- 机制门禁即使 FAIL，也与 E20 结果一起保留并明确区分。

## 6. 结果判据

结果首先按连续指标报告，不因未达标而丢弃。为了与父 Spec 可比较，保留下列
预注册分组：

- **Success**：person drop `>=0.55`；non-target macro drop `<=0.07`；
  person-cooccur non-target drop `<=0.09`；至少 16/19 类 drop `<=0.10`，无新类别
  drop `>0.15`。
- **Failure**：person drop `<0.40`，或 cooccur drop 相对历史 P1 E20 `0.132122`
  改善 `<0.015`，或至少 5 个非目标类相对历史 P1 额外下降 `>0.05`。
- 其他为 `inconclusive_divergence_protection_tradeoff`。

单 seed E20 只能形成 tentative evidence，不宣称 E200、多 seed、跨架构或鲁棒性成立。

## 7. 成本、停止与关机

- 单次 GPU boot 顺序固定：precheck → full P4 mechanism → P4 integrity/binding →
  sparse materialization → C0 E20 → C0 gate → M1 E20 → compare → evidence manifest。
- 总 GPU hard cap：`2 小时`；预期 `55–100 分钟`。
- mechanism hard cap：`20 分钟`；M1 materialization hard cap：`40 分钟`；每个 victim
  train+eval 使用全局剩余预算，任一阶段不得绕过总上限。
- 10 分钟无有效进度、NaN/Inf/OOM/Traceback、磁盘门禁、hash/count/support/fresh-init
  错误均立即停止。
- GPU 上不修代码；如果 bug 诊断累计达到 20 分钟，立即保存证据并关机。
- success、failure、inconclusive、timeout、signal、fatal 任一终态均请求自动关机。

## 8. 开机前门禁

下列全部完成后才通知用户开启 GPU：

1. 用户明确批准本 Spec，特别是“两层门禁、A 通过即继续 AP50”的修订；
2. 新 P4 state/binder/materializer/controller 只做显式、向后兼容的最小扩展；
3. focused tests、旧 P1/E20 regression、Python 3.8 AST/compile 和 CLI 静态检查通过；
4. exact branch/commit 普通 push，远端能够 exact checkout；
5. pre-run review 冻结 exact command、数据盘路径、输入 hashes、fresh roots、成本与
   shutdown trap，结论为 `pass_allow_run`。

在以上门禁完成前，不需要用户开启 GPU。
