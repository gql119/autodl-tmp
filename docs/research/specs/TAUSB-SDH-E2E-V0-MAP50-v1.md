---
spec_id: TAUSB-SDH-E2E-V0-MAP50-v1
title: SDH end-to-end V0 paired 20-epoch AP50 feasibility pilot
status: approved
experiment_type: end_to_end_feasibility
created: 2026-08-11
approved: 2026-08-11
approval_evidence: user approved obtaining real end-to-end data before further carrier optimization
---

# SDH 端到端 V0：先验证真实 AP50，再优化载体

## 1. 问题与决策

当前程序不是无法运行：`HIDING-S0-R2` 与 `HIDING-S0-SB25-R1` 都正常完成并自动
关机。真正的断点是，正式 `tausb_sdh` 只允许通过全部 hiding/mechanism 门禁的 P1
state 进入 materialization 和 victim，而现有两个 hiding 版本分别失败于：

- r2 / `hf_subband_scale=1.0`：secret retrieval、SSIM、L1 margin、像素多样性、泄漏、
  Linf 与 support 均通过，但高频比例和原 RMS-CV 门禁失败；
- SB25 / `hf_subband_scale=0.25`：高频比例通过，但 retrieval 与 L1 identity margin
  失败。

继续试 `0.50` 能回答频谱容量问题，却仍不能回答最优先的问题：当前完整方法是否在
fresh YOLOv8 victim 上呈现 person AP50 下降且非目标类相对稳定的方向性信号。

用户批准本轮改变实验顺序：先用当前最可工作的 r2 载体取得一组真实 paired AP50，
再决定是否值得继续优化载体。高频比例、RMS-CV 和原 mechanism 数值门禁全部保留为
诊断证据，但不再阻断本轮 V0。正式 200-epoch 协议和正式 `tausb_sdh` 门禁不被削弱。

## 2. 方案比较

| 方案 | 优点 | 缺点 | 决策 |
|---|---|---|---|
| 继续调 `hf_subband_scale=0.50` | 单变量、机制解释干净 | 可能继续陷入 carrier 调参，仍无 AP50 | 不采用 |
| 只用 r2 hiding carrier 直接投毒 | 最快 | 不包含已确定的 D-LFC/CICR/CGR/NLA 更新 | 不采用 |
| r2 → 当前 T0/T1/P0/P1 mechanism → P1 V0 state → paired victim | 包含当前完整方法，直接测真实 AP50 | 机制门禁可能失败，结论只能是 feasibility | **采用** |

不运行本实验的代价是继续依据 surrogate/mechanical 指标争论载体，而没有任何当前
方法的 fresh-victim 数据。

## 3. 冻结方法

### 3.1 载体与支持区域

- 载体来源：hash-verified `HIDING-S0-R2` checkpoint：
  `/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-r2/hiding/hiding_checkpoint.pt`。
- checkpoint SHA-256：
  `a765e27a62bb1a1939aaae487ff6e61ec405f457056d2329c1c49f91e02c9f36`。
- `hf_subband_scale=1.0`；不增加 `0.50`、频谱 loss、RMS loss 或 host-diversity loss。
- 固定单一 primary semantic secret；每个 person GT bbox 是唯一嵌入区域。
- `eps=16/255`；support 外必须严格为 0；EOT 与 JND 关闭。
- r2 的高频/RMS 失败必须进入 metrics/provenance，不得改写成 hiding PASS。

### 3.2 Detector-aware 更新

从 r2 checkpoint 运行现有、未经改义的 8-step matched mechanism：

```text
T0: hiding + easy route
T1: T0 + D-LFC + CICR/floor
P0: T1 + target→non-target CGR
P1: P0 + explicit assigned-logit NLA descent
```

- 仍执行所有原 D-LFC/CICR/CGR/NLA 计算、逐类梯度、SVD 投影和最多五次回溯。
- 原 target/protection gate 按原阈值计算并落盘，但本轮只作 diagnostic。
- 只要 P1 state、梯度、张量、support 和 checkpoint provenance 有限且一致，就生成
  `p1_feasibility_sdh_state.pt`；payload 必须显式记录
  `hiding_gate_passed=false` 及真实 `mechanism_gate_passed`。
- 若 NaN/Inf、OOM、checkpoint/hash 漂移、support/Linf 违规或 P1 state 缺失，仍然
  fail closed，禁止 materialization。

### 3.3 Feasibility materializer

正式 `tausb_sdh` loader 的默认行为不变。只有同时满足以下条件才允许加载未通过
科学门禁的 V0 state：

```text
protocol_id == "TAUSB-SDH-E2E-V0-MAP50-v1"
materialization_mode == "p1_feasibility_state"
allow_failed_scientific_gates == true
exact r2 checkpoint / P1 state / mechanism metrics hashes match
target_class_id == 14
epsilon == 16/255
support_type == bbox
EOT == false and JND == false
```

该路径必须在 manifest 与最终 metrics 中标记
`evidence_scope=end_to_end_feasibility_not_formal_method`。缺少任一绑定时不得静默回退到
`tausb_mask`、SIRC 或 carrier-only。

## 4. 实验臂与阶段

### 4.1 本地/无卡门禁

- Python 3.8 AST/compile、配置解析、checkpoint schema/hash、formal rollback；
- r2 只允许原失败项为 `rms_diversity|delta_high_frequency`，其余 hiding checks 必须通过；
- V0 P1 payload 构建/加载、gate provenance、support/Linf、deterministic rendering；
- C0 `poisoning_ratio=0` 与 M1 `poisoning_ratio=1` 分支；
- 20-epoch 仅在 V0 protocol 允许，formal config 仍强制 200 epochs；
- VOC20 逐类 AP50 比较、缺类/零基线/错误 arm/hash fail closed。

### 4.2 真实数据 integration smoke

- 从完整 VOC train 确定性选择 40 张 person 图像与 160 张 person-free 图像；选择清单
  和 label-content hash 落盘，C0/M1 共用。
- C0 与 M1 各训练 fresh YOLOv8n-style victim 1 epoch；训练集为真实图像，validation
  仍是完整 clean VOC val。
- smoke 只验证：materialization → dataset → victim → clean evaluation → 20-class metrics
  文件全链路，不以 AP 数值判断方法有效性，也不运行 aggregate success gate。
- 任一 arm 不产生 checkpoint、metrics.json 或 finite VOC20 AP50 时停止，不进入 E20。

### 4.3 Paired 20-epoch pilot

| Arm | Train data | Victim | Run tag |
|---|---|---|---|
| C0 | 全部 clean VOC2007+2012；`poisoned_count=0` | fresh YOLOv8n-style，seed 0，20 epochs | `C0` |
| M1 | 6,095 张含 person 图像用冻结 P1 V0 state 渲染；其余图像 clean | 独立 fresh YOLOv8n-style，seed 0，20 epochs | `M1` |

- 两臂 `imgsz=640`、`batch=36`、SGD、初始化结构、优化器、seed、完整 clean val 完全一致。
- 两臂分别使用独立 run root 与 poisoned root，禁止覆盖或串行继承 checkpoint。
- 为避免编码格式成为混杂，两臂均通过同一 generation path materialize；person 图像
  都写为 PNG，非 person 图像采用相同保存协议。
- 输出 person AP50、其他19类逐类 AP50、19类宏平均、逐类 drop/retention、person-free/
  cooccur non-target AP、mAP50_all、poisoned_count、actual Linf、PSNR 与 LPIPS gap。
- 本轮不做 target-carrier-val、学习动力学、JPEG/blur/gray、transfer 或 200 epochs。

## 5. 实现范围

1. 在 `sdh_experiment.py` 增加 exact-spec feasibility 分支：验证并加载 r2，运行原
   mechanism，保存带真实 gate flags 的 P1 V0 state。
2. 在 `sdh_materializer.py` 增加严格协议绑定的 failed-scientific-gate loader；formal
   loader 保持原样。
3. 在 `config.py` 增加 V0 配置校验：只允许 smoke 1 epoch或 pilot 20 epochs；正式
   `tausb_sdh` 仍要求 200 epochs 与全部 gate PASS。
4. 增加 mechanism、smoke-C0、smoke-M1、E20-C0、E20-M1 配置；所有路径使用新根。
5. evaluation/aggregate 接受 V0 provenance，明确记录原 gate 失败，不把 mechanism
   diagnostic FAIL 当作运行错误。
6. 增加 paired comparison 入口，读取两个明确 metrics.json，输出20类 CSV/JSON。
7. 增加最相关单元/集成测试和 pre-run implementation review。

禁止删除或修改旧 r1/r2/SB25 artifacts；禁止提交 checkpoint、数据集或 victim 权重。

## 6. Research Contract

### Hypothesis

即使 r2 carrier 的频谱解释尚不理想，当前固定语义载体经过 D-LFC、CICR、CGR 与 NLA
联合更新后，仍可能在20-epoch fresh victim 中形成可观测的类别选择性方向：相对 matched
C0，M1 的 person AP50 下降，同时19类非目标宏平均没有同量级崩塌。

### Success Signal

integration smoke 必须先完整通过。E20 满足全部以下条件视为方向性成功：

1. `AP50_person(C0)-AP50_person(M1) >= 0.10`；
2. 非目标19类宏平均 AP50 drop `<=0.08`；
3. 至少15/19个非目标类 AP50 drop `<=0.15`；
4. `poisoned_count=6095`、actual `Linf<=16/255+1/255`、20类 AP50 全部 finite；
5. C0/M1 配置、seed、数据 manifest、victim 初始化协议和 clean val hash 匹配。

这是20-epoch、单 seed 的 directional feasibility PASS，不是论文最终成功。

### Failure Signal

以下任一项独立否定当前 V0 的推进价值：

1. person AP50 drop `<0.03`，表示没有可辨别的目标类不可学习方向；
2. 非目标宏平均 drop `>0.15`；
3. 至少5/19个非目标类 drop `>0.20`；
4. M1 `poisoned_count!=6095`、Linf/support/hash 违规、materialization 不确定；
5. C0/M1 任一 arm 未完成20 epochs、评估非 finite、类别映射/clean val 不一致。

介于 Success 与 Failure 之间的结果标记 `inconclusive_tradeoff`，不立即调 carrier；先检查
逐类/共现分组证据。

### Metric & Split

- Dataset：VOC2007+2012，train 16,551 images，含 person 6,095；target id 14。
- Victim：YOLOv8n-style from scratch；seed 0；E20；imgsz 640；batch 36；SGD。
- Primary：clean VOC val 的 person AP50 drop 与19类 non-target macro drop。
- Secondary：20类逐类 AP50/drop/retention、person-free/cooccur、mAP50_all。
- Quality：poisoned_count、actual Linf、PSNR；LPIPS 缺失必须标 validation_gap。
- hiding 高频、RMS 与 mechanism gates：diagnostic only，本轮不得据此改写 AP50 结果。

### Stop Condition

- 本地测试或 pre-run review 未通过，不启动 GPU。
- smoke 未完整打通，不启动 E20。
- NaN/Inf、OOM、Traceback、输入/hash 漂移、输出根已存在、GPU 无有效进程、日志/状态
  长时间不增长时立即停止并自动关机。
- bug 诊断在 GPU 上超过20分钟，立即关机并转无卡修复。
- E20 是合法长训练，不受20分钟总时长限制；每臂运行时长上限和费用预算必须在
  pre-run review 根据一次 smoke 实测后冻结。
- E20 完成后不自动启动200 epochs、载体调优或新 seed。

### Claim Boundary

- smoke 不产生科学结论。
- E20 仅是单 seed、20-epoch、真实 fresh-victim 方向性证据。
- 因使用未通过原科学门禁的 feasibility state，不得声称 hiding/mechanism 已验证、
  shortcut 机制成立、正式 UE 成功、鲁棒性、迁移性或 SOTA。
- 只有 E20 出现方向性信号，才讨论 carrier 优化与后续200-epoch正式实验。

## 7. 运行身份与成本

- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`。
- ExpID：`TAUSB-SDH-E2E-V0-S0-E20`。
- mechanism、smoke C0/M1、E20 C0/M1 使用五个不同 artifact/control/tmux 身份。
- 所有 GPU 命令进入 tmux，带完成/失败/超时自动关机和20分钟 bug guard。
- 运行前记录 exact branch+commit、数据/secret/surrogate/r2 hashes、GPU、环境、磁盘、
  输出根和日志路径。

## 8. Pre-run Review

- reviewed branch/commit：`pending`
- exact mechanism/smoke/E20 commands：`pending`
- r2/P1/data/secret/surrogate hashes：`pending`
- config → mechanism → V0 state → materializer → victim → metrics sink：`pending`
- formal rollback and non-overwrite：`pending`
- smoke-derived E20 cost cap：`pending`
- result：`pending`

## 9. 结果落盘

- Mechanism diagnostic：`pending`
- Smoke artifacts：`pending`
- E20 C0/M1 metrics：`pending`
- VOC20 comparison：`pending`
- H→E→N：`pending`
- STATE decision：`pending`
