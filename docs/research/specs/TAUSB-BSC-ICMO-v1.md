---
spec_id: TAUSB-BSC-ICMO-v1
title: 实例坐标规范化背景频谱载体与匹配系数优化机制验证
status: draft
experiment_type: probe
csv: issues/TAUSB-BSC-ICMO-v1.csv
created: 2026-07-30
parent_spec: TAUSB-BSC-RC-GR-v1
---

# 实例坐标规范化背景频谱载体与匹配系数优化机制验证

## 1. 问题锚点

- STATE 关联：本实验是目标检测类别选择性不可学习样本的新机制 probe，
  不替代当前 TAUSB seed0 tentative best，也不更新 Current Best。
- 触发证据：
  - `TAUSB-BSC-RC-GR-v1` 的真实 VOC Phase A 在 seed 0 上按冻结门禁停止；
  - artifact：
    `ue_project/runs_research_local/TAUSB-BSC-RC-GR-v1-local-phaseA-20260729/phase_a_metrics.json`；
  - review：
    `issues/TAUSB-BSC-RC-GR-v1.review.md`；
  - evidence commit：
    `72fbe3713ba4811dc9c82af3d4c26c1748db5dcf`。
- 已知结果：
  - C0 held-out CICR median 为 `0.575505`；
  - C2-LM 为 `0.523650`，相对 C0 为 `-0.051855`；
  - C2-LM non-target/target residual-energy ratio 为 `0.354418`，
    优于 C0 的 `0.404013`；
  - C2-LM 的 paired median delta bootstrap 95% CI 为
    `[-0.069847,-0.027142]`。
- 新发现的实现错位：
  1. 当前背景 pattern 固定在整图坐标，再乘 person support。不同位置与尺度的
     person 实际看到的是全图 pattern 的不同裁剪，而不是相同的实例相对 pattern。
  2. 当前 C0 与背景 basis 使用不对称的系数到 pattern 映射。背景分支使用
     `base/base.abs().amax()` 的逐步归一化，几乎消除 coefficient 的幅度自由度；
     C0 则保留连续幅度。
  3. 当前 CICR 对一张图中所有 person 正样本做图像级聚合，没有按真实 TAL
     `target_gt_idx` 分离 person 实例。
- 本轮可测问题：

  > 在修复上述三个错位后，同一组经过匹配优化的 phase-scrambled natural
  > low+mid basis，能否比匹配的 synthetic Fourier basis 在未参与优化的
  > person 实例上形成更一致的 classification residual，同时保持较低的
  > non-target 与 box residual leakage？

- 非目标：
  - 不回调 `TAUSB-BSC-RC-GR-v1` 的 Phase A 判据；
  - 不强行绕过原 Phase A 进入原 Phase B；
  - 本轮不测试 R− TAL evasion；
  - 本轮不启用 non-target gradient projection；
  - 不生成 poisoned dataset；
  - 不训练 fresh victim；
  - 不运行 clean VOC mAP 或 robustness evaluation；
  - 不声称 true instance mask。

## 2. Idea Source

### 2.1 来源类型

- 用户机制假设：通过反向传播调整稳定、简单的背景低/中频组合，使同一
  universal pattern 在不同 person 实例上产生相似特征残差。
- 实验信号：C2-LM 没有 frozen CICR 优势，但降低了 non-target residual
  leakage，说明该载体可能适合作为受优化的低泄漏搜索空间。
- 代码审计：现有 renderer 和 coefficient parameterization 没有忠实实现
  “每个实例看到同一个 δ”与“matched coefficient optimization”。

### 2.2 新增机械诊断

使用 frozen shared split 中的 96 张 held-out 图像、184 个 person 实例，
对当前 C2-LM 全图 pattern 做真实 bbox 裁剪并统一缩放：

- instance-crop pairwise cosine median：`0.009146`；
- pairwise cosine Q25：`-0.061825`；
- 与共同 centroid 的 cosine median：`0.152538`；
- centroid cosine Q25：`0.018734`；
- small / medium / large centroid cosine median：
  `0.099737 / 0.070726 / 0.272649`。

这证明当前 global-coordinate renderer 没有让不同实例接收稳定的相对 pattern。

在同一 coefficient direction 下审计 coefficient scale：

| scale | C0 materialized abs-max | C2-LM materialized abs-max | C0 RMS | C2-LM RMS |
|---:|---:|---:|---:|---:|
| `0.0001` | 0.000658 | 0.312920 | 0.000166 | 0.074794 |
| `0.001` | 0.006576 | 0.996924 | 0.001663 | 0.539874 |
| `0.01` | 0.065670 | 0.999329 | 0.016620 | 0.645659 |
| `0.25` | 0.925401 | 0.999329 | 0.360143 | 0.645789 |

因此，直接运行现有 Phase B 会把 basis、幅度、饱和程度和优化条件混在一起，
不能回答用户提出的因果问题。

### 2.3 文献边界

- Angelic Patches 在目标检测中对每个 object instance 同时施加 patch，并报告
  affine/deformable transformation 下的效果，支持“实例对齐渲染是需要控制的
  变量”，但它研究的是检测增强，不证明不可学习样本有效：
  https://openaccess.thecvf.com/content/CVPR2023/html/Si_Angelic_Patches_for_Improving_Third-Party_Object_Detector_Performance_CVPR_2023_paper.html
- EOT 说明若希望一个 pattern 穿过一组几何/成像变化，应把变化分布纳入优化；
  本轮只使用确定性的轻量 object-relative transform audit，不扩展到物理攻击：
  https://proceedings.mlr.press/v80/athalye18b.html
- Semantic-Aware Multi-Label Adversarial Attacks 支持把非目标保持写成约束或投影，
  而不是与目标损失直接加权相加；本轮只记录相关梯度，不启用投影：
  https://openaccess.thecvf.com/content/CVPR2024/html/Mahmood_Semantic-Aware_Multi-Label_Adversarial_Attacks_CVPR_2024_paper.html
- The Translucent Patch 同时评价目标类与非目标类，但属于 test-time detector
  attack，不能证明 training-time shortcut 或 victim unlearnability：
  https://openaccess.thecvf.com/content/CVPR2021/html/Zolfi_The_Translucent_Patch_A_Physical_and_Universal_Attack_on_Object_CVPR_2021_paper.html

## 3. 候选方案比较

| 方案 | 改动 | 能回答的问题 | 主要风险 | 决策 |
|---|---|---|---|---|
| O1：直接绕过旧 Phase A 跑现有 Phase B | 最小 | 现有代码能否降低 route loss | renderer 与幅度严重不匹配，结果不可解释 | 拒绝 |
| O2：实例坐标规范化 + 对称参数化 + 2×2 matched optimization | 中等、独立 probe | 分离 coordinate gain 与 natural-basis gain | 仍只是 surrogate mechanism | 采用 |
| O3：训练小型 pattern generator / context-conditioned generator | 最大 | 更强表达能力是否有效 | 同时改变容量、条件输入和优化，易过拟合 | 暂缓 |

不运行 O2 的代价：无法区分旧失败是“自然频谱假设错误”，还是实现根本没有让
不同 person 看到同一 pattern。直接进入 victim 会浪费 GPU 且无法形成可信机制解释。

## 4. 核心机制

### 4.1 对称 coefficient parameterization

对 C0 和 C2-LM 都在 canonical resolution `Hc=Wc=640` 显式构造
`K=16` 个 zero-mean、unit-L2 的空间 basis：

```text
B_f in R^(K x Hc x Wc), f in {C0, C2-LM}
```

- C0：把当前 16 个 Fourier coordinates 显式物化为 real-valued basis 后做
  zero-mean、unit-L2 与 sign canonicalization；
- C2-LM：复用已冻结的 phase-scrambled natural low+mid basis；
- 两组 basis 使用同一个 coefficient tensor shape、同一个初始化、同一个
  `gamma`、同一个 optimizer 和同一个映射：

```text
q_f,c = sum_k tanh(z_k,c) * B_f,k
P_f,c = eps * tanh(gamma * q_f,c)
```

- 禁止使用依赖当前 `z` 的 `absmax`、RMS 或 L2 归一化；
- `gamma` 在模型运行前由 basis-only mechanical calibration 冻结：
  用 seed `2032` 生成 256 个两组 basis 共用的 Gaussian coefficient direction，
  每个 direction 归一到 max-abs `0.25`；在 C0 与 C2-LM 的 512 个 canonical
  pattern 上共同用确定性 bisection 选择唯一共享 `gamma`，使 pooled median
  pre-JND RMS 为 `0.35*eps`；不能为不同 arm 分别选择 `gamma`，也不能读取
  VOC feature、held-out metric 或 victim 结果；
- `eps=16/255`；
- canonical pre-JND RMS 目标区间冻结为
  `[0.30*eps, 0.40*eps]`；
- 优化使用同一个 amplitude penalty，不能为不同 arm 单独调权重；
- `P_f` 已经包含 `eps*tanh(...)`，renderer 后不得再调用第二个 tanh；
  只允许 JND、support 和最终 epsilon clamp；
- 记录 active-pixel RMS、Linf、PSNR proxy、饱和像素比例和 coefficient
  saturation。
- coefficient saturation ratio 定义为
  `mean(abs(tanh(z))>=0.95)`；
- pattern saturation ratio 定义为 active support 内
  `mean(abs(delta)>=0.95*eps)`；
- active basis 定义为
  `norm(tanh(z_k,:))_2>1e-3`，top-1 basis share 使用各 basis coefficient
  L2 energy 计算。

### 4.2 实例坐标规范化 renderer

对每个 target person annotation box `b_j`：

1. 从同一个 canonical pattern `P_f` 出发；
2. 用 differentiable bilinear resize 将 `P_f` warp 到 `b_j`；
3. 使用由该 annotation box 构造的 forced pseudo ellipse 作为实例 support；
4. 多个 person support 重叠时按有效 support 权重取平均，禁止求和放大；
5. 再应用与现有 TAUSB 一致的 JND 和 `[-eps,+eps]` clamp。

公式：

```text
delta(x) =
  clamp_eps(
    JND(x) *
    sum_j M_j(x) * Warp(P_f, b_j)(x)
    / max(sum_j M_j(x), 1)
  )
```

其中 `M_j` 是 forced pseudo fallback support，不是真实 instance mask。
annotation box 仅用于坐标变换；不得把该实现描述成 segmentation-mask method。

保留 `render_mode=global` 作为因果 control：

```text
global:   P_f 固定在整图坐标后乘 support
instance: P_f 分别 warp 到每个 person 实例坐标
```

### 4.3 Instance-CICR

当前 image-level CICR 改为 instance-aware CICR：

- 从 clean real TAL assignment 读取 `target_gt_idx`；
- 对每个 person GT、每个 P3/P4/P5 scale 分别聚合其被选 PAG positives；
- 每个实例产生一个 classification residual：

```text
r_i,j,s =
  masked_mean(
    F_adv_s - stopgrad(F_clean_s),
    PAG_s and assigned_gt == j
  )
```

- prototype 仍为 train-only stop-gradient EMA，按 scale 分离；
- held-out 时 prototype 完全冻结；
- 低能量 residual 不得静默丢弃：必须同时报告 valid-instance coverage、
  low-energy ratio 和 zero-norm ratio；
- primary statistical unit 为 image：一张图内多个 instance 先取 median，
  防止多实例图造成 pseudo-replication；
- instance-level 分布只作为 secondary evidence。

### 4.4 Matched optimization

所有 arm：

- surrogate frozen；
- 仅 `z in R^(16x3)` 可训练；
- target route 固定为 `easy_cls`；
- Adam，learning rate `0.01`；
- seed `0`；
- 四个 arm 使用 seed `2033` 生成的同一个 Gaussian `z0`，并归一到
  max-abs `0.25`；
- batch size `4`；
- calibration batch order 完全相同；
- 4 个 route-only warmup steps；
- 40 个 `Instance-CICR + easy_cls + amplitude` steps；
- prototype 只由 calibration 更新；
- held-out 不参与任何 loss、prototype 或 early stopping；
- 不启用 non-target loss、gradient projection 或 backtracking。
- 优化过程不使用随机 EOT，避免把 renderer 与 transform robustness 混为一项；
  只在 held-out 上额外运行以下固定 audit：
  - centered scale `0.90`、`1.10`；
  - object-relative x translation `-0.05w`、`+0.05w`；
  - object-relative y translation `-0.05h`、`+0.05h`。

目标：

```text
L_matched =
  1.0 * L_instance_cicr
  + 1.0 * L_easy_cls_and_box_teacher
  + 1.0 * L_canonical_rms
```

其中：

```text
L_canonical_rms =
  (RMS(P_f) / (0.35*eps) - 1)^2
```

该归一化与权重在运行前冻结，不得根据四个 arm 的 held-out 结果分别调整。

## 5. 2×2 因果实验

| Arm | Basis | Renderer | Optimization | 作用 |
|---|---|---|---|---|
| G-C0 | matched synthetic Fourier | global | matched | synthetic/global control |
| G-C2LM | phase-scrambled natural low+mid | global | matched | 复测 global natural basis |
| I-C0 | matched synthetic Fourier | instance-canonical | matched | 估计 coordinate gain |
| I-C2LM | phase-scrambled natural low+mid | instance-canonical | matched | 完整候选 |

主要 contrasts：

1. `I-C2LM - G-C2LM`：实例坐标规范化收益；
2. `I-C2LM - I-C0`：在相同实例 renderer 下的 natural-basis 收益；
3. `I-C0 - G-C0`：synthetic basis 的纯 coordinate gain；
4. interaction：

```text
(I-C2LM - G-C2LM) - (I-C0 - G-C0)
```

解释矩阵：

- contrasts 1 与 2 都 PASS：支持“实例对齐的自然低/中频 carrier”；
- 仅 1 和 3 PASS：只支持 instance-coordinate shortcut，不支持自然背景 basis；
- 仅 2 PASS：支持 natural basis，但实例规范化不是主要因素；
- 均不 PASS：停止该路线，不进入 gradient routing 或 victim。

## 6. 输入与冻结协议

- Dataset：Pascal VOC train；
- target：person，class id `14`；
- source manifest：
  `research_workspace/sources/bsc_background_manifest.json`；
- source manifest hash：
  `3a13b0f38b06006fd7f68ae03c7206b4b047d4b6129ee7357b05b966641d47af`；
- shared split：
  `research_workspace/sources/TAUSB-ALCE-CTX-AUDIT-v1.json`；
- split hash：
  `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1`；
- label hash：
  `0c8b6f6424061bc31b84ddf42b7370dcbd074f26805433d0ba275c24815e3248`；
- calibration：32 person-only + 32 person-cooccur；
- held-out：32 person-only + 64 person-cooccur；
- surrogate：
  `voc20_surrogate.pt`；
- checkpoint hash：
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`；
- C2-LM basis hash：
  `0395c41541d6bcb51ce81805a96271cd099253eee20c94945968d6e1b0f881c1`；
- C0 Fourier coordinate pack：
  `ue_project/runs/artifacts/tausb_mask/steps40/seed0/noise/global_params.pt`；
- C0 coordinate pack hash：
  `6e4bf48f3193c3394d19e7a971b3262c5f8e05e013b9f7456180cbc1617e4d46`；
- seed：0；
- image size：640；
- clean TAL/PAG gate 与当前 probe 一致。

## 7. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | 新 `instance_canonical_carrier.py` | 显式 C0 basis、对称 parameterization、global/instance renderer、overlap mean | translation/scale reconstruction、support containment、overlap bounded、finite gradient |
| 2 | `bsc_rc_gr_probe.py` 或新独立 probe module | `ProbeBatch` 增加 target instance boxes/support；不改原 Phase A artifacts | old probe tests 不变；实例数量与 label 对齐 |
| 3 | 新 `instance_cicr.py` | 按 `target_gt_idx` 聚合 per-instance residual | two-instance separation、zero assignment、multi-scale、held-out freeze |
| 4 | 新 matched probe/config | 4 arms、共同初始化/optimizer/batch order、完整 diagnostics | config → runtime → loss/metric sink 可追踪 |
| 5 | tests | 参数化 scale sweep、C0/C2LM RMS matching、saturation、feature-off | 全部 finite；RMS ratio 与阈值一致 |
| 6 | local cheap smoke | 真实 surrogate + 1 个 person-only + 1 个 cooccur batch | 仅 mechanical PASS，不作为机制结果 |

### 7.1 必须通过的 mechanical gates

正式 mechanism run 前：

1. canonical pattern 分别 warp 到不同平移/尺度 bbox，再逆向 resize 回 canonical
   坐标，pairwise NCC median `>=0.98`、Q25 `>=0.95`；
2. global control 必须复现低 instance-crop similarity，不允许两个 render mode
   实际走同一路径；
3. 同一初始化下 C0/C2-LM canonical pre-JND RMS ratio 位于 `[0.98,1.02]`；
4. active-pixel RMS ratio 位于 `[0.95,1.05]`；
5. coefficient scale 从 `1e-4` 到 `0.5` 时映射连续、finite，不出现一个 arm
   已饱和而另一个接近零的旧行为；
6. 重叠实例 perturbation 不超过 `eps`，且非 support 区域严格为零；
7. forced pseudo fallback support 的面积与原 support 统计一致；不引入 bbox
   全覆盖；
8. 原 `tausb_mask` 与 `TAUSB-BSC-RC-GR-v1` 入口数值不变。

## 8. Research Contract

### 8.1 Hypothesis

在对称 coefficient parameterization 下，把同一个 phase-scrambled natural
low+mid canonical pattern 分别 warp 到每个 person 的实例坐标，并用
instance-aware CICR 优化，能够比：

1. 同 basis 的 global-coordinate renderer；以及
2. 同 renderer 的 matched synthetic Fourier basis

在 disjoint held-out person 图像上产生更一致的 target classification
residual。该收益不能依赖更大的有效扰动、原背景 layout、非目标 residual
泄漏或 box tower 破坏。

### 8.2 Success Signal

完整假设 PASS 必须同时满足：

1. I-C2LM held-out image-level CICR median `>=0.60`，Q25 `>=0.20`；
2. coordinate contrast：
   `CICR(I-C2LM)-CICR(G-C2LM) >=0.08`；
3. natural-basis contrast：
   `CICR(I-C2LM)-CICR(I-C0) >=0.05`；
4. `I-C2LM - G-C2LM` 与 `I-C2LM - I-C0` 使用 seed `2040`、
   10,000 次、按 person-only/person-cooccur 分层的 paired image-id
   bootstrap；每个 contrast 的有效 paired image `>=80`，95% CI 下界均
   `>0`；
5. valid-image coverage `>=0.90`，target residual zero/low-energy ratio
   `<=0.20`；
6. I-C2LM 的 held-out easy classification + box-teacher loss 相对其初始值
   下降 `>=10%`；
7. I-C2LM non-target/target residual-energy ratio
   `<=1.05 * ratio(I-C0)` 且绝对值 `<=0.40`；
8. I-C2LM box residual energy `<=1.05 * box(I-C0)`；
9. person-only/person-cooccur CICR median gap `<=0.15`；
10. small/medium/large CICR median 最大差值 `<=0.20`；
11. 每个固定 object-relative affine audit 的 CICR median 均
    `>=0.90 * identity CICR median`；
12. I-C2LM intended low+mid energy `>=0.70`，source max-abs correlation
    `<=0.30`；
13. 四个 arm 的 final active-pixel RMS 最大值/最小值 `<=1.05`，
    Linf 均 `<=16/255`；
14. 每个 arm coefficient saturation ratio `<0.25`，
    active basis fraction `>=0.25`，单一 basis energy share `<0.80`；
15. 所有 core metric、gradient、coefficient、prototype 与 pattern finite，
    input/basis/split/checkpoint/config hash 完整。

### 8.3 Failure Signal

以下任一项成立即停止后续梯度路由与 victim 推进：

1. instance renderer mechanical NCC median `<0.98` 或 Q25 `<0.95`；
2. C0/C2-LM 初始 RMS ratio 超出 `[0.98,1.02]`，或正式 arm 的
   active-pixel RMS ratio 超出 `[0.95,1.05]`；
3. I-C2LM calibration CICR 相对初始提高 `>=0.10`，但 held-out 提高
   `<0.02` 或 calibration-heldout gap `>0.15`：判为 coefficient overfit；
4. natural-basis contrast 不 PASS，但 `I-C0-G-C0` 与
   `I-C2LM-G-C2LM` 均 `>=0.08` 且各自 paired bootstrap 95% CI 下界
   `>0`：只能支持 coordinate alignment，不能支持背景频谱假设；
5. person-cooccur non-target/target leakage 高于 person-only `1.5x`：
   判为共现 collateral leakage；
6. source max-abs correlation `>0.30`：不能声称减少了场景语义依赖；
7. coefficient saturation ratio `>=0.25`、active basis fraction `<0.25`
   或 top-1 basis share `>=0.80`：判为幅度/单 basis 捷径；
8. route loss 下降但 held-out CICR `<0.20`：只支持目标响应优化，不支持
   统一 residual；
9. instance residual coverage `<0.70`：当前 TAL/PAG gate 不足以支撑结论；
10. pattern/gradient 非 finite、basis rank `<8`、hash 变化或 split overlap。

### 8.4 Metric & Split

- Primary：
  - held-out image-level Instance-CICR median/Q25；
  - coordinate contrast；
  - natural-basis contrast；
  - stratified paired bootstrap 95% CI。
- Secondary：
  - instance-level CICR distribution；
  - route effect；
  - valid-instance/image coverage；
  - person-only/cooccur 与 small/medium/large 分组；
  - non-target/target residual-energy ratio；
  - box residual energy；
  - source correlation；
  - coefficient/basis usage；
  - canonical/active perturbation RMS、Linf、PSNR proxy、band energy。
- Split：
  - calibration/held-out 严格复用 frozen shared split；
  - bootstrap unit 为 image id；
  - person-only/cooccur 分层重采样；
  - held-out 不进入 prototype、loss、early stopping 或 hyperparameter
    selection。
- Clean VOC mAP、victim AP、PSNR/LPIPS dataset aggregate：
  本机制 probe 均为 `not_applicable`。

### 8.5 Stop Condition

- mechanical gates 任一失败；
- NaN、Inf、OOM、Traceback；
- 日志无进度且进程/GPU 无有效活动；
- source/basis/split/checkpoint/config hash 变化；
- train/held-out overlap；
- 连续 5 step target residual valid-instance coverage `<0.50`；
- 连续 5 step active-pixel RMS 偏离共同目标 `>20%`；
- coefficient/gradient 非 finite；
- run root 已存在；
- 4 arms 未使用完全相同 batch order、step 数、optimizer 或初始化；
- ICMO mechanism 未 PASS 时，不创建 gradient-routing Spec；
- 未经新的 Spec 与用户批准，不生成 poisoned dataset 或训练 victim。

### 8.6 Claim Boundary

- 本实验只能支持或反驳 surrogate 上的实例规范化 residual mechanism；
- mechanism PASS 不等于类别选择性 UE 有效；
- 单 seed 只能标 `tentative mechanism evidence`；
- annotation bbox 只用于坐标 warp，support 仍是 forced pseudo fallback，
  不得称为 true instance mask；
- 如果只通过 coordinate contrasts，只能声称实例坐标规范化有效；
- 只有 I-C2LM 显著优于 I-C0，才能声称 natural low+mid basis 提供额外收益；
- phase scrambling 只能声称减少可识别 layout，不能证明信息论意义无语义；
- 本轮不支持 gradient routing、non-target AP、victim target collapse、
  cross-model transfer 或物理鲁棒性声明；
- 不得用本轮结果修改旧 Phase A 的冻结阈值。

## 9. Canonical 接入与回退

建议新增独立入口：

```text
ue_framework/tools/probe_tausb_bsc_icmo.py
```

配置：

```text
ue_framework/configs/exp_voc_person_tausb_bsc_icmo_probe.yaml
```

参数链：

```text
CLI/config
-> frozen source/split/checkpoint hashes
-> explicit matched C0/C2LM basis
-> symmetric coefficient parameterization
-> global | instance-canonical renderer
-> forced pseudo per-instance support
-> clean real TAL/PAG + target_gt_idx
-> Instance-CICR + easy_cls + amplitude constraint
-> four matched arms
-> paired/group/quality diagnostics
-> status and gate decision
```

回退：

- 新入口与现有 `launch_one.py`、`tausb_mask`、旧 BSC probe 隔离；
- `render_mode=global` 只作为新 matched control；
- 不修改 current best YAML；
- 删除/关闭新入口时，现有训练和评估数值路径不变。

## 10. 远程运行

- 入口：
  `python -u -m ue_framework.tools.probe_tausb_bsc_icmo`；
- Config：
  `ue_framework/configs/exp_voc_person_tausb_bsc_icmo_probe.yaml`；
- Phase：`matched`；
- Device：single GPU；
- Seed：0；
- Remote project root：
  `/root/autodl-tmp/ue_project`；
- ExpID：
  `TAUSB-BSC-ICMO-MECH-S0`；
- artifact root：
  `/root/autodl-tmp/ue_project/runs_research/TAUSB-BSC-ICMO-v1`；
- 预计耗时：单 GPU 小于 1 小时；以 pre-run review 的实际 smoke 为准；
- run root 已存在即 fail closed；不自动删除、不覆盖、不 resume；
- 正式运行前必须：
  - 本地 mechanical gates 与相关 tests 通过；
  - `pre-run-implementation-review=pass`；
  - 绑定 branch + commit；
  - 远程 source/split/checkpoint/global params hash 与 Spec 一致；
  - 使用独立 tmux session 和日志。

## 11. Pre-run Review

- reviewed branch / commit：pending
- exact command：pending
- parameter sink probe：pending
- renderer reconstruction evidence：pending
- coefficient scale-sweep evidence：pending
- instance residual assignment evidence：pending
- baseline/disable-path evidence：pending
- output non-overwrite check：pending
- result：`pending`

## 12. 结果落盘

- Remote artifacts：
  `research_workspace/experiments/TAUSB-BSC-ICMO-MECH-S0/remote_artifacts/`
- Metrics summary：
  `research_workspace/experiments/TAUSB-BSC-ICMO-MECH-S0/metrics-summary.json`
- H→E→N analysis：
  `research_workspace/experiments/TAUSB-BSC-ICMO-MECH-S0/analysis/`
- Experiment ledger：
  `research_workspace/00-实验记录.md`
- STATE update decision：
  仅在 artifacts、metrics 与 H→E→N 完成后提出；由用户决定。
