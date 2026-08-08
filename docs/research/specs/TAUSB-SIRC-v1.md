---
spec_id: TAUSB-SIRC-v1
title: 分层语义图像载荷与检测残差一致性机制验证
status: approved
experiment_type: probe
csv: issues/TAUSB-SIRC-v1.csv
created: 2026-08-01
approved: 2026-08-08
parent_spec: TAUSB-BSC-ICMO-v1
---

# 分层语义图像载荷与检测残差一致性机制验证

## 1. 问题锚点

- STATE 关联：本实验是新的 surrogate mechanism probe，不替代当前 TAUSB
  seed0 tentative best，也不更新 Current Best。
- 当前依赖：`TAUSB-BSC-ICMO-v1` 已完成实现和本地 mechanical smoke，正式
  四臂结果尚未产生；本 Spec 不修改其冻结协议和代码快照。
- 文献触发：Meng et al., *Semantic Deep Hiding for Robust Unlearnable
  Examples*，本地 PDF：`D:/Googledownload/2406.17349v1.pdf`，SHA256：
  `8a2703b7f9851854636e4ac20c5e8f98b10c278ada795f6c9c080f7454248bea`。
- 用户假设：在 person 区域嵌入跨样本稳定、具有全局语义形状但局部纹理可变的
  载荷，并使其在 YOLOv8 检测特征中产生相似残差，可能比无结构或
  phase-scrambled 的低/中频载体更易被学习且更耐常见变换。
- 本轮可测问题：

  > 在完全匹配 epsilon、support、实例 renderer、频谱幅度、系数容量、优化
  > 步数与 split 后，保留共同语义形状的低/中频载体，能否比其精确
  > phase-scrambled control 在 held-out person 实例上形成更一致、变换后更稳定
  > 的检测特征残差，同时不显著增加 non-target 与 box residual leakage？

- 非目标：
  - 不直接复现或训练完整 HiNet/INN；
  - 不用 Stable Diffusion/ControlNet 在线生成载荷；
  - 不改变当前 TAUSB best；
  - 不生成 poisoned dataset；
  - 不训练 fresh victim；
  - 不运行 clean VOC mAP；
  - 不启用 non-target gradient projection；
  - 不把 forced pseudo fallback 描述成真实 instance mask；
  - 不把 classification 论文结果直接外推到 detection。

## 2. Idea Source 与机制边界

### 2.1 论文实际做了什么

论文包含三个不同机制，必须分开理解：

1. **Deep Hiding**：用 DWT + INN 把 semantic image 隐藏进 clean image，得到
   `x_ue`；同时通过 revealing loss 保留可恢复的隐藏语义。
2. **Latent Feature Concentration (LFC)**：定义像素扰动图
   `x_pm=x_ue-x_c`，用预训练 ResNet-18 最后一层卷积提取 `G(x_pm)`，最小化
   同类扰动图特征之间的余弦距离。
3. **Semantic Images Generation (SIG)**：同类 semantic images 共享文本与
   Canny 全局形状，但局部纹理不同；不同类选择相互远离的文本语义。

因此，论文的 LFC 与我们的 residual consistency 思想相近，但并不相同：

- LFC 对齐的是**像素扰动图经过外部分类器后的特征**；
- ICMO/本 Spec 对齐的是**检测器内部 clean/poison 差分**：
  `F_s(x+delta)-stopgrad(F_s(x))`；
- 前者按分类标签做 class-wise 聚合；后者必须按真实 TAL `target_gt_idx`、
  person 实例和 P3/P4/P5 scale 分组。

### 2.2 为什么不直接照搬完整 INN

- INN 产生 `delta_i=F(x_i,h_i)-x_i`，强烈依赖整张 clean image；在检测场景中
  容易把背景和共现 non-target 的信息编码进扰动，增加 collateral damage。
- 论文为隐蔽性倾向把信息藏入高频子带；“hidden image 有高级语义”不等于
  “实际小扰动主要是稳健低频”。
- 论文自身在 JPEG 和 adversarial training 下仍有明显失效，不能把其
  robustness 表述为对所有 countermeasure 都稳定。
- 完整 INN 同时改变参数容量、样本自适应、频谱、训练数据与恢复目标，若直接
  接入，无法判断改善来自语义载荷还是生成器容量。

### 2.3 可迁移的最小机制

本 Spec 只迁移两点：

1. **共享全局形状 + 少量局部纹理变体**，在 class-wise 单一载荷和
   sample-wise 独立载荷之间形成可控中间态；
2. **残差特征集中**，但改为检测器实例级 residual，并单独增加变换后的
   residual consistency。

另外保留第三个思想作为 **held-out 诊断而非训练损失**：在轻度削弱 person
纹理后，载体是否仍能降低 target route loss。该诊断回答“聚合后的响应是否与
person 预测有关”，但不让 `L_cue` 反向传播，避免第一轮同时改变 carrier、
优化目标和标签语义。

## 3. 候选方案比较

| 方案 | 核心改动 | 可回答的问题 | 风险/成本 | 决策 |
|---|---|---|---|---|
| O1：完整 HiNet/INN instance hiding | 训练 sample-adaptive 隐藏网络与 reveal decoder | 图藏图能否直接用于 detection UE | 容量、频谱、背景依赖和目标同时变化；高成本、不可解释 | 暂缓 |
| O2：单一固定 semantic template | 所有 person 使用同一 phase-preserved pattern | 精确重复语义载荷是否易学 | 可能退化为单一 trigger；无法检验 class/sample 平衡 | 作为对照 |
| O3：分层语义残差载体 SIRC | 共同全局 phase/shape + 4 个 texture variants + shared residual prototype | 语义结构与纹理多样性是否同时保留易学性和稳健性 | 中等；仍是 surrogate mechanism | 采用 |

不运行 O3 的代价：无法区分“自然图像频谱有效”与“保留跨样本共同语义结构
有效”，也无法知道当前 phase scrambling 是否恰好删除了最有价值的结构信息。

## 4. 核心机制

### 4.1 冻结语义锚点与纹理变体

- 语义结构锚点：`bg-tree-08`，SHA256
  `dc8ffacadaa07cae3aa38658099574f81787b8803a56b1962c6ccc68a332f310`。
- 选择理由：单一居中树形具有稳定全局轮廓，不属于 VOC20 类别；它不是
  person，也不是 person 常见共现类。正式运行前仍必须通过 surrogate
  contamination screen。
- 纹理 donor 只使用已经授权且 person-free 的四张图：
  - `bg-waves-01`：
    `02d5976d61ca704bb9cbd547fd6bf9bbecd3baf28cf936716c4b59aad35ee778`；
  - `bg-bubbles-02`：
    `75ff3a5039c8f1d2c5f74262d9259d7373f9a78af5e2f489452b46496b29f374`；
  - `bg-beach-03`：
    `a37692584c6208dd108613a0bd4e08b087e28edf01c20ba5057de34bdf948981`；
  - `bg-field-04`：
    `42cb21ff8b0605a25932871dd63a4b340092c54810bb31e747aaad442f9a092f`。
- 输入统一使用确定性的中心正方形 crop，再 resize 到 `640x640`；不根据 VOC
  feature 或 held-out metric 选择 crop。
- 结构/纹理解耦使用 Fourier phase-amplitude construction：

```text
phi_anchor = angle(FFT(anchor))
A_m        = abs(FFT(texture_donor_m))
V_m        = real(IFFT(A_m * exp(i * phi_anchor)))
```

- 四个 `V_m` 共享 anchor phase，因此保留共同全局结构；donor amplitude 不同，
  因而局部纹理统计不同。
- 只保留半径 `[2,24]` 的 low+mid frequency，禁止读取 VOC held-out 结果调整
  频段。
- 每张训练图使用稳定的 `hash(image_id, seed=2102) mod 4` 选择一个 variant；
  同一图中所有 person 使用同一 variant，避免 person support 重叠时混合不同载荷。

### 4.2 精确 phase-scrambled control

对每个 `V_m` 构造 `PS_m`：

```text
PS_m = real(IFFT(abs(FFT(V_m)) * exp(i * phi_random_m)))
```

- `phi_random_m` 使用 seed `2101`，满足共轭对称；
- `PS_m` 与 `V_m` 逐频点 amplitude spectrum 相同，只破坏 phase/shape；
- 因而 `semantic-preserved - phase-scrambled` contrast 不被频谱能量混淆。

### 4.3 匹配 basis 与参数化

每个 carrier family 用相同的 16 个 radial-orientation masks 把 `[2,24]` 频谱
分解为显式 basis；所有 basis 做 zero-mean、unit-L2 和 sign canonicalization。

实现冻结项：4 个径向区间固定为 `[2,5.5)`、`[5.5,10)`、`[10,16)`、
`[16,24)`，每个区间再按 modulo-pi orientation 均分为 4 个共轭对称子带。
为避免 unit-L2 后丢失原 carrier 的子带能量比例，每个 basis 同时保存一个固定的
有符号重建尺度 `a_mk`，并在每个 variant 内做 L2 归一化。可训练系数只做正值
子带调制：

```text
P_m(z) = eps * tanh(
  gamma * sum_k a_mk * [1 + tanh(z_kc)] * B_mk
)
```

因此 `z` 不能任意把单个子带翻相；semantic 与 phase-scrambled family 使用相同
系数时仍保持逐子带 amplitude matching。该约束不增加参数，仍只有 `16x3=48`
个可训练系数。由冻结源、上述实现和 seed 生成的 640-resolution semantic bank
SHA256 为
`0c503d900676c05212cef3443ab4a11d0dafe17e0ab82d0a384bc870a2b945d2`。

所有 arm 共享：

- `eps=16/255`；
- `K=16`、coefficient shape `16x3`；
- seed `2103` 的同一个 `z0`，max-abs `0.25`；
- 共同 gamma calibration：seed `2104`、256 directions、pooled target RMS
  `0.35*eps`；
- Adam、learning rate `0.01`、batch size `4`；
- 4 个 route-only warmup steps + 40 个 matched optimization steps；
- 同一 calibration batch order、support、JND、clamp 和 amplitude loss；
- 禁止逐 arm gamma、逐步 max/RMS normalization 或 held-out tuning。

### 4.4 Instance-canonical renderer

复用 ICMO 的 instance-canonical renderer：

```text
delta(x) = clamp_eps(
  JND(x) *
  sum_j M_j(x) * Warp(P_variant, b_j)(x)
  / max(sum_j M_j(x), 1)
)
```

- `M_j` 是 forced pseudo ellipse support，不是真实 instance mask；
- 每个 person box 都看到对象相对坐标一致的共同结构；
- 重叠区域取均值；support 外扰动必须精确为 0。

### 4.5 检测残差集中与变换一致性

沿用 clean TAL/PAG assignment，按 person GT 和 FPN scale 计算：

```text
r_i,j,s = masked_mean(
  F_s(x_i + delta_i) - stopgrad(F_s(x_i)),
  PAG_s and target_gt_idx == j
)
```

基础 Instance-CICR：

```text
L_cicr = mean_s,i,j [1 - cos(r_i,j,s, stopgrad(mu_s))]
```

`mu_s` 只由 calibration split 的有效 target residual 更新；held-out 完全冻结。

变换残差集中 TRC：对同一 poisoned image 使用两种确定性采样的可微变换，clean
与 poisoned 分支使用相同几何变换：

```text
L_trc = mean_T,s,i,j [1 - cos(r^T_i,j,s, stopgrad(mu_s))]
```

训练 EOT 只包含：

- object-relative scale `[0.90,1.10]`；
- translation `[-0.05w,+0.05w] x [-0.05h,+0.05h]`；
- Gaussian blur `3x3, sigma in [0.4,0.8]`；
- grayscale probability `0.25`。

JPEG50 只做 held-out audit，不参与反向传播。

### 4.6 目标函数与非目标边界

Phase A 所有 arm 使用完全相同的目标：

```text
L_A = 1.0 * L_instance_cicr
    + 1.0 * L_easy_cls_and_box_teacher
    + 1.0 * L_canonical_rms
```

`shape_ncc` 是 canonical carrier 与 anchor band-limited gradient map 的独立
诊断指标，不进入任何 arm 的优化。另将 canonical pattern 映射到
`P_vis=clamp(0.5+0.5*P/eps,0,1)`，分别把 `P_vis` 与 anchor 输入同一个 frozen
YOLO surrogate backbone，记录 P5 global-pooled cosine 作为
`semantic_proxy_cosine`。该 proxy 也不参与优化，只检验高层特征中是否仍能区分
phase-preserved 与 phase-scrambled carrier。这样主要 contrast 只改变 basis
phase/shape，不被额外 auxiliary loss 混淆。若 semantic structure 在优化中消失，
则按 Failure Signal 停止；是否增加 shape regularizer 必须另建 Spec。

Phase B 使用完全相同的 EOT forward：

```text
I-SV-E0: L_B = L_A + 0.0 * L_trc
I-SV-E1: L_B = L_A + 1.0 * L_trc
```

本 probe 不主动优化 non-target loss 或 gradient projection，只记录：

- target/non-target logit residual RMS ratio；
- non-target FPN residual energy；
- box residual energy；
- person-only / person-cooccur 分组；
- target gradient 与 non-target preservation gradient cosine。

若 SIRC 机制 PASS，non-target orthogonal projection 才进入下一独立 Spec，避免
把 carrier 与梯度路由混成一个实验。

### 4.7 Shortcut sufficiency held-out audit

仅在 held-out 上构造一个确定性的轻度 person 外观弱化视图。对 forced pseudo
target support `M_t` 内的 RGB 像素先转灰度并复制为 3 通道，再做 Gaussian blur
`5x5, sigma=1.0`；support 外保持原图：

```text
A_t(x) = (1-M_t) * x + M_t * Blur5x5_sigma1(Gray3(x))
x_cue_clean  = A_t(x)
x_cue_carrier = clamp(A_t(x) + delta, 0, 1)
```

载体在外观弱化之后叠加，避免 blur 同时破坏待测 cue。TAL/PAG assignment 固定
来自原始 clean `x`，两条 cue 分支使用完全相同的 target units，且均不反向传播。
定义：

```text
cue_gain = [L_route(x_cue_clean) - L_route(x_cue_carrier)]
           / [L_route(x_cue_clean) + 1e-8]
```

`cue_gain>0` 只说明在冻结 surrogate 的既有 target route 上，carrier 在真实
纹理被轻度削弱时仍提供正向 person 证据；它不等价于 fresh victim 已学习到
shortcut。I-SV 必须与 I-SPC-V 做 paired comparison，防止把任意可见扰动导致的
响应变化误称为 semantic shortcut。

### 4.8 第一阶段明确不激活的组件

附件中的完整目标
`L_hide + L_sem + L_latent + L_res + L_cue + L_nt` 不在一次实验中联合启用：

- 不训练 sample-adaptive `H_psi`。它可能把背景、尺度和共现 non-target 编入
  `delta_i`，使 semantic-family 与样本泄漏无法区分；
- 不把外部 `G(delta_i)` concentration 加入优化。微幅、零中心的 `delta` 与
  ImageNet/CLIP 输入分布不匹配，第一轮先用结构 proxy 与检测 residual 取证；
- 不训练 `P_l` projection head。projection 可以把本来不一致的 residual 映射成
  表面紧簇；当前按 P3/P4/P5 在原生通道空间分别维护 `mu_s`；
- 不激活 `L_cue`。过强外观弱化会使原标签失真并产生新的 augmentation artifact，
  因此第一轮只做 4.7 的 matched held-out audit；
- 不按 P3/P4/P5 切换三套 carrier。先用同一个 canonical carrier 跨尺度渲染并
  报告 small/medium/large retention，只有小目标结构确实消失才单独立项；
- 不激活 `L_nt` 或正交投影。carrier 可学性与 non-target 梯度路由必须分开归因。

余弦集中还必须同时报告 residual norm、zero-norm ratio、valid coverage 和 route
effect；否则接近零的 residual 也可能得到看似良好的相似度。

## 5. 因果实验

### 5.1 Phase A：语义结构与 class/sample 平衡

| Arm | Basis/variant | Residual loss | 作用 |
|---|---|---|---|
| I-C2LM | 当前 phase-scrambled natural low+mid | CICR | 现有自然频谱参考 |
| I-SPC-F | 与固定 semantic template 精确 amplitude-matched 的单一 phase-scrambled control | CICR | fixed semantic phase control |
| I-SF | 单一 phase-preserved semantic template | CICR | class-wise 固定载荷 |
| I-SPC-V | 与四个 semantic variants 分别 amplitude-matched 的 4 个 phase-scrambled controls | CICR | variant semantic phase control |
| I-SV | 4 个共同 shape、不同 texture variants | CICR | class/sample 中间态候选 |

主要 contrasts：

1. `I-SF - I-SPC-F`：固定载荷下、相同 amplitude spectrum 的 semantic
   phase/shape gain；
2. `I-SV - I-SPC-V`：四变体下、逐 variant amplitude-matched 的 semantic
   structure gain；
3. `I-SV - I-SF`：从 exact class-wise 到四变体层级载荷的代价；
4. `I-SV - I-C2LM`：新载体对现有背景低/中频候选的净收益。

### 5.2 Phase B：变换残差集中

只在 Phase A mechanical gates 全部通过且 I-SV 没有触发独立 failure signal 时运行：

| Arm | EOT forward | TRC | 作用 |
|---|---|---|---|
| I-SV-E0 | 2 samples | 0 | matched EOT control |
| I-SV-E1 | 2 samples | 1 | 检验 TRC 的独立收益 |

两臂从 Phase A 的同一个 I-SV final coefficient、同一个冻结 prototype-bank
snapshot 和同一个 optimizer reset 状态开始；各自再运行 40 步。不得让 E1 继承
E0 的任何更新，也不得从不同随机初始化开始。

### 5.3 后续组件的进入顺序（不属于本 Spec 的运行范围）

只有 Phase A/B 和 shortcut sufficiency audit 同时通过，才按下列单变量顺序另建
Spec；不得直接跳到完整 INN：

1. 冻结 encoder 的 `L_latent` off/on，先固定 encoder checkpoint、输入可视化映射
   和 amplitude-matched negative control；
2. `L_cue` off/on，保留无 carrier 的同变换 control，并限制弱化强度；
3. 若 small-person structure retention 明确失败，再比较统一 carrier 与固定的
   scale-adaptive simplification；
4. carrier 机制成立后，再比较 non-target response preservation 与正交梯度投影；
5. 只有确定性 carrier bank 已证明有效且样本泄漏审计可做时，才评估
   sample-adaptive hiding network。

## 6. 输入与冻结协议

- Dataset：Pascal VOC train；
- Target：person，class id `14`；
- Shared split：`research_workspace/sources/TAUSB-ALCE-CTX-AUDIT-v1.json`；
- Split hash：
  `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1`；
- Label hash：
  `0c8b6f6424061bc31b84ddf42b7370dcbd074f26805433d0ba275c24815e3248`；
- Calibration：32 person-only + 32 person-cooccur；
- Held-out：32 person-only + 64 person-cooccur；
- Surrogate：VOC20 YOLOv8，checkpoint hash：
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`；
- Source manifest：`research_workspace/sources/bsc_background_manifest.json`；
- Seed：0；image size：640；forced pseudo support；
- Primary statistical unit：image；一图多 person 先取 median；
- Paired stratified bootstrap：seed `2110`，10,000 iterations，按
  person-only/person-cooccur 分层；
- held-out 不参与 basis、variant、weight、prototype、early stop 或阈值选择。

### 6.1 Source contamination screen

正式优化前必须在 frozen surrogate 上检查 anchor 和四个 donor：

- person detection confidence 必须 `<0.05`；
- 任一 VOC20 detection confidence 必须 `<0.25`；
- anchor 与 donor 只用于外部 carrier 构造，不得与 VOC train/val image hash 重复；
- 任一项失败则 stop，不得根据 held-out CICR 挑选替代图像；替代 source 必须先
  修改 Spec 并由用户重新批准。

## 7. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | 新 `ue_framework/methods/semantic_residual_carrier.py` | phase-amplitude variants、matched control、16-basis decomposition、hash | phase preservation、spectrum equality、rank、determinism tests |
| 2 | 复用 `instance_canonical_carrier.py` | 不改 renderer 语义，只允许 variant-index input | exact old-path regression、support/overlap tests |
| 3 | 新 `ue_framework/methods/sirc_probe.py` | Phase A/B arms、TRC、source screen、metrics/gates | synthetic sink probe、finite backward、held-out bank immutable |
| 4 | 新 `ue_framework/tools/probe_tausb_sirc.py` 与 formal YAML | CLI/config 到 active sink | config parse、validate-only hash、feature-off path |
| 5 | local real smoke | 2 images、至少 2 person、I-SPC-V/I-SV 各一次 forward，I-SV-E1 一次 backward | real TAL、variant assignment、finite loss/grad、artifact schema |

本地 smoke 只证明机械链路，不证明 semantic carrier 或 UE 有效。

## 8. 远程运行

- 入口：`python -u -m ue_framework.tools.probe_tausb_sirc`；
- Config：`ue_framework/configs/exp_voc_person_tausb_sirc_probe.yaml`；
- Stage：`all`；Device：single GPU；Seed：0；
- Remote project root：`/root/autodl-tmp/ue_project`；
- ExpID：`TAUSB-SIRC-MECH-S0`；
- artifact root：`/root/autodl-tmp/ue_project/runs_research/TAUSB-SIRC-v1`；
- 预计时长：单 GPU 小于 2 小时，以 pre-run smoke 实测为准；
- fresh root only：路径已存在即 fail closed；不删除、不覆盖、不 resume；
- 正式运行前必须通过独立 `pre-run-implementation-review` 并绑定 branch/commit。

建议内层命令：

```bash
cd /root/autodl-tmp/ue_project
python -u -m ue_framework.tools.probe_tausb_sirc \
  --config ue_framework/configs/exp_voc_person_tausb_sirc_probe.yaml \
  --stage all \
  --device 0
```

## 9. Research Contract

### 9.1 Hypothesis

在频谱幅度、预算、support、renderer 和优化容量匹配时，保留共同全局 phase/shape
且允许少量 texture variation 的 instance-canonical carrier，会比 phase-scrambled
control 在未参与优化的 person 实例上产生更一致且更耐变换的 YOLOv8 feature
residual；这一收益不会以显著增加 non-target 或 box residual leakage 为代价。

### 9.2 Success Signal

Phase A 必须同时满足：

1. mechanical：所有 arm basis rank `>=8`、initial/active RMS ratio `<=1.05`、
   active Linf `<=16/255`、outside-support max `=0`、所有值 finite；
2. structural：I-SV 四变体之间 band-limited gradient NCC median `>=0.70`，
   I-SV 对 anchor median `>=0.60`，I-SPC-V 对 anchor median `<=0.20`；
   `semantic_proxy_cosine(I-SV)-semantic_proxy_cosine(I-SPC-V) >=0.10`；
3. texture diversity：四个 I-SV variant 的 pairwise normalized amplitude distance
   median `>=0.10`；
4. I-SV held-out image-level Instance-CICR median `>=0.60`、Q25 `>=0.20`、
   valid-instance coverage `>=0.75`；
5. `CICR(I-SV)-CICR(I-SPC-V) >=0.05`，paired bootstrap 95% CI lower bound
   `>0`；`CICR(I-SF)-CICR(I-SPC-F) >=0.05`；
6. `CICR(I-SV) >= CICR(I-SF)-0.05`，证明四变体未明显破坏共同 shortcut；
7. I-SV 的 affine/blur/grayscale/JPEG50 CICR retention median 均为 identity 的
   `>=0.75`；
8. I-SV easy classification + box-teacher route loss 相对初始下降 `>=10%`；
9. I-SV held-out `cue_gain` median `>=0.10`，且
   `cue_gain(I-SV)-cue_gain(I-SPC-V) >=0.05`、paired bootstrap 95% CI lower
   bound `>0`；
10. I-SV non-target/target logit residual ratio `<=1.10 * I-SPC-V`，box residual
   energy `<=1.10 * I-SPC-V`；
11. person-only 与 person-cooccur 的 I-SV CICR median 均 `>=0.50`，两组差值
    absolute `<=0.20`；
12. coefficient saturation `<0.25`、active basis fraction `>=0.25`、top-1
    energy share `<0.80`、high-frequency energy ratio `<=0.30`，且 residual
    zero-norm ratio `<0.10`。

Phase B 必须同时满足：

1. I-SV-E1 相对 I-SV-E0 的 transformed CICR median gain `>=0.05`，paired
   bootstrap 95% CI lower bound `>0`；
2. identity CICR 降低不超过 `0.03`；
3. non-target/target residual ratio 与 box residual 分别不超过 E0 的 `1.10x`；
4. gradient、coefficient、prototype、pattern 全部 finite，coverage `>=0.75`。

只有 Phase A 与 Phase B 都 PASS，才支持“hierarchical semantic residual carrier”
进入 poisoned-dataset/fresh-victim Spec。

### 9.3 Failure Signal

以下任一项成立即作为独立 failure evidence：

1. source contamination screen 发现 person 或高置信 VOC20 object；
2. I-SF 与 I-SPC-F、或任一 I-SV/I-SPC-V variant 的 amplitude spectrum
   不匹配超过相对误差 `1e-5`；
3. semantic carrier 在 JND/clamp/instance warp 后对 anchor 的结构 NCC `<0.30`，
   或 `semantic_proxy_cosine(I-SV)-semantic_proxy_cosine(I-SPC-V) <0.03`；
4. I-SF 明显优于 I-SPC-F，但 I-SV CICR 比 I-SF 低 `>0.15`，说明收益依赖精确
   重复像素而不是共享语义结构；
5. I-SV calibration CICR gain `>=0.10`，但 held-out gain `<0.02`，说明
   carrier/coefficients 过拟合 calibration；
6. I-SV 的 `cue_gain <=0`，或其相对 I-SPC-V 的 paired cue gain `<=0`，说明
   聚合响应没有提供可辨识的 semantic shortcut evidence；
7. I-SV target CICR 提升时 non-target/target residual ratio 或 box residual
   超过 I-SPC-V 的 `1.25x`；
8. TRC 使 identity route effect 恶化 `>20%`，或 non-target leakage 增加
   `>1.25x`；
9. high-frequency energy `>0.40`，说明 semantic hiding 退化为高频载荷；
10. valid-instance coverage `<0.60` 或 residual zero-norm ratio `>=0.25`，无法
    可靠评价 instance residual；
11. coefficient/pattern 饱和、NaN/Inf、split overlap、hash 变化或 held-out
    参与任何选择。

### 9.4 Metric & Split

- Primary：held-out image-level Instance-CICR、I-SV vs I-SPC-V paired delta、
  transformed CICR retention；
- Secondary：route effect、shortcut `cue_gain`、structure NCC、semantic proxy
  cosine、texture diversity、scale/cooccur 分组；
- Protection：non-target/target logit residual ratio、non-target FPN residual、
  box residual、target-vs-preservation gradient cosine；
- Mechanical/quality proxy：active RMS/Linf、saturation、frequency energy、
  support containment、PSNR proxy；
- Split：冻结 calibration/held-out；primary unit 为 image；
- JPEG/blur/gray 只属于额外 mechanism robustness audit，不是 clean VOC eval。

### 9.5 Stop Condition

- source/split/label/checkpoint/config/basis/variant hash 不一致；
- source contamination、mechanical、frequency 或 spectrum matching gate 失败；
- artifact root 已存在；
- NaN、Inf、OOM、Traceback、进程消失、GPU 无活动或日志不增长；
- Phase A 触发任一 failure signal 时不进入 Phase B；
- Phase B 失败时不进入 poisoned dataset 或 victim training。

### 9.6 Claim Boundary

- 该 probe 只能证明或否定 surrogate-level semantic residual mechanism；
- 不证明 fresh-victim unlearnability，不产生 target/non-target mAP；
- 不证明完整 INN deep hiding 在 detection 中有效；
- semantic image 的 robustness 是待验证假设，不是由自然图像身份自动保证；
- forced pseudo support 不能描述成真实 instance mask；
- 单 seed 结果只能标 `tentative`；
- 未有真实 poisoned dataset 时不得用 PSNR proxy 支撑数据集视觉质量声明。

## 10. Pre-run Review

- reviewed branch / commit：pending；
- exact command：pending；
- source contamination evidence：pending；
- phase/amplitude reconstruction evidence：pending；
- config → carrier → renderer → residual loss/metric sink：pending；
- baseline/feature-off evidence：pending；
- output non-overwrite：pending；
- result：`pending`。

## 11. 结果落盘

- Remote artifacts：
  `research_workspace/experiments/TAUSB-SIRC-MECH-S0/remote_artifacts/`；
- Metrics summary：
  `research_workspace/experiments/TAUSB-SIRC-MECH-S0/metrics-summary.json`；
- H→E→N：
  `research_workspace/experiments/TAUSB-SIRC-MECH-S0/analysis/`；
- Experiment ledger：`research_workspace/00-实验记录.md`；
- STATE candidate：只有 artifacts、metrics 与 H→E→N 完成后提出，由用户决定。
