---
spec_id: TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3
title: 样本自适应语义深藏、扰动隐特征聚合与检测残差一致性的选择性不可学习验证
status: approved
experiment_type: method
csv: issues/TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3.csv
created: 2026-08-10
approved: 2026-08-11
approval_evidence: user explicitly approved the Spec and requested the first single-image validation
supersedes_draft: TAUSB-MALC-CGR-NTLA-MAP50-v3
historical_input: TAUSB-SIRC-MALC-CGR-MAP50-v2
---

# 样本自适应语义深藏、扰动隐特征聚合与检测残差一致性的选择性不可学习验证

## 1. 问题锚点

- 用户方向修订：旧 `tausb_mask` 不再是当前方法主线，只作为历史
  AP50 参照；固定 Fourier basis 与共享 48 维系数的 SIRC 也不是本轮
  carrier。
- 当前可测主张：将同一张高语义图像作为 `person` 类的固定隐藏
  载荷，但由宿主条件化图像隐藏网络对每个 person 实例生成不同的
  `delta_ij`；再分别使这些扰动的隐特征集中、对 YOLO 的 clean-to-poison
  实例残差方向一致，从而形成兼有 sample-wise 适应性与 class-wise 稳定性
  的易学捷径。
- 这里的“难以预测”特指 victim 的**训练偏好**，不是说 person 区域无法定位：
  clean person 外观、尺度、姿态和背景的条件熵较高，而 carrier 诱导特征只在
  person 训练实例上重复出现且跨样本方差较低。假设 victim 会优先拟合这条更简单
  的相关性；clean test 不含 carrier 时 person AP50 因此下降。该解释必须由
  fresh-victim 的 carrier counterfactual 验证，不能只由 D-LFC/CICR 数值推断。
- 非目标主张：clean 分支作 teacher，对 clean TAL 真实非目标前景位置
  加入显式、类平衡的 clean/poison assigned-class logit 对齐损失；目标
  复合梯度的攻击分量必须投影到逐类非目标梯度的正交补空间。
- 本轮问题：该 sample-adaptive semantic hiding carrier 是否能在 held-out person
  实例上同时提高扰动隐特征集中度与检测残差方向一致性；显式
  logit 对齐在正交攻击之上是否能继续降低非目标漂移，而不使目标
  捷径梯度或渲染扰动坍缩？
- 非目标：首轮不使用 ALCE、旧 band-aware Fourier bank、MALC 替代式合并、
  late repair、EOT、JPEG/blur/gray、多 final-secret carrier、多 target class、迁移、多 seed
  或鲁棒性宣称；机制门禁不通过不训练 fresh victim。

## 2. Idea Source 与方案比较

- 来源：用户明确的方法修订 + *Semantic Deep Hiding for Robust Unlearnable
  Examples* 中 Deep Hiding/LFC 的分离机制 + 已有 YOLO clean TAL/PAG 实例
  残差与 CGR 代码。
- 本地文献：`D:/Googledownload/2406.17349v1 (1).pdf`。
- 历史证据：`TAUSB-MALC-GRAD-GEOMETRY-S0` 只说明固定 SIRC 参数更新在
  渲染 pattern 中被强烈压缩，不能证明当前 sample-adaptive hiding 方案有效或
  无效。

| 方案 | 样本依赖 | 优点 | 主要问题 | 决策 |
|---|---|---|---|---|
| A. 固定 SIRC/Fourier pattern | 仅图像级 variant，系数共享 | 参数少、易投影 | 不符合用户定义；不是真正的图藏图；已观察到参数到 pattern 压缩 | 放弃为主方法 |
| B. 每张图独立存储 `delta_i` | 完全 sample-wise | 表达能力强 | 6,095 张的独立参数大；类间共性只靠损失；难泛化到新实例 | 仅作小样本上界，首轮不运行 |
| C. 宿主条件化语义隐藏网络 | `delta_ij=eps*tanh(H_phi(x_ij,s))` | 同一 secret 保持类级语义，输出随宿主变化；可端到端优化 | 容量和训练成本更高；可能编码背景/共现类；需防止 reveal 共谋坍缩 | **采用** |

最终 carrier 的两种组织方式另行比较：

| final-secret 方案 | 易学性 | 鲁棒性潜力 | 因果可解释性 | 本轮决策 |
|---|---|---|---|---|
| 单一、固定、已筛查 secret | 重复度最高，最容易形成低方差捷径 | 可能记住精确 carrier | 最容易验证“重复载体导致 shortcut” | **首轮 P1-V 采用** |
| 同语义、多细节 secret family | 共享高层语义且保留细节变化 | 可能减少精确像素记忆 | 同时改变 family diversity，首次实验难归因 | 条件性下一 Spec；首轮只保留候选，不混入 victim |

`bg-mountain-cloud-10` 与 `bg-road-mountain-11` 只能称为自然图像的宽泛
“山地/天空”候选族，并非扩散模型控制变量。未来若采用 diffusion family，必须冻结
生成模型、prompt、negative prompt、seed 列表、采样器与输出 hash，并重新执行同一
人工/频谱/VOC20/VOC-data 去重门禁。

本轮使用检测适配的轻量 DWT/coupling hiding module，不声称精确复现论文
HiNet。若实施时无法满足可恢复语义和 `eps` 门禁，必须停止，不得静默换成
固定 Fourier carrier。

首轮架构冻结为：`256x256` 输入、1-level fixed Haar DWT、host/secret
wavelet 通道拼接后的 4 个 affine coupling blocks、子网络宽度 64、3x3
convolution + LeakyReLU(0.2)、无 BatchNorm/Dropout；逆 DWT 后由两层 3x3
residual adapter `omega` 输出 3 通道 `r_ij`。实施前必须将每层通道数、
参数量和 architecture hash 写入 pre-run packet；不得因为依赖或显存问题静默
更换架构。

## 3. 机制与接入

### 3.1 固定高语义载荷与 sample-adaptive hiding

- 隐藏图不能只依赖“person-free”人工标签或 surrogate 的一个低置信度。
  每个候选 secret 必须同时通过：
  1. 人工内容审计：无 person，无可见 VOC20 其他类实例；
  2. 频谱门禁：去 DC 后半径 `[1,24)` 能量占比 `>=0.85`、`[64,+inf)`
     `<=0.10`、`r90<=48`；
  3. 冻结 VOC20 surrogate 筛查：`person_max<=0.05`、其他任一 VOC20
     类 `max_conf<=0.10`；
  4. source hash 与 VOC train/val 无重复。
- 筛查协议和当前 11 张授权图的结果冻结在
  `research_workspace/sources/low_frequency_secret_screen_v1.json`。检测模型未报告
  某类不能覆盖人工审计；例如 `bg-cliff-beach-07` 含有可见 boat，必须
  拒绝。
- 首轮 person carrier 的最终候选冻结为 `bg-building-sky-09`，SHA-256
  `66bd89ebef12b21d578341e945d56ed315372b213e52c2cc07d2110543cc6a48`。其去 DC
  频谱中 `[1,24)` 占 `92.33%`、`[64,+inf)` 占 `3.60%`、`r90=15.51`，
  surrogate 未产生 person detection，最高 VOC20 非目标置信度为 `0.005430`。
  该图含清晰 building 语义，但 building 本身不是 VOC20 类；预处理固定为 center-square
  crop 后缩放到 `256x256`，实施时还必须冻结预处理后 tensor hash。
- 用户指定的本地 VOC 根已完成只读输入审计：train/val 图像数为
  `16551/4952`，person 图像数为 `6095/2007`，image-label stems 一一对应，
  20 类 YOLO 标签合法，4 张 secret source 与全部 VOC JPEG 的 SHA-256
  重复数为 0。证据为 `research_workspace/sources/voc_input_audit_v1.json`。
  授权原图及 center-square `256x256` 预处理版已持久化到
  `research_workspace/sources/secret_assets/`，原图/PNG/uint8/float32 hashes 冻结在
  `manifest.json`；正式 config 不得引用 Temp/Desktop 路径。远程 AutoDL
  拷贝仍必须在 pre-run 再对齐同一 manifest。
- 上述选择只能证明 secret source 是低频且低 VOC 响应；图像隐藏网络仍可能
  把该信息编码到高频 `delta` 中，因此必须另外落盘实际扰动的频谱
  能量，不得由 secret 的低频性直接宣称扰动鲁棒。
- hiding pretrain 不允许只用最终 secret，否则 `R_psi` 可以通过始终
  输出 building secret 获得虚假 revealing loss。预训练只能使用通过同一人工、
  频谱、VOC20 响应与 hash 门禁的非 primary secrets；不足 3 个时停止并
  请用户补充图像，不得使用未通过图像补数。`bg-building-sky-09` 完全留作
  unseen-secret held-out 与最终 person carrier。
- 对第 `i` 张图中第 `j` 个 person GT box，从 clean image 取该矩形 crop
  `u_ij`，确定性缩放到 `256x256`。本 VOC/YOLO 协议没有全量可信的
  person instance mask，因此首轮将“person实例为宿主”精确实现为
  **每个 person GT box 独立调用隐藏网络**；不加在线实例分割模型。这里使用 bbox
  是由现有标注协议决定的，不是因为 person 语义“难以预测”。
- 隐藏网络和 reveal decoder：

```text
r_ij       = H_(phi,omega)(u_ij, s)
delta_ij   = eps * tanh(r_ij)
u_stego_ij = clamp(u_ij + delta_ij, 0, 1)
s_hat_ij   = R_psi(u_stego_ij)
```

- `H_phi` 必须因 `u_ij` 不同产生不同 `delta_ij`；不允许使用 image-id lookup
  表伪造 sample-adaptive。
- hiding pretrain 在多 secret bank 上同时优化 cover distortion 与 secret revealing；进入
  检测优化后
  冻结 `R_psi` 和 hiding trunk，只优化小型 residual head/adapter `omega`。这防止
  reveal decoder 与 encoder 共谋退化，也将逐类梯度投影限制在可控参数空间。
- 固定 decoder 的 revealing loss：

```text
L_reveal = mean_ij [ L1(s_hat_ij, s) + 0.2 * (1 - SSIM(s_hat_ij, s)) ]
```

- `delta_ij` 缩放回对应 person GT box；该 box 就是首轮的完整嵌入区域。
  同图重叠 person boxes 取平均，所有 person boxes 并集外严格为 0，整图
  实际 `Linf<=16/255`。首轮不使用 forced pseudo ellipse，不得将 bbox 方案
  描述为真实 instance mask。
- 必须分别记录 person box 与非目标 GT box 的 overlap ratio，并在 person-only/
  cooccur/overlap 组报告 D-LFC、CICR 和 NLA；首轮不根据 overlap 动态改变 support，
  避免不同图像暗中使用不同嵌入协议。
- 首轮 EOT 和 JND 均关闭；只保留 hard `Linf`、person bbox union support 与最终像素
  clamp，避免把语义隐藏能力与可见性调制混在同一首轮实验。

### 3.2 检测器适配的扰动隐特征聚合（D-LFC）

LFC 约束的仍是“扰动本身”，不得与下节的 clean-to-poison 检测残差
合并为一个损失。

```text
delta_vis_ij = clamp(0.5 + delta_ij / (2*eps), 0, 1)
h_ij_l       = normalize(GAP(Z_cls_l(delta_vis_ij))) , l in {P3,P4,P5}
q_l          = stopgrad(normalize(mean_calibration h_ij_l))
L_D-LFC      = mean_l mean_valid_ij [1 - cosine(h_ij_l, q_l)]
```

- `Z_cls_l` 使用冻结 VOC20 YOLO surrogate 的 P3/P4/P5 `cv3` pre-logit
  classification-tower feature，不加额外 ResNet-18 checkpoint。
- 输入是单个 person 的 canonical perturbation crop，不是含有背景的整图
  `delta`；各尺度先求实例均值，再等权平均，避免 P3 或大实例支配。
- `q_l` 只使用固定 calibration split 拟合，held-out 和 victim 不得回写。
- 同时记录 pixel diversity 和隐特征集中；若不同 `delta_ij` 在像素上几乎
  相同，不得将高 LFC 称为 sample-adaptive 成功。

### 3.3 检测实例残差方向一致性（CICR）

clean 分支提供冻结 real TAL/PAG assignment。对 person GT 实例 `j`、尺度
`l`、对应 positive `a`：

```text
r_ijl = sum_a w_hat_ijla *
        [Z_cls_l(x_poison)_ia - stopgrad(Z_cls_l(x_clean)_ia)]

mu_l       = stopgrad(normalize(mean_calibration normalize(r_ijl)))
L_CICR     = mean_l mean_direction_valid [1 - cosine(r_ijl, mu_l)]
L_floor    = mean_l mean_eligible relu(tau_l - RMS(r_ijl)) / (tau_l + eps)
```

- `w_hat` 由 clean person assigned score 归一化，并受 PAG 与 `target_gt_idx`
  约束；不在 background/non-target positives 上聚合。
- `mu_l` 与 `tau_l=0.5*Q25(RMS(r_ijl))` 只由固定 calibration split 拟合后
  冻结。
- coverage 使用 detector-eligible 分母：先记录 clean TAL/PAG 在该尺度是否
  真正为该 GT 分配 positive，再区分 pooling-valid 与 direction-valid；低能量
  eligible 实例仍必须进入 `L_floor` 分母。

### 3.4 目标复合目标

```text
L_target = lambda_easy   * L_easy_cls_box_teacher
         + lambda_reveal * L_reveal
         + lambda_lfc    * L_D-LFC
         + lambda_cicr   * L_CICR
         + lambda_floor  * L_floor
         + lambda_rms    * L_rms
```

- `L_easy_cls_box_teacher` 使 poisoned person 在冻结 surrogate 上按原标签易学，
  并使用 clean box teacher；这是训练时捷径诱导，不是 test-time evasion。
- `L_reveal` 保证扰动仍携带固定高语义载荷；`L_D-LFC` 保证不同
  样本扰动在检测适配的隐空间聚合；`L_CICR/L_floor` 保证它们对
  person 检测特征产生方向统一且非零的影响。
- 权重只能在固定 warm-up batches 上按对 `omega` 的 median gradient norm 一次标定，
  随后冻结；不得根据 AP50 事后调权。

### 3.5 显式非目标 logit 对齐（NLA）

clean surrogate 是 teacher。对 batch 中每个活跃非目标类 `c`，`A_c` 只包含
clean real TAL 分配给该类 GT 的前景 positive：

```text
L_nt,c = mean_(a in A_c) SmoothL1(
           z_poison(a,c), stopgrad(z_clean(a,c)))

L_NLA  = mean_(c in active non-target classes) L_nt,c
```

- 对齐的是 logit，不是 sigmoid 后概率；首轮只对齐 assigned/GT class `c`，
  不一次强制复制全部 19 维响应。
- 每类先求均值，再对 active classes 宏平均；不将 background、person positives、
  伪匹配位置或 anchor 数量混入权重。
- `lambda_nla` 使用 warm-up 梯度规模标定，保护梯度初始目标为投影后
  target gradient norm 的 `0.25x`；必须落盘标定值与 clip 状态。
- 额外记录其余非目标类 logit vector 的 shape drift，但不纳入首轮损失。

### 3.6 目标攻击梯度与非目标梯度正交（CGR）

对每个 active `c` 计算：

```text
g_t   = grad_omega(L_target)
g_c   = normalize(grad_omega(L_nt,c))
G     = stack(g_c)
Pnull = I - V_r^T V_r                 # SVD relative tolerance 1e-4
g_atk = Pnull * g_t
g_nt  = grad_omega(L_NLA)
g     = g_atk + lambda_nla * g_nt
omega <- omega - learning_rate * g
```

- “正交”约束的是 target attack component `g_atk`；最终更新还显式包含
  使 `L_NLA` 下降的保护分量，因此不应宣称“最终总梯度与非目标梯度
  正交”。
- 首轮对所有 active non-target classes 进行投影，不只在 near-boundary 触发；
  无 active class 时只用 `g_t`。
- 后选更新必须在真实非线性 forward 中重算逐类 assigned-class probability
  drop，容差仍为 `0.005`；最多 5 次回溯，仍不满足则 skip。
- 只对冻结 hiding trunk 之上的 residual adapter `omega` 建立 SVD 投影。若为了
  省显存改为只投影最后一层且未预注册，必须停止并修订 Spec。

### 3.7 参数链与回退

canonical 参数链：

```text
config/CLI
-> generate stage
-> person crop + fixed semantic secret
-> frozen hiding trunk + trainable residual adapter
-> instance renderer
-> frozen YOLO clean/poison branches
-> D-LFC + CICR + easy route
-> per-class NLA gradients
-> CGR projector + explicit protection update
-> frozen carrier materialization
```

必须提供独立开关：`enable_dlfc`、`enable_cicr`、`enable_cgr`、
`enable_nla_loss`。`enable_deep_hiding=false` 在正式方法中 fail closed，不得静默回退到
`tausb_mask` 或固定 SIRC。clean C0 是唯一无投毒基线。

## 4. 最小判别实验

### 4.1 Hiding 机械门禁

- 只训练/评估 hiding module，不启动 victim。
- 预训练使用固定 calibration person crops 与至少 3 张通过全部筛查的非
  primary secrets；评估使用只读 held-out person crops，同时覆盖 seen
  secrets 与 unseen `bg-building-sky-09`。多 secret 在此只用于排除 decoder
  常量输出；最终 person carrier 仍只有一张。
- 对每个 recovered secret，在冻结的 pretrain+unseen secret bank 中做固定
  L1/SSIM 最近邻
  retrieval；这是排除 decoder 恒等输出某一 secret 的主门禁。
- 额外将 unseen primary secret 与其 phase-scrambled 版本输入同一冻结 hiding checkpoint，
  仅记录语义结构是否改变扰动/LFC 特征，不为 control 重新训练网络。
- 记录 secret retrieval accuracy、recovery SSIM/L1、actual Linf、PSNR、频谱
  能量、跨宿主 pixel cosine/diversity。
- 为排除 sample-adaptive 扰动只是把宿主/共现信息重新编码，在 calibration
  上只读拟合一个线性 probe，从 `D-LFC` 特征预测 person-only/cooccur 二分组
  与活跃非目标类；held-out 只读评估。该 probe 仅作泄漏门禁，不参与
  carrier 反向传播。

### 4.2 目标机制 arms

| Arm | sample-adaptive hiding | D-LFC | CICR/floor | CGR | 显式 NLA | 用途 |
|---|---:|---:|---:|---:|---:|---|
| T0 | on | off | off | off | off | 图藏图 + easy route 基线 |
| T1 | on | on | on | off | off | 检验隐特征聚合和检测残差一致的联合作用 |

- T0/T1 从同一 hiding checkpoint、adapter 初值、batch 顺序与冻结 prototype 分离开始。
- 在相同 matched batches 上额外记录 `grad(L_D-LFC)` 与 `grad(L_CICR)`
  cosine；若两者强冲突，下一 Spec 再做单组件调和，本轮不就地换
  PCGrad 或调 AP 权重。

### 4.3 保护机制 arms

| Arm | Target base | target→non-target 正交 | 显式 NLA | 用途 |
|---|---|---:|---:|---|
| P0 | 通过门禁的 T1 | on | off | 只验证正交攻击的保护边界 |
| P1 | 通过门禁的 T1 | on | on | 检验显式 logit 对齐的额外收益与攻击代价 |

- P0/P1 从同一 T1 initial adapter 状态开始，不得串行继承。
- 先只运行短 GPU mechanism：最少 16 calibration batches、24 held-out batches、8-step
  matched microtrajectory；如果 hiding 模块已需要单独训练，该 checkpoint 必须在所有
  arms 共享。

### 4.4 Fresh-victim arms

只有 hiding、T1 target 机制与 P1 protection 门禁全部通过才允许：

- **C0**：clean VOC2007+2012，YOLOv8n-style victim from scratch。
- **P1-V**：冻结 hiding trunk、adapter、secret 和 renderer，投毒全部 6,095 张含
  person 的 train images，其余训练图保持 clean；用该数据从头训练独立
  victim。
- 为节省成本，首轮不训练 T0/T1/P0 victim；因此 fresh-victim 只证明完整
  P1 方法，各模块的独立因果贡献只由短 mechanism arms 支持。
- P1-V 训练完后冻结 checkpoint，对同一 VOC val 执行两次评估：
  1. `clean-val`：不嵌入 carrier，这是 UE 和非目标保护的唯一 primary metric；
  2. `target-carrier-val`：只为可证伪机制审计，用冻结 renderer 将同一
     `bg-building-sky-09` carrier 加到 val 的 person GT boxes，不更新任何权重。
  若 P1-V 在 clean-val 的 person AP50 下降，但 target-carrier-val 不能明显恢复，
  则不得声称 victim 学到了 carrier shortcut；这更符合无结构损坏或训练失配。
- 为直接检验“更易学”而不只是最终相关，保留 P1-V epoch
  `{1,5,10,20}` 的小型 audit checkpoint，在固定、只读 person audit subset 上比较
  同一宿主的 carrier 版本与 remove-carrier clean counterfactual：

```text
R_e = (L_person_cls(clean_counterfactual; theta_e)
       - L_person_cls(carrier; theta_e))
      / (abs(L_person_cls(clean_counterfactual; theta_e)) + 1e-8)
```

  `L_person_cls` 仅在固定 GT-person 对应的分类监督上计算；两个 view 共享
  image id、resize、label 和 assignment 协议。`R_e>0` 表示当前 victim 对 carrier view
  的 person 监督拟合得更快。该 audit 不回传，不用于选 epoch 或调参。
- 另外在 person-free val 上将 carrier 移植到尺度/位置匹配的非目标 GT boxes，
  只记录 person logit/假阳性变化。该 intervention 不属于真实 clean-test 协议，
  无论结果如何都不代替 19 类 clean AP50。

## 5. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | 新 `semantic_hiding_carrier.py` | 目标实例 crop、fixed secret、DWT/coupling hiding、reveal decoder、bounded residual | 同 host 确定性、异 host 输出差异、decoder freeze、Linf/support/overlap/finite backward |
| 2 | 新 `detector_lfc.py` | canonical `delta_vis` 的 YOLO P3/P4/P5 扰动特征集中 | 冻结 extractor、prototype held-out 只读、multi-scale balance、零/常数扰动 fail closed |
| 3 | `instance_cicr.py` / `malc.py` | 保留独立 CICR/floor，改用 detector-eligible coverage | 无 assignment、零 score、零 residual、单尺度、非 finite 首因测试 |
| 4 | 新 `non_target_logit_alignment.py` | clean TAL 非目标 positives 的逐类 assigned logit SmoothL1 | teacher detach、类不平衡、target/background 排除、exact-clean=0 |
| 5 | `constraint_gradient_router.py` integration | 逐类 NLA 梯度 SVD 正交 + 显式 NLA descent + 5 次回溯 | row-dot、rank0/full-rank、有/无 active class、保护分量符号、非线性拒绝 |
| 6 | config/runtime/generate | 新 method/config、checkpoint provenance、开关回退、冻结 materializer | config parse、Python3.8 AST/import、CLI→loss sink、同 state hash 重渲染 |
| 7 | evaluation | VOC20 命名 AP50、19 类宏平均/delta/retention、person-free/cooccur | synthetic class map、缺类 fail closed、clean split 校验 |
| 8 | pre-run review | 参数链、exact commit、fresh roots、费用与自动关机门禁 | review=`pass` 后才允许 GPU |

实施前不删除旧 SIRC/MALC 代码和 artifacts；新 method 使用独立名称和输出根。

## 6. 数据、运行与成本门禁

- Dataset：VOC2007+2012，20 类；target=`person` / class id `14`。
- 本地授权数据根：`F:/dateset/VOC_0712_Kaggle_Ready/VOC_0712_Kaggle_Ready`；
  实际结构为 `images/{train,val}` 与 `labels/{train,val}`。远程运行可使用
  AutoDL 上的等价拷贝，但必须通过 split count、label/class 映射和 manifest/hash
  对齐，不得只因目录名相同就认为数据一致。
- Method/config：`tausb_sdh` /
  `ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml`。
- 建议 branch / ExpID：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3` /
  `TAUSB-SDH-LFC-CICR-CGR-NLA-S0`。
- 独立 run root：`/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0`；不覆盖旧
  TAUSB/SIRC/MALC artifacts。
- Surrogate/victim：YOLOv8n-style VOC20；clean/poison 分支共享冻结 surrogate。
- Victim 预注册：seed 0、200 epochs、imgsz 640、batch 36、SGD。
- Poison scope：6,095 张含 person 的 train images；其余 train images clean。
- 本地/无卡先完成 import、单元测试、配置、小 tensor backward 和 pre-run review。
- 第一次 GPU 只允许 hiding pilot，上限 20 分钟；不越过首个 finite progress 或
  门禁无改善则自动关闭实例。短 mechanism 目标上限 15 分钟，完成/停止后
  自动关闭。
- 正式 victim 只在用户再次开启 GPU 且所有机制门禁通过后运行；不得用
  bug 修复时间消耗 GPU。

## 7. Research Contract

- **Hypothesis**：clean person 外观的跨样本变异高，而固定高语义 secret 经宿主
  条件化 hiding 后会在所有 person 训练实例上产生低方差、重复出现的
  detector feature，victim 因而优先学习该简单相关性。具体而言，该 sample-wise 扰动
  在 D-LFC 与 CICR 共同约束后，会保留非平凡像素多样性，同时在扰动
  隐空间和 person 检测残差中形成稳定类级方向；将目标攻击分量投影到
  逐类非目标 NLA 梯度的正交补空间，并加入小而显式的 NLA 保护分量，
  能改善 target/non-target Pareto 而不抹除捷径形成。因此，P1-V 应在无 carrier
  的 clean-val 上失去 person 检测能力，但在受控的 target-carrier-val 上部分恢复。
- **Hiding Success Signal**（held-out）：
  1. actual `Linf<=16/255+1/255`，support 外为 0，所有值 finite；
  2. 冻结 secret-bank retrieval top-1 accuracy `>=0.90`，且 unseen primary secret 的 median recovery
     SSIM `>=0.50`；
  3. unseen primary secret 的 recovered image 对真 primary secret 的 L1 至少比对其他
     secret 的最优错误 L1 低 `20%`，排除常量/误识别 decoder；
  4. 异 host 的 canonical `delta_ij` pairwise pixel cosine median `<0.98`，且每通道
     RMS 的变异系数 `>=0.05`，排除退化为同一像素 pattern；
  5. 所有 pretrain/final secrets 均与 VOC train/val 图像 hash 无重复，人工/
     频谱/VOC20 source audit 通过。
  6. held-out 扰动特征的 person-only/cooccur 线性 probe balanced accuracy 不高于
     多数类基线 `+0.10`；活跃非目标类 macro-AUROC 不高于 `0.60`。
  7. 实际 `delta` 的 `[64,+inf)` 频谱能量占比 `<=0.40`；该项是生成
     扰动门禁，不由 secret source 的低频性替代。
- **Target Mechanism Success Signal**（T1 vs T0，held-out）：
  1. D-LFC P3/P4/P5 等权实例 cosine median 提高 `>=0.10`，Q25 `>0`；
  2. detector-eligible CICR residual cosine median 提高 `>=0.10`，Q25 `>0`；
  3. `C_any>=0.80`，至少 2/3 尺度 pooling coverage `>=0.80`，zero-residual ratio
     `<=0.20`，floor-pass `>=0.80`；
  4. T1 的渲染 pattern 相对 T0 发生非平凡分离：`D_pattern>=0.01`；
  5. `grad(D-LFC)` vs `grad(CICR)` cosine median `>=-0.10`，Q25 `>=-0.25`。
- **Protection Success Signal**（P1 vs P0，held-out）：
  1. 类宏平均 assigned-class logit SmoothL1 降低 `>=25%`；
  2. assigned-class probability drop macro `<=0.005`，且至少 80% active classes 不劣于
     P0；
  3. `max dot(g_atk,g_c)<=1e-5`，SVD null dimension `>0`，target attack retention
     median `>=0.20`；
  4. P1 的 CICR cosine median 相对 P0 下降不超过 `0.02`，`D_pattern>=0.80x P0`；
  5. backtrack+skip ratio `<0.50`，无非 finite gradient/SVD/candidate。
- **Fresh-victim Success Signal**（seed 0，tentative）：
  1. `AP50_person(C0)-AP50_person(P1-V)>=0.30`；
  2. `AP50_person(P1-V,target-carrier-val)-AP50_person(P1-V,clean-val)>=0.20`；
  3. 学习动力学中 `R_10>0` 且 `R_20>=0.10`；
  4. `mAP50_non_target_macro(C0)-mAP50_non_target_macro(P1-V)<=0.05`；
  5. 19 类中至少 16 类 AP50 下降 `<=0.10`；
  6. `poisoned_count=6095`，actual Linf/PSNR/LPIPS/per-class AP50 齐全且 finite。
- **Failure Signal**（独立）：
  1. hiding 在 held-out 上无法恢复真 secret，或退化为与 host 无关的固定
     pattern；
  2. D-LFC 提高但 CICR 无提高，表明只聚合了扰动表征，未形成稳定
     detector effect；
  3. 非目标/target 分类塔 residual energy ratio 超过 T0 `1.25x`，或扰动可由
     非目标共现类预测，表明 sample-adaptive generator 编码了 collateral semantics；
  4. CGR nullspace 消失、attack retention `<0.20`，或 NLA 使 CICR/D-pattern 超出上述
     退化阈值；
  5. materialization 不确定、support 外非零、`Linf>16/255+1/255`、计数
     `<0.95*6095`或 frozen checkpoint hash 不一致；
  6. fresh victim 的 non-target macro AP50 下降 `>0.10`，或至少 5/19 类下降
     `>0.15`；
  7. fresh victim 的 clean person AP50 虽下降，但 target-carrier-val 恢复 `<0.10`；
     此时必须把“carrier shortcut”机制解释判为失败；
  8. `R_10<=0` 且 `R_20<0.05`；此时可能仍存在最终 carrier 依赖，但不支持
     “victim 优先学习载体”的动力学声明。
- **Metric & Split**：
  - mechanism：固定 calibration/held-out person-only + person-cooccur，prototype 和权重只由
    calibration 拟合；held-out 只读；
  - primary victim：clean VOC val `AP50_person down` 与 `mAP50_non_target_macro up`；
  - shortcut counterfactual：同一冻结 P1-V 在 clean-val 与 target-carrier-val 的
    person AP50 差；person-free carrier transplant 只作 descriptive secondary metric；
  - learning preference：epoch `{1,5,10,20}` 的 `R_e`，使用固定 audit subset，
    不参与 checkpoint selection；
  - secondary：19 类逐类 AP50/delta/retention、person-free/cooccur non-target AP、
    mAP50_all、hiding/LFC/CICR/CGR/NLA diagnostics；
  - quality：actual Linf、PSNR、LPIPS、perturbed area、poisoned_count；
  - clean validation 不加 robustness transforms。
- **Stop Condition**：任一前置门禁失败不进入下一阶段；此外 NaN/Inf、OOM、
  loss 长时间不变、surrogate/data/hash 不匹配、output root 已存在、类别映射
  不完整、成本超时或自动关机失效时立即停止。
- **Claim Boundary**：hiding/T0/T1/P0/P1 只是 surrogate mechanism evidence；只有 P1-V
  fresh victim 可支持完整方法的 UE 声明；单 seed 只能称 tentative；不声称
  robustness、transferability、独立模块的 fresh-victim 因果贡献或 SOTA。单一
  secret 成功不支持“同语义扩散图族更鲁棒”的声明；该主张需要下一
  Spec 的 single-secret vs frozen semantic-family 对照。

## 8. Pre-run Review

- reviewed branch / commit：`pending`
- exact hiding/T0/T1/P0/P1/C0/P1-V commands：`pending`
- source/secret/data/label/surrogate hashes：`pending`
- CLI/config -> hiding/D-LFC/CICR/NLA/CGR/materializer/metric sinks：`pending`
- feature-off/fail-closed regression：`pending`
- fresh roots/recovery/auto-shutdown：`pending`
- result：`pending`

## 9. 结果落盘

- Hiding artifacts：`pending`
- T0/T1 mechanism artifacts：`pending`
- P0/P1 protection artifacts：`pending`
- C0/P1-V artifacts：`pending`
- Per-class comparison：`pending`
- H→E→N analysis：`pending`
- Experiment ledger：`pending`
- STATE update decision：`pending`
