---
spec_id: TAUSB-BSC-RC-GR-v1
title: 背景频谱载体、分类塔残差一致性与约束梯度路由
status: approved
experiment_type: probe
csv: issues/TAUSB-BSC-RC-GR-v1.csv
created: 2026-07-28
approved: 2026-07-29
---

# 背景频谱载体、分类塔残差一致性与约束梯度路由

## 1. 问题锚点

- STATE 关联：
  - 当前单-seed best 仍为 tentative；
  - seed1/seed2 稳定性和 PSNR/LPIPS/poisoned_count 统计链属于 P0，本 Spec
    不替代这些工作；
  - 本轮只批准 surrogate-only probe，不自动生成正式 poisoned dataset 或训练
    victim。
- 用户 idea：
  1. 从少量 person-free 自然背景图中提取低频/中低频结构，形成简单、稳定、
     可学习的人工 carrier；
  2. 优化 carrier，使不同 person 图像产生一致 feature residual；
  3. 分解 target detection response，区分“背景化/匹配失败”和“保留前景但
     person 分类失败”；
  4. 用 clean branch 作为 non-target teacher，并将 target gradient 限制在
     non-target constraint 的一阶无害子空间。
- 代码事实：
  - 当前模型输出为 `4` 个 decoded box 通道加 `20` 个 class logits；
  - 当前 YOLOv8-style `Detect` head 没有独立 objectness 输出；
  - 当前 ShadowTAL 连续量为：

    ```text
    alignment = sigmoid(person_logit)^alpha * CIoU^beta
    ```

  - 因此 Translucent Patch 的
    `P(objectness) * P(target_class)` 不能原样接入；
  - `ours_mask` 已实现过 per-image mid-frequency pattern、
    TAL-alignment suppression、target class suppression 和 clean/adv
    non-target probability MSE，但当前仓库没有找到对应正式 metrics artifact；
    它只能作为实现/机制对照，不能宣称已失败或有效；
  - 当前 `tausb_mask` 是 universal carrier，Fourier coefficient 在不同 target
    图像间共享；
  - 当前 best 的 Fourier basis 由 real amplitude、zero phase 的单频 basis
    构成，尚不支持带自然图像 phase/空间结构的 basis；
  - 当前 non-target preserve 已在 clean TAL 的真实 non-target foreground 上
    计算 logits/margin，cooccur-specific branch 当前关闭。
- 本轮可测问题：
  1. 自然背景的频谱包络是否比当前 synthetic Fourier basis 更容易产生跨图像
     一致的 person classification-tower residual？
  2. 该收益来自可识别的原背景结构，还是来自去语义后的频谱统计？
  3. low-only carrier 是否真优于 low+mid；还是会因过平滑、support 边界泄漏和
     JND 调制失去有效频带？
  4. shortcut-aligned 的正向 error-minimizing route 与 test-time-evasion
     route，哪一个能产生更稳定、更低 collateral 的 residual？
  5. 在 carrier coefficient 空间中，对 active non-target constraints 做投影后，
     是否仍保留非平凡 target direction？
- 非目标：
  - 不把自然图像低频直接称为“没有语义信息”；
  - 不把 Translucent Patch 的 YOLOv5 objectness 公式称为 YOLOv8 原生输出；
  - 不把 test-time evasion 结果等同于 training-time UE；
  - 不改变 target id、`eps=16/255`、forced pseudo fallback support 或 clean VOC
    evaluation；
  - 不重新启用并宣称历史 cooccur protect 有效；
  - mechanism probe 不构成 fresh-victim UE 有效性证据。

## 2. Idea Source 与文献边界

- 来源类型：新方法假设 + 当前代码信号 + 文献迁移。
- Yi et al., *Towards Effective and Robust Unlearnable Examples Against
  Object Detection*, ICIP 2025：
  - SPS 将同一 pattern 嵌入所有 object boxes，通过降低 poisoned training 的
    object-presence loss 形成易学 shortcut；
  - 中频 SPS 优于极低或极高频；
  - 它是 all-object detector collapse，不证明 `person` selective preservation。
- Yu et al., *Availability Attacks Create Shortcuts*, KDD 2022；
  Ren et al., *Transferable Unlearnable Examples*, ICLR 2023；
  Zhu et al., *Why Do Unlearnable Examples Work: A Novel Perspective of
  Mutual Information*, ICLR 2026：
  - 支持跨样本一致、类内低 covariance、易分离 perturbation 可能更容易被模型
    学成 shortcut；
  - 证据主要来自分类，不能替代 detection/victim 实验。
- Zolfi et al., *The Translucent Patch*, CVPR 2021：
  - 在带独立 objectness 的 YOLO 设置中，作者预实验发现
    `P(objectness)*P(target_class)` 比单独最小化其中之一更有效；
  - 用 clean/patch class confidence 差保持 untargeted detections；
  - 它是测试时、全图 on-sensor physical evasion attack，不是训练时 UE；
  - 论文的 non-target 结果来自特定交通场景和 detector-generated labels，
    不能直接外推到 VOC 19 类。
- Mahmood and Elhamifar, *Semantic-Aware Multi-Label Adversarial Attacks*,
  CVPR 2024：
  - 将 target attack direction 投影到 aggregate other-label gradient 的正交
    子空间，以一阶近似保持 other-label loss；
  - 它是测试时 multi-label recognition，且还使用 consistent-target-set
    knowledge graph；
  - 对 detection coefficient-space、多 non-target class constraint 的迁移是
    本项目新假设，不是论文已证明结论。
- Zhang et al., *CD-UAP*, AAAI 2020：
  - class-discriminative universal perturbation 明确观察到 target attack 与
    non-target preservation 的 trade-off；
  - 它支持冲突诊断的必要性，不证明 training-time selective UE。
- Bao et al., *GLOW*, CVPR 2024：
  - 研究攻击计划的语义/几何 layout consistency，不是低频自然图像 carrier；
  - 不作为本 idea 的直接依据。
- 本地 PDF：
  - `D:/Googledownload/2021-CVPR-The-Translucent-Patch-A-Physical-and-Universal-Attack-on-Object-Detectors.pdf`
  - `D:/Googledownload/Mahmood_Semantic-Aware_Multi-Label_Adversarial_Attacks_CVPR_2024_paper.pdf`
  - `E:/实验/notebook/Towards_Effective_and_Robust_Unlearnable_Examples_Against_Object_Detection (1).pdf`
  - `E:/实验/notebook/lsp-2022-ACM KDD-Availability attacks create shortcuts.pdf`
  - `E:/实验/notebook/2023-ICLR-Transferable unlearnable examples.pdf`
  - `E:/实验/notebook/2603.03725v1.pdf`
  - `E:/实验/notebook/CD-UAP_Class_Discriminative_Universal_Adversarial_Perturbation.pdf`

## 3. 核心判断

### 3.1 背景频谱 carrier

可行，但原始表述需要三项修正：

1. 海洋、天空、道路等背景有真实场景语义，不能直接称为 semantic-free；
2. 仅保留低频可能过于平滑，容易被归一化/颜色增强吸收；已有 detection SPS
   证据反而偏向中频；
3. 即使 source pattern 是低频，乘以 person support、JND map、clamp 和 dropout
   后也会产生高频谱泄漏，最终频段必须在 materialized perturbation 上测量。

因此默认候选不是 raw background patch，而是：

```text
phase-scrambled background spectral basis
+ remove DC
+ fixed low/low+mid band
+ SVD orthogonalization
+ trainable low-dimensional coefficients
```

raw background phase 只作为 semantic-dependence control。

### 3.2 feature residual consistency

可行，且比 local-context absolute prototype 更贴合“共享 carrier”假设，但优化位置
应从共享 FPN 主目标改到 decoupled classification tower：

- primary：YOLO Detect `cv3` 的 pre-final-conv feature residual；
- secondary：P3/P4/P5 shared FPN residual；
- leakage：Detect `cv2` box tower residual 和真实 non-target foreground residual。

理由：只在共享 FPN 强制 residual，可能继续把 perturbation 压力扩散到定位分支和
其他类别；classification tower 更接近“保留前景、破坏/劫持 person 分类”的路径。

### 3.3 两种 target route 必须互斥

#### R+：shortcut-aligned error minimization

poisoned image 仍标注为 person。为了让 victim 更容易用 carrier 拟合 person，
R+ 在 clean TAL/PAG target positives 上降低正标签分类损失：

```text
L_easy_cls = mean softplus(-person_logit_adv)

L_box_teacher = mean smooth_l1(
    box_adv[target_pag_positive],
    stopgrad(box_clean[target_pag_positive])
)
```

`L_box_teacher` 只在 clean TAL/PAG 的 target positives 上计算，用来保持 person
前景与定位响应接近 clean teacher。该方向与 EM/SPS 的“poisoned training 更容易”
解释一致。

#### R-：evasion-aligned TAL suppression

YOLOv8 对 Translucent Patch 的最近似量是：

```text
A_person = sigmoid(person_logit_adv)^alpha * CIoU_adv^beta
L_evasion = mean A_person
```

它降低 target matching/confidence，属于即时 evasion proxy。它可以作为
adversarial-poison control，但不能预先称为易学 shortcut。

禁止把 `L_easy_cls` 和 `L_evasion` 直接相加；二者可能具有相反 target gradient。

### 3.4 non-target teacher 与梯度约束

clean teacher 只在 clean TAL 的真实 non-target foreground positives 上定义 one-sided
constraints：

```text
h_cls[k] = mean relu(
    p_clean_assigned[k] - p_adv_assigned[k] - tau_cls
)

h_box[k] = mean relu(
    ciou_clean_assigned[k] - ciou_adv_assigned[k] - tau_box
)
```

- `tau_cls=0.005`，沿用现有未激活 branch 的初始容忍度；
- `tau_box=0.02`，只用于 probe，查看结果后不得回调；
- 预测改善不处罚；
- 无该 class 的真实 foreground 时不构造梯度行；
- person-only batch 的 non-target constraint 标 `not_applicable`。

与对所有 non-target logits 做全量 MSE 相比，该约束避免：

- 大量背景 anchor 稀释；
- 非目标置信度改善也被拉回；
- 始终产生强 preserve gradient。

## 4. 背景频谱 basis

### 4.1 source bank

- 8 张由用户合法持有/明确许可、无 person 的自然背景图；
- 不来自 VOC train/val；
- 每张图固定两个 deterministic crops，共 16 个 source samples；
- repository source manifest 写：
  - stable source id；
  - SHA256；
  - 尺寸；
  - ownership/license note；
  - person-free 人工确认；
- 本地绝对路径仅写入不提交 Git 的 local-only mapping，不进入仓库 manifest；
- 原图不提交 Git；只提交派生 basis 的小型数值文件、manifest 和生成脚本。

### 4.2 basis construction

统一在 `640x640` reference grid 上：

1. RGB→luminance；
2. FFT；
3. remove DC；
4. 按 current 定义截取：
   - low：radius `[2,8)`；
   - mid：radius `[8,24)`；
5. 构造四个 carrier controls：

| ID | Carrier | 作用 |
|---|---|---|
| C0 | current synthetic band-aware Fourier | 当前 basis control |
| C1-L | raw-phase natural low-only | 用户原始 low-frequency 假设 |
| C2-L | phase-scrambled natural low-only | 去除可识别 layout 的 low-only control |
| C2-LM | phase-scrambled natural low+mid | 检测 mid-frequency 必要性 |

6. C1/C2 各自在相同 band energy 下做 SVD，保留 16 个空间 basis；
7. basis zero-mean、unit-L2、固定符号，保存 construction seed/hash；
8. 每个空间 basis 使用独立 RGB coefficient，coefficient dimension
   `d=16*3=48`，与 current active coefficient 维度匹配；
9. perturbation 仍受 `eps=16/255`、support、JND 和 clamp 约束。

必须在实际 support/JND/clamp 后重新计算 low/mid/high energy ratio，source
frequency label 不能替代 materialized spectrum。

## 5. Cross-instance Classification Residual（CICR）

对每个 scale `l`，在
`model.22.cv3.{0,1,2}.2` 的 forward pre-hook 捕获 pre-final-conv
classification feature。沿用 clean TAL/PAG target gate：

实现前必须断言三处 module path、输入 shape 和最终 class-conv 的连接关系与当前
YOLOv8 checkpoint 一致；断言失败时直接停止，不静默回退到其他层。

```text
r_cls[b,l]
  = masked_mean(
      F_cls_adv[b,l] - stopgrad(F_cls_clean[b,l]),
      M_target_pag[b,l]
    )

p_cls[l]
  = EMA_train(normalize(mean_b,eot(normalize(r_cls[b,l]))))

L_cicr
  = mean_b,l(1 - cosine(r_cls[b,l], stopgrad(p_cls[l])))
```

- prototype 按 P3/P4/P5 分层；
- 只由 optimization-train split 更新；
- held-out 只能读取冻结 prototype；
- 使用 residual energy floor，阈值固定为 common warmup train residual norm Q25
  的 `0.5x`；
- zero residual 不得靠 cosine epsilon 假装一致；
- 同时记录：
  - shared FPN residual；
  - `cv2` box-tower residual；
  - real non-target `cv3` residual；
  - person-only/person-cooccur；
  - small/medium/large；
  - carrier-swap/wrong-carrier specificity。

## 6. Gradient Routing feasibility

设 carrier coefficient 为 `theta in R^48`：

```text
g_t = grad_theta(L_cicr + lambda_route * L_route)
```

对当前 batch 中 active/near-boundary 的 non-target class constraints，逐类构造归一化
gradient row，形成 `G in R^(K_active x 48)`。使用 SVD，保留
`sigma/sigma_max >= 1e-4` 的 rank：

```text
P_null = I - V_r^T V_r
g_safe = P_null g_t
attack_retention = ||g_safe|| / (||g_t|| + eps)
```

三态更新策略：

1. 无 active constraint：使用 target gradient；
2. constraint 在容忍边界附近但未违反：只允许 projected target gradient；
3. constraint 已违反：repair-only，当前 step 不做 target update。

正式接入前必须用 backtracking 检查实际 nonlinear response；若 projected step 仍使
任一 active constraint 超容忍度，则缩小 step，最多 5 次，仍失败则 skip 并记录。

Phase A/B 只计算 feasibility metrics，不修改 optimizer；只有 Phase B PASS 后才允许
在独立 Phase C probe 中实际使用 routing。

## 7. 候选方案与最小判别顺序

| 方案 | Carrier | Route | Residual | Projection | 用途 |
|---|---|---|---|---|---|
| A0 | C0 synthetic | current ALCE | current FPN | off | current mechanism control |
| A1 | C0 synthetic | R+ | CICR | diagnostic only | 判断 residual/route 本身 |
| A2 | C2-L | R+ | CICR | diagnostic only | 判断 low-only 背景谱 |
| A3 | C2-LM | R+ | CICR | diagnostic only | 判断 low+mid 背景谱 |
| A4 | Phase A 最佳 carrier | R- | CICR | diagnostic only | evasion route control |
| A5 | Phase B 最佳组合 | selected | CICR | on | 仅在 routing feasibility PASS 后 |

C1-L raw-phase 不进入优化主臂，只做 frozen semantic-dependence control。

顺序：

1. Phase A：frozen carrier forward probe；
2. Phase B：A1/A2/A3/A4 short optimization probe；
3. Phase C：仅最佳组合做 routing-on short probe；
4. Phase D：需要再次批准的 fresh-victim seed0 gate。

## 8. Phase A：frozen carrier probe

- frozen surrogate；
- frozen carrier amplitude，matched `Linf`、support、JND；
- 不做 optimizer step；
- calibration/held-out split 沿用
  `TAUSB-ALCE-CTX-AUDIT-v1` 的 image ids/hash；
- primary identity view，deterministic EOT secondary；
- 比较 C0/C1-L/C2-L/C2-LM：
  - target `cv3` residual coherence；
  - non-target/target residual energy ratio；
  - `cv2` box leakage；
  - materialized band-energy ratio；
  - carrier source retrieval/correlation；
  - linear probe 对 clean/poisoned target crop 的可分性。

Phase A 只筛 carrier，不声称 UE。

## 9. Phase B：route 与 CICR short probe

- A1/A2/A3 从相同 zero coefficient、optimizer state、batch order、EOT 参数开始；
- common warmup 只建立非零 residual，不选择结果后最优切换点；
- R+ 与 R- 分开运行；
- 记录：
  - held-out CICR；
  - `L_easy_cls` / `L_evasion` 的 route-specific 变化；
  - `cos(grad L_cicr, grad L_route)`；
  - target/non-target gradient matrix rank；
  - attack retention；
  - constraint directional violation；
  - post-mask spectrum；
  - coefficient/basis usage；
- R+ 若不能形成 route effect，A4 仍可作为“即时 evasion 是否成立”的诊断对照，
  但不得据此推进易学 shortcut 路线；
- 背景谱 carrier 若不优于 C0，不进入 Phase C。

## 10. Phase C：routing-on short probe

仅对 Phase B 最佳组合：

- feature-off/control：同一 checkpoint fork；
- routing off/on；
- 相同 batch/EOT/step；
- 只在 coefficient space route；
- surrogate、FPN、head 参数全部 frozen；
- 输出每 step 的：
  - active class ids；
  - constraint rank/null dimension；
  - attack retention；
  - repair-only/projected/skip counts；
  - backtracking count；
  - actual post-step constraint change。

Phase C PASS 仍不构成 victim UE 证据。

## 11. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | 新 `background_spectral_basis.py` | source manifest、FFT/band/phase/SVD/basis hash | synthetic image tests；DC=0；band energy；同 seed bitwise 一致 |
| 2 | `tausb_universal.py` | basis mode 接入，默认 synthetic 不变 | feature-off 与 current pattern 数值一致 |
| 3 | 新 hook helper | 捕获 `cv3` pre-final feature 和 `cv2` monitor | 3 scales shape/gate 对齐；hook remove 测试 |
| 4 | `alce_losses.py` 或新 loss 文件 | CICR/prototype/energy floor | zero mask/residual、single sample、finite backward |
| 5 | `shadow_tal.py` | R+/R- route primitives | YOLOv8 无 objectness assertion；clean-gate frozen |
| 6 | 新 gradient router | multi-constraint SVD projection | analytic orthogonal/rank-zero/full-rank/near-singular tests |
| 7 | 新 probe/config | Phase A/B/C、split/hash、全量 diagnostics | 同 seed 重跑一致；无 train/held-out overlap |
| 8 | compile/config parse | 相关 Python/YAML | `py_compile`、config parse、CPU synthetic tests |

计划新增：

- `ue_project/ue_framework/methods/background_spectral_basis.py`；
- `ue_project/ue_framework/methods/constraint_gradient_router.py`；
- `ue_project/ue_framework/tools/probe_tausb_bsc_rc_gr.py`；
- `ue_project/ue_framework/configs/exp_voc_person_tausb_bsc_rc_gr_probe.yaml`；
- `ue_project/tests/test_background_spectral_basis.py`；
- `ue_project/tests/test_cicr_and_gradient_router.py`。

本地没有 VOC dataset，真实 surrogate/data forward 为远程 probe 前
`validation_gap`，synthetic tests 不得冒充机制结果。

## 12. canonical 参数链与回退

```text
probe config
→ carrier_basis_mode
→ background_spectral_basis / current Fourier basis
→ compose_delta_batched
→ clean TAL/PAG gate
→ cv3 pre-final residual
→ L_cicr
→ target_route: easy_cls | tal_evasion
→ non-target one-sided constraints
→ diagnostic/projected coefficient gradient
→ full diagnostics sink
```

回退开关：

```yaml
carrier_basis_mode: synthetic_fourier
enable_cicr: false
target_route: current_alce
enable_constraint_gradient_routing: false
```

全部回退时：

- current best config、loss、frequency selection、optimizer step 和
  materialization 必须数值一致；
- 只允许新增不参与计算的 diagnostics；
- 不修改当前 best YAML 默认值。

## 13. Research Contract

- **Hypothesis**：
  phase-scrambled、去 DC、低维的自然背景频谱 basis，在保留低/中频统计而减少
  可识别场景语义后，能够比 current synthetic Fourier basis 更稳定地穿过 person
  外观、尺度和背景变化，在 YOLOv8 classification tower 产生共享 residual。
  与 labeled poisoned person 一致的 R+ error-minimizing route 比即时
  TAL-evasion route 更符合易学 shortcut 机制。若 coefficient space 存在
  non-target constraint 的非平凡 nullspace，gradient routing 能在保留 target
  direction 的同时减少一阶 non-target violation。
- **Phase A Success Signal**：
  1. C2-L 或 C2-LM 相对 C0 的 held-out target `cv3` residual cosine median
     提高 `>=0.10`，且 Q25 `>0`；
  2. non-target/target residual-energy ratio 不高于 C0 的 `1.05x`；
  3. `cv2` box residual energy 不高于 C0 的 `1.05x`；
  4. intended band（C2-L 为 low；C2-LM 为 low+mid）在 materialized
     perturbation 中占比 `>=0.70`；
  5. core metrics finite，source/split/basis hash 完整。
- **Phase B Success Signal**：
  1. 最佳背景 carrier + R+ 相对 A1 的 held-out CICR median 再提高
     `>=0.05`；
  2. R+ 的 held-out positive person classification loss 相对 clean/fork
     下降 `>=10%`；
  3. person-only/person-cooccur CICR median gap `<=0.15`；
  4. small/medium/large 最大 CICR median gap `<=0.20`；
  5. active-constraint projection attack retention median `>=0.30`、Q25
     `>=0.10`；
  6. projected first-order violation ratio `<=0.02`。
- **Phase C Success Signal**：
  1. routing-on 相对 matched routing-off 的 held-out CICR 不低于 `0.95x`；
  2. actual post-step non-target violation rate 降低 `>=50%`；
  3. repair-only + skipped step 比例 `<0.50`；
  4. constraint null dimension median `>=8`；
  5. 所有 gradient/step finite，feature-off 回退通过。
- **Failure Signal**：
  1. C1-L 明显优于 C2-L（residual cosine gap `>0.10`），但 source
     retrieval/correlation 同时显著升高：收益依赖原背景 layout/语义，
     “semantic-free carrier”不成立；
  2. low-only 的 post-mask high-frequency energy `>0.30` 或
     target residual coherence `<0.10`：不能将该实现称为稳定 low-frequency
     shortcut；
  3. R- 能即时降低 TAL alignment，但 held-out CICR `<0.10` 或 R+ route
     effect 不成立：只支持 evasion proxy，不支持易学 shortcut；
  4. CICR 达标但 person-cooccur non-target/target residual ratio 高于
     person-only `1.5x`：机制可能依赖共现 collateral leakage；
  5. attack retention `<0.10` 的 batch 比例 `>=0.80` 或 null dimension
     median `<4`：当前 coefficient space 几乎不存在可行选择性方向；
  6. routing projection 数学测试通过，但 actual post-step violation 不降：
     一阶近似不足，不进入 victim。
- **Metric & Split**：
  - primary：
    held-out target `cv3` CICR、non-target residual leakage、route-specific
    target proxy、constraint violation；
  - secondary：
    shared FPN residual、`cv2` box residual、carrier specificity、linear
    probe、band energy；
  - groups：
    P3/P4/P5、small/medium/large、person-only/person-cooccur、identity/EOT；
  - Phase A/B/C source split：VOC train，surrogate-only；
  - Phase D 才使用 clean VOC validation mAP。
- **Stop Condition**：
  - NaN、Inf、OOM、Traceback、日志无进度；
  - source/basis/split hash 变化或 train/held-out overlap；
  - basis rank `<8` 或 intended-band construction 失败；
  - target residual zero-norm ratio `>0.25`；
  - active constraint rank/shape 非法或 SVD non-finite；
  - attack retention `<0.10` 的 batch 比例达到 `0.80`；
  - actual post-step violation 连续 20 step 不下降；
  - feature-off 不等价；
  - Phase A 未 PASS 时不运行 Phase B；
  - Phase B 未 PASS 时不运行 Phase C；
  - Phase C 未 PASS 时不生成 poisoned dataset。
- **Claim Boundary**：
  - Phase A 只能评价 carrier feature response；
  - Phase B 只能评价 route/residual mechanism；
  - Phase C 只能评价 gradient-routing feasibility；
  - raw natural background 不能声称 semantic-free；
  - phase scrambling 只能声称减少可识别 layout，不证明信息论意义上无语义；
  - Translucent Patch 只支持 test-time YOLO objectness/class confidence 的
    启发，不证明 YOLOv8 training poison；
  - mechanism PASS 不等于 fresh-victim UE；
  - seed0 victim 只能 tentative；
  - forced pseudo fallback support 不描述为 true instance mask；
  - 所有阈值不得在查看 held-out/victim 结果后回调。

## 14. Phase D：fresh-victim gate（本轮不自动批准）

仅当：

1. `TAUSB-ALCE-CTX-AUDIT-v1` 完成；
2. Phase A/B/C PASS；
3. P0 质量统计链修复；
4. pre-run implementation review 为 `pass`；
5. 用户再次批准昂贵运行；

才允许 matched seed0：

- D0：current TAUSB fresh baseline；
- D1：最佳 carrier + CICR + R+，routing off；
- D2：仅当 D1 target collapse 保留但 non-target 不达标时，运行 routing on；
- 相同 victim init、200 epochs、poisoned count、support、eps 和 clean VOC eval；
- `all` 后单独 `aggregate`；
- 独立 run root，不覆盖 current best。

Phase D tentative Success：

- `mAP50_target <=0.10`；
- `mAP50_non_target >=0.725`；
- `AP_person_cooccur_non_target >=0.570`；
- `AP_person_free_non_target >=0.584`；
- PSNR/LPIPS/poisoned_count finite/有效；
- 相对 matched D0：
  - `PSNR >= PSNR_D0 - 0.5 dB`；
  - `LPIPS <= LPIPS_D0 + 0.01`。

通过 seed0 后另立 seed1/seed2 audit，不在本 Spec 自动扩展。

## 15. 远程运行与产物

- Phase A/B/C 入口：
  `ue_project/ue_framework/tools/probe_tausb_bsc_rc_gr.py`；
- config：
  `ue_project/ue_framework/configs/exp_voc_person_tausb_bsc_rc_gr_probe.yaml`；
- Remote project root：
  `/root/autodl-tmp/ue_project`；
- 独立 run root：
  `/root/autodl-tmp/ue_project/runs_research/TAUSB-BSC-RC-GR-v1`；
- ExpID：
  `TAUSB-BSC-RC-GR-MECH-S0`；
- GPU：
  单 GPU surrogate-only；
- 预计时长：
  pre-run smoke 后填写；
- resume：
  fresh probe 不使用 `--force_resume`；
- rollback：
  关闭全部新 flags，不删除或覆盖历史 artifact。

结果落盘：

- Remote artifacts：
  `research_workspace/experiments/TAUSB-BSC-RC-GR-MECH-S0/remote_artifacts/`；
- Metrics summary：
  `research_workspace/experiments/TAUSB-BSC-RC-GR-MECH-S0/metrics-summary.json`；
- H→E→N：
  `research_workspace/experiments/TAUSB-BSC-RC-GR-MECH-S0/analysis/`；
- Experiment ledger：
  `research_workspace/00-实验记录.md`；
- STATE：
  mechanism probe 不更新 Current Best，结论是否进入机制叙述由用户决定。

## 16. Pre-run Review

- reviewed branch / commit：待实现并提交后冻结；
- exact command：待实现后冻结；
- source/basis/split hash：pending；
- parameter sink probe：pending；
- feature-off evidence：pending；
- optimizer scope：必须只有 carrier coefficient/prototype state；
- output non-overwrite check：必须确认独立 run root；
- result：`pending`。

## 17. Spec 自审

- 无未解释 placeholder；`pending` 只用于实现后才能冻结的
  branch/commit/command/hash/review；
- 明确区分了 shortcut R+ 与 evasion R-，未把相反目标混成一个 loss；
- 明确 YOLOv8 没有独立 objectness，使用 TAL alignment 仅作近似对照；
- raw background 与 phase-scrambled control 可以检验收益是否依赖场景 layout；
- low-only 与 low+mid control 可以检验用户的低频假设，而不事后改频段；
- residual primary sink 位于 classification tower，shared FPN/box tower 作为
  leakage monitor；
- non-target constraint 只用真实 foreground、one-sided、active class；
- projection 先诊断再接 optimizer，且有 rank/retention/nonlinear backtracking
  门禁；
- feature-off 回退和 current best 不覆盖路径明确；
- Phase A/B/C 与 fresh-victim Phase D 的声明边界分离；
- Success 与 Failure Signal 独立；
- P0 稳定性/质量工作未被新方法草案替代。
