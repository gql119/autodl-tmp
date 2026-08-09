---
spec_id: TAUSB-SIRC-MALC-CGR-MAP50-v2
title: SIRC-MALC-CGR 检测器适配特征集中与选择性不可学习验证
status: approved
approved: 2026-08-09
execution_state: active
experiment_type: method
csv: issues/TAUSB-SIRC-MALC-CGR-MAP50-v2.csv
created: 2026-08-09
supersedes: TAUSB-SIRC-LFC-CGR-MAP50-v1
---

# SIRC-MALC-CGR 检测器适配特征集中与选择性不可学习验证

## 1. 问题锚点

- STATE 关联：目标仍是在 VOC20 / YOLOv8n 上降低 `person` AP50，同时保持
  其他 19 类。SIRC 目前只有 surrogate mechanism/mechanical 证据，没有
  fresh-victim mAP50 证据。
- 触发证据：Semantic Deep Hiding 的 LFC 面向图像分类，它对扰动图
  `x_pm=x_ue-x_c` 的 ResNet-18 最后卷积层扁平特征做同类两两余弦距离，
  原文权重 `omega_3=1e-4`。该设计不包含目标实例、TAL assignment、P3/P4/P5
  尺度或非目标共现。
- 代码证据：现有 `YOLODetectTowerCapture` 已能获取 P3/P4/P5 的 `cv3`
  classification-tower pre-logit features；`instance_cicr.py` 已能使用 clean TAL/PAG
  将 clean-to-poison residual 按 person GT 实例池化。
- 本轮问题：将分类式 LFC 改造为检测器内部的多尺度、assignment-aware、
  instance-balanced 残差集中，是否能比只用 easy-classification route 产生更稳定的
  person shortcut，并在 CGR 后仍保留非平凡 target gradient？
- 非目标：不训练、不加载额外 ResNet-18；不对扰动图的全局 embedding
  做盲目聚类；不使用 ALCE、EOT、标量 non-target loss、late repair、robustness
  evaluation、multi-seed 或跨模型迁移。forced pseudo fallback 不表述为真实
  instance mask。

## 2. Idea Source

- 来源类型：文献机制改造 + 现有 SIRC/CICR 代码审计。
- 证据链：
  `D:/Googledownload/2406.17349v1 (1).pdf`；
  `docs/research/specs/TAUSB-SIRC-LFC-CGR-MAP50-v1.md`；
  `ue_project/ue_framework/methods/detector_tower_hooks.py`；
  `ue_project/ue_framework/methods/instance_cicr.py`；
  `ue_project/ue_framework/methods/shadow_tal.py`；
  `ue_project/ue_framework/methods/constraint_gradient_router.py`。
- 为什么现在做：尚未开始 v1 的方法实现和远程训练，现在修订可避免
  增加一个任务不对齐的 checkpoint 依赖与重复目标。

### 2.1 候选方案比较

| 方案 | 约束对象 | 优点 | 主要风险 | 决策 |
|---|---|---|---|---|
| A. 原始 ResNet-18 LFC | 扰动图的分类特征 | 与原文最接近 | 分类/检测任务错位；忽略实例和尺度；额外 checkpoint | 不作为主方法 |
| B. 将 `delta_vis` 单独输入冻结 YOLO | 载体在 YOLO 中的响应 | 与检测器同构 | `delta_vis` 是 OOD 灰底图；全局特征可被 support 外区域支配 | 只作备选诊断 |
| C. MALC：多尺度 assignment-aware residual concentration | 真实 person positive 上的 YOLO `cv3` clean-to-poison residual | 实例、尺度、检测任务与实际扰动效果对齐；复用现有代码 | 可与 route 或 CGR 冲突；需要防止零残差和单尺度支配 | **采用** |

## 3. 机制与接入

### 3.1 载体与冻结检测器

- 载体保持 v1 的 shared SIRC family：16 bases / 48 RGB coefficients，半径
  `[2,24]`，`eps=16/255`，instance-canonical warp，deterministic JND，forced pseudo
  fallback support。
- 只更新 carrier coefficients `theta`。surrogate YOLO 全部冻结，clean 与 poison
  分支共享同一组权重。
- 捕获点固定为 YOLO Detect head 的 P3/P4/P5 `cv3` 最后分类卷积之前；
  `cv2` box tower 只作泄漏监测。

### 3.2 MALC：Multi-scale Assignment-aware Latent Concentration

对 batch 中第 `i` 张图像、第 `j` 个 person GT、尺度 `l in {P3,P4,P5}`，
clean real TAL 提供 `target_gt_idx` 和 assigned target score，PAG 给出检测相关位置。
对同一实例内的 positive locations 用 clean assigned score 归一化加权：

```text
w(i,j,l,a) = PAG(i,j,l,a) * stopgrad(score_clean(i,l,a,person))
w_hat      = w / (sum_a w + eps)

r(i,j,l) = sum_a w_hat(i,j,l,a) *
           [Z_cls_l(x_p)[i,:,a] - stopgrad(Z_cls_l(x)[i,:,a])]
```

`r(i,j,l)` 不是原始 FPN 图或最终 logit，而是直接输入 person 分类卷积的
pre-logit latent residual。这使它既保留检测特征容量，又比 backbone 全局特征更接近
`person` 分类决策。

### 3.3 冻结原型、尺度平衡与非零门禁

- 在固定 calibration split 的 route warm-up 后，对每个尺度拟合：
  - 方向原型 `mu_l = normalize(mean normalize(r(i,j,l)))`；
  - 能量中心 `m_l = median RMS(r(i,j,l))`；
  - 能量下限 `tau_l = 0.5 * Q25(RMS(r(i,j,l)))`。
- `mu_l`、`m_l`、`tau_l` 在正式 carrier optimization 和 held-out 上全部冻结，
  不允许 held-out/victim 回写。
- 方向、幅值和低能量损失都先按实例平均，再对当前有效的 P3/P4/P5
  等权平均，避免 P3 的 anchor 数量或大 person 实例支配优化：

```text
L_dir_l   = mean_valid(i,j) [1 - cosine(r(i,j,l), stopgrad(mu_l))]
L_mag_l   = mean_valid(i,j) SmoothL1(
              log((RMS(r(i,j,l)) + eps) / (stopgrad(m_l) + eps)), 0)
L_floor_l = mean_assigned(i,j) relu(tau_l - RMS(r(i,j,l))) / (tau_l + eps)

L_MALC = mean_valid_scales(L_dir_l + L_mag_l + L_floor_l)
```

- `L_floor` 覆盖已分配但低能量的实例，不允许把它们从方向统计中删除后
  冒充高集中度。完全无 assignment 时 fail closed。
- 记录每尺度 valid count、direction cosine、log-energy MAD、floor pass ratio、
  scale contribution share、person size group 与 person-only/cooccur 分组。

### 3.4 与原 LFC / CICR 的边界

- MALC 保留原 LFC 的核心命题：同一个受保护类别应具有低方差、易学的
  共享 latent shortcut。
- MALC 改变了约束对象：从“扰动图的分类特征”改为“真实目标实例在
  检测分类塔中的 clean-to-poison residual”。
- MALC **取代** v1 的外部 `L_LFC` 和独立 `L_CICR`；两者不再重复加入
  目标函数。现有 `instance_cicr.py` 作为实现起点，但需扩展 score-weighting、
  scale-balanced reduction 和 magnitude/floor 门禁。

### 3.5 目标函数与机械梯度标定

```text
L_target = L_easy_cls
         + lambda_malc * L_MALC
         + lambda_rms  * L_rms
```

- `L_easy_cls` 使 poisoned person 在冻结 surrogate 上容易按原标签拟合，并使用 clean
  box teacher；它不是 test-time suppression。
- 不把原文 `omega_3=1e-4` 或 v1 的所有权重 `1.0` 直接照搬。在固定
  warm-up calibration batches 上计算 `L_easy_cls`、`L_MALC`、`L_rms` 对 `theta` 的
  median gradient norm，以 route gradient 为参考做一次性标定：

```text
lambda_k = clip(
    median ||grad_theta L_easy_cls|| /
    (median ||grad_theta L_k|| + eps),
    0.1, 10.0)
```

- 标定后所有 `lambda` 冻结；任一组件 gradient 断开、非有限或被 clip 到边界的
  比例超过 `0.50` 时停止，不用 victim mAP 事后调权。

### 3.6 非目标唯一保护：CGR

- 沿用 v1 已批准约束：clean TAL 真实 non-target foreground positives，逐类
  assigned-class probability drop，`tolerance=0.005`、`near_boundary=0.005`。
- 每类先求均值、梯度行 L2 归一化、SVD 相对阈值 `1e-4`；无 active 约束使用
  target gradient，near-boundary 使用零空间投影，已违反则 repair-only。
- 最多 5 次 nonlinear backtracking，仍不满足则 skip。不加标量保持损失、
  non-target feature loss 或 late repair。box/CIoU 只监测。

### 3.7 参数链与回退

- canonical 入口：`ue_framework/launch_one.py`。
- 新 method/config 将 MALC 参数绑定到：
  `config -> generate stage -> SIRC carrier optimizer -> YOLO cv3 capture -> TAL/PAG -> MALC -> CGR -> coefficient update -> materializer`。
- `enable_malc=false` 精确回退为 SIRC easy-route + RMS + CGR；`enable_cgr=false` 只用于
  本地/strategy mechanism control，不是正式 M1。新 method 全关时回退原 `tausb_mask`。

## 4. 最小判别实验

### 4.1 Mechanism gate（不是 fresh-victim 证据）

| Arm | Carrier/route | MALC | CGR | 用途 |
|---|---|---|---|---|
| A0 | shared SIRC + easy-cls | off | on | 检验没有特征集中时的基线 |
| A1 | 与 A0 完全相同 | on | on | 检验 MALC 是否改善 held-out shortcut signature |

- 复用固定 calibration/held-out split；原型只用 calibration 拟合，held-out 只读。
- A1 只有通过 mechanism gate 才允许进入 M1 fresh-victim。

### 4.2 Fresh-victim arms

- **C0**：原始 VOC train，从头训练 YOLOv8n victim。
- **M1**：用 A1 通过门禁后冻结的 carrier 扰动全部 `6,095` 张含 person 的
  train 图像，再从头训练独立 victim。
- 首轮不额外训练 A0 victim；因此 fresh-victim 结果证明整体 M1 是否有效，
  MALC 的单独因果贡献只由 mechanism gate 支持。

## 5. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | `semantic_residual_carrier.py` / renderer | 接入 shared carrier、deterministic JND 和 materializer | hash/rank/zero-mean/unit-L2/finite；outside-support=0；Linf |
| 2 | `instance_cicr.py` 或新 `malc.py` | 增加 score-weighted instance residual、scale-balanced direction/magnitude/floor | 单实例/多实例/无 assignment/低能量/尺度不平衡测试 |
| 3 | generator optimizer | 接入一次性 gradient calibration 和冻结 prototype bank | 同 seed 权重/原型 hash 一致；disconnected/nonfinite fail closed |
| 4 | `constraint_gradient_router.py` integration | 将 MALC composite target gradient 接入逐类 CGR | orthogonality/rank/full-rank/repair/backtracking/actual-constraint tests |
| 5 | mechanism A0/A1 | 实现固定 split 的 held-out comparison | 不更新 held-out；指标和门禁逐项落盘 |
| 6 | pipeline/config | 接入 generate/train/evaluate，禁用 EOT/robustness | CLI/config/runtime/sink 测试；feature-off exact regression |
| 7 | evaluation | 输出 VOC20 命名 AP50、19 类宏平均、delta/retention | synthetic mapping；缺类/NaN/order fail closed |
| 8 | Python 3.8 / AutoDL no-card review | import、compile、配置解析、真实 VOC 便宜 smoke | 不把 smoke 表述为 UE 效果 |

## 6. 数据、训练和指标

- Dataset：`/root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready`；train `16,551`，
  val `4,952`；数据只读。
- Target：`person` / class id `14`；poisoned train images 预期 `6,095`。
- Victim：YOLOv8n-style，from scratch，seed 0，epochs 200，imgsz 640，batch 36，SGD。
- Evaluation：原始 clean VOC val；不加 robustness transform。
- 必须落盘：20 类命名 AP50、`AP50_person`、19 类命名 AP50、
  `mAP50_non_target_macro`、`mAP50_all`、C0->M1 逐类 delta/retention、PSNR、LPIPS、
  poisoned_count、actual Linf，以及 MALC/CGR diagnostics。

## 7. 远程运行

- Branch：`codex/tausb-sirc-malc-cgr-map50-v2`（实施后绑定 commit）。
- Run root：`/root/tausb-sirc-runs/TAUSB-SIRC-MALC-CGR-MAP50-v2`。
- 顺序：A0/A1 mechanism gate -> C0 train/evaluate -> M1 generate/train/evaluate -> aggregate。
- 所有进程使用 tmux；只有进程、首个进度、finite loss、GPU process、status/log
  均存在才记为 `running_remote`。
- 使用独立 clean checkout 和 fresh roots；不清理 `/root/autodl-tmp` dirty worktree，
  不覆盖旧 artifacts，不默认 `force_resume`。

## 8. Research Contract（首次远程运行前冻结）

- **Hypothesis**：对 clean TAL/PAG person positives 的 YOLO classification-tower residual 做
  scale-balanced 方向、幅值和非零集中，会使 shared SIRC carrier 在不同 person
  外观、尺度与背景中形成稳定的检测捷径；CGR 能移除其中对真实非目标
  foreground 有害的一阶分量，同时保留非平凡 target direction。
- **Mechanism Success Signal**（A1 相对 A0，held-out）：
  1. level-balanced residual cosine median 提高 `>=0.10`，且 Q25 `>0`；
  2. log-energy MAD 不高于 A0 的 `0.90x`；
  3. valid-instance coverage `>=0.80`、zero-norm ratio `<=0.20`、floor-pass ratio `>=0.80`；
  4. 至少 2/3 个尺度在 `>=0.80` 的 held-out batches 中有有效实例；
  5. CGR `max_projected_row_dot<=1e-5`、attack-retention median `>=0.20`，
     `repair_only+skip<0.50`。
- **Fresh-victim Success Signal**（seed 0，tentative）：
  1. `AP50_person(C0)-AP50_person(M1)>=0.30`；
  2. `mAP50_non_target_macro(C0)-mAP50_non_target_macro(M1)<=0.05`；
  3. 19 类中至少 16 类 AP50 下降 `<=0.10`；
  4. `poisoned_count=6095`，所有关键量 finite。
- **Failure Signal**（独立定义）：
  1. MALC 任一组件与 `theta` 断开、gradient/SVD 非 finite，或梯度标定系数在
     `>0.50` 组件上命中 clip 边界；
  2. 超过 `0.80` 的集中损失来自单一尺度，或仅一个尺度长期有效；
  3. A1 的 non-target/target residual-energy ratio 或 box residual energy 超过 A0 `1.25x`；
  4. CGR 满秩/零空间退化使 attack-retention median `<0.20`，或
     `repair_only+skip>=0.50`；
  5. materialization support 外扰动非零、`Linf>16/255+1/255`、poisoned_count
     `<0.95*6095`；
  6. non-target macro AP50 下降 `>0.10`，或至少 5/19 类下降 `>0.15`。
- **Metric & Split**：primary 为 clean val `AP50_person down` 与
  `mAP50_non_target_macro up`；secondary 为逐类 AP50/delta/retention、mAP50_all 和 MALC/CGR
  diagnostics；quality 为 PSNR/LPIPS/Linf/poisoned_count。mechanism 使用固定
  calibration/held-out，victim 只用 clean VOC val。
- **Stop Condition**：A0/A1 机制门禁不通过时不进入 M1；此外 NaN/Inf、OOM、
  日志/GPU 无进度、input/hash 不一致、artifact root 已存在、类别 AP50 映射不完整
  或 feature-off 回退失败时停止。
- **Claim Boundary**：A0/A1 只是 mechanism evidence；只有 M1 fresh victim 可支持 UE
  效果；单 seed 只称 tentative；不声称 robustness、transferability、MALC 的独立
  fresh-victim 因果贡献或 SOTA。

## 9. Pre-run Review

- reviewed branch / commit：pending
- exact A0/A1/C0/M1 commands：pending
- dataset/label/source/surrogate hashes：pending
- CLI/config -> MALC/CGR/metric sinks：pending
- prototype/gradient-calibration split and hashes：pending
- feature-off / MALC-off / CGR-off / no-EOT regression：pending
- fresh output roots and recovery command：pending
- result：`pending`

## 10. 结果落盘（运行后填路径，不事后改判据）

- Mechanism A0/A1 artifacts：pending
- C0 artifacts：pending
- M1 artifacts：pending
- Per-class comparison：pending
- Metrics summary：pending
- H->E->N analysis：pending
- Experiment ledger：pending
- STATE update decision：pending
