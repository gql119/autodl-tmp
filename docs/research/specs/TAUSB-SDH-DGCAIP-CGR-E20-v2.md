---
spec_id: TAUSB-SDH-DGCAIP-CGR-E20-v2
title: 分布漂移引导的共现实例保护与约束梯度路由消融
status: approved
experiment_type: ablation
supersedes_draft: TAUSB-SDH-CAIP-CGR-E20-v1
parent_specs:
  - TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3
  - TAUSB-SDH-E2E-V0-SPARSE-E200-v1
created: 2026-08-17
approved: 2026-08-17
---

# 分布漂移引导的共现实例保护与约束梯度路由消融

## 1. 问题锚点

- 当前 E200 单 seed 结果中，person AP50 drop 为 `0.616561`，说明目标捷径持续；但非目标宏平均 drop 为 `0.068731`，未达到 `<=0.05`。
- person-free 非目标 drop 仅 `0.027430`，person-cooccur 非目标 drop 为 `0.131628`；保护缺口主要集中在与 person 共现的实例。
- current NLA 只在 clean real-TAL positives 上对齐 assigned-class raw logit，不能定位同一图像中哪个非目标 GT 的完整类别响应、bbox 或 TAL alignment 受损最重。
- 本轮只修改非目标保护模块；固定 secret、person bbox support、hiding state、D-LFC、CICR、目标攻击、数据、扰动预算、surrogate/victim 与 clean AP50 协议。

## 2. Idea Source 与方向修正

Idea source：*Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior*（arXiv:2502.21048v2）的 sample reweighting。

原论文对 transformed clean/adversarial 分类分布计算

```text
D_KL(P || Q)
w_attack = 1 / D_KL(P || Q)
```

其目的，是给 **KL 小、尚未被有效攻击** 的 hard-to-fool semantic samples 更大攻击权重。

本研究方向相反：需要优先修复 **clean/poison 非目标响应差异大** 的实例。因此：

```text
低相似度 = 高分布散度 = 更高保护优先级
```

不得照搬 `1/KL`。本模块采用随散度单调增加、但有界且停止梯度的保护权重。

## 3. 方案比较

| 方案 | 优点 | 风险 | 决策 |
|---|---|---|---|
| 整图所有 anchors 直接算 KL | 实现最简单 | anchor/assignment 不对齐，背景数量淹没真实 GT，person 的预期变化污染分数 | 不采用 |
| 每张共现图像一个 KL 排名 | 能筛图 | 多实例混合，无法定位受损对象，大目标/多目标图像偏置 | 仅作诊断，不作优化权重 |
| clean TAL 对齐的非目标 GT 实例级散度 | 同一物理实例、同一 anchors、可与 bbox/TAL 保护结合 | 需要 GT-instance 聚合和数值稳定处理 | **采用** |

模块名称：**DG-CAIP — Divergence-Guided Co-occurrence-Aware Instance Preservation**。

## 4. 机制定义

### 4.1 合法响应单元

只处理同时含 person 与非目标 GT 的训练图像。对非目标 GT 实例 `k`，使用冻结 surrogate 的 clean real-TAL：

```text
A_ik = {a | clean_fg_ia = 1,
              clean_target_gt_idx_ia = k,
              class(k) != person}
```

- clean/poison 两分支都在同一个 `A_ik` 上比较；不得用 poison TAL 重新选点。
- clean logits/boxes 是 detached teacher。
- 散度计算排除 person 维度，避免把预期的目标攻击当作非目标损伤。
- 无有效 `A_ik` 的实例不参加排名和损失，并记录 coverage。

### 4.2 检测头适配的分布散度

YOLOv8 分类头是独立 sigmoid，而不是单标签 softmax。默认不强制把 19 类归一化为互斥分布；在每个 clean positive anchor 上计算 19 个 Bernoulli 输出的平均 Jensen-Shannon divergence：

```text
p_ac = sigmoid(z_clean[a,c] / T)
q_ac = sigmoid(z_poison[a,c] / T)
m_ac = (p_ac + q_ac) / 2

d_ik = mean_(a in A_ik, c != person) [
    0.5 * KL(Ber(p_ac) || Ber(m_ac))
  + 0.5 * KL(Ber(q_ac) || Ber(m_ac))
]
```

- 固定 `T=2.0`；概率 clamp 到 `[1e-6, 1-1e-6]`。
- JS 由 KL 构成、对称且有界，更适合作为 clean/poison 无方向相似度；同时报告 one-way Bernoulli KL 作为诊断，不用于主权重。
- `d_ik` 的可微版本构成分布对齐损失 `L_dist,ik=d_ik`；用于排序的副本必须 `detach()`，防止优化器通过操纵权重而非修复响应来降低目标。

### 4.3 高散度难例权重

在每个 batch 的全部有效共现非目标实例上，对 detached `d_ik` 求 percentile rank `r_ik in [0,1]`：

```text
h_ik = 1 + 2 * r_ik^2
raw_w_ik = geometric_risk_ik * h_ik
w_ik = mean-normalize(clip(mean-normalize(raw_w_ik), 0.5, 2.0))
```

- `geometric_risk_ik` 沿用 CAIP v1 的 person-support overlap/distance 风险；feature-off 时为 1。
- 有效实例少于 4 个时不做 batch 排名，令 `h_ik=1`，避免小样本伪排序。
- 权重只在有效实例集合上均值归一化，因此重分配保护预算，而不增加 batch 总保护强度。
- 不按 `bicycle/bottle/dog/horse` 等类别硬编码；逐类结果仅用于事后验证。

### 4.4 被加权的保护目标

每个实例的保护损失为：

```text
L_DGCAIP,ik = w_ik * (
    L_cls_drop,ik
  + lambda_box  * L_box_drop,ik
  + lambda_align * L_TAL_align_drop,ik
  + lambda_dist * L_dist,ik
)
```

- `L_cls_drop`、`L_box_drop`、`L_TAL_align_drop` 沿用 CAIP v1 的实例平衡定义与 tolerance：`0.005 / 0.02 / 0.05`。
- 四项 lambda 只在固定 warm-up calibration batches 上按对 residual adapter `omega` 的 median gradient norm 一次标定并冻结；held-out 与 AP50 不参与。
- 实例先聚合为 class loss，再对 active classes 宏平均，防止大目标、多 positive 与高频类别主导。

### 4.5 NLA、CGR 与总预算

```text
L_protect,c = L_NLA,c + L_DGCAIP,c
g_atk = P_null({grad L_protect,c}) * grad L_target
g_total = g_atk + g_protect_budgeted
```

- current NLA 保留，不简单增大全局权重。
- CGR row space 使用每个 active class 的 combined protection gradient。
- 显式 NLA+DG-CAIP 总保护梯度仍固定为 `0.25 * ||g_atk||`；不得在原 NLA 预算外叠加额外预算。
- 最多 5 次 nonlinear backtracking，同时检查 assigned probability、IoU、TAL alignment 与 JS divergence；前三项沿用 `0.005 / 0.02 / 0.05`，逐类 JS 不得高于该步 pre-update baseline（数值容差 `1e-9`），不另设可调阈值。失败则缩步，五次仍失败则 skip。
- `enable_dgcaip=false` 必须精确回退 current NLA+CGR；不得回退旧 `tausb_mask`、ALCE/PAG 或 late repair。

## 5. 最小判别实验

### 5.1 D0：先验证“能否快速找到受损实例”

对 frozen current P1 的固定 held-out 共现样本只读计算 `d_ik`，不更新参数：

1. 报告实例级 `d_ik` 与 positive assigned-probability drop、IoU drop、relative TAL-alignment drop 的 Spearman correlation；
2. 将初始实例按 `d_ik` 固定分为 Q1/Q2/Q3/Q4，报告每组上述三项损伤；
3. 报告每个非目标类的实例数、coverage、mean/median/P90 divergence，不以最终 AP50 反向调权。

D0 通过条件：

- `d_ik` 与三项损伤的等权 z-score composite Spearman `>=0.35`；
- Q4 composite damage 至少为 Q1 的 `1.5x`；
- finite coverage `>=95%`。

若 D0 不通过，说明 KL/JS 不能可靠定位当前 protection gap，本 Spec 停在诊断，不实现动态加权。

### 5.2 Mechanism arms

所有 arms 使用同一 frozen hiding state、adapter 初值、batch 顺序和 target loss：

| Arm | current NLA+CGR | CAIP cls/box/TAL | uniform `L_dist` | high-divergence weight | 用途 |
|---|---:|---:|---:|---:|---|
| P1-R | on | off | off | off | 排除代码回归 |
| P2-CAIP | on | on | off | off | 结构保护基线 |
| P3-DIST | on | on | on | off | 判别分布对齐本身 |
| P4-DGCAIP | on | on | on | on | 判别散度排序/重分配增益 |

沿用 `16` calibration batches、`24` held-out batches、`8` optimization steps；GPU hard cap `20` 分钟，任一终态自动关机。只有 P4 通过 mechanism gate 才冻结新 state。

### 5.3 Victim E20

- 只训练通过 gate 的 P4；不为四个 mechanism arms 各训 victim。
- matched C0 仅在 victim config、seed、fresh-init tensor hash、manifest 与评估代码 hash 完全一致时复用，否则重训。
- M2 仍只物化 6,095 张含 person 的 poisoned PNG，其余 10,456 张引用原始 JPEG；fresh YOLOv8n seed0、20 epochs、clean VOC val。
- E20 通过后另拟 E200 Spec；本轮不直接运行 E200。

## 6. 实施与本地验证

| Step | 原子改动 | 必需证据 |
|---|---|---|
| 1 | 新增实例级 Bernoulli-JS、one-way KL 诊断和 ranking | identical logits=0、对称性、有界、clamp、person 维排除、finite backward |
| 2 | clean TAL `target_gt_idx` 到 GT instance 聚合 | 同一 positives、无 poison re-assignment、大小实例平衡、invalid GT fail closed |
| 3 | CAIP + `L_dist` + detached hardness | 权重单调、少于 4 实例回退、均值 1、范围、无 weight gradient |
| 4 | combined per-class CGR 与 fixed budget | row-dot、null dimension、总保护 norm ratio、feature-off exact fallback |
| 5 | D0/P1-R/P2/P3/P4 runner 与诊断 | shared init/batch/hash、held-out read-only、arm switches |
| 6 | sparse materializer 与 E20 compare | 6095/support/Linf、VOC20 per-class、person-free/cooccur split |
| 7 | pre-run review | exact branch+commit+commands+data/state/root/cost/shutdown，结论 pass |

## 7. Research Contract

### Hypothesis

在 person 共现图像上，clean real-TAL 对齐的非目标实例级分布散度能够识别当前扰动造成的高风险非目标响应；在固定总保护预算下，将 NLA/CAIP 保护重分配到高散度实例，可降低共现非目标 AP50 损失，而不显著削弱 person shortcut。

### Mechanism Success Signal

全部满足：

1. D0 locator gate 通过；
2. 在按初始 `d_ik` 固定的 Q4 实例上，P4 相对 P2 的 JS、probability、IoU、alignment positive damage 至少三项改善 `>=20%`；
3. P4 相对 P3 的 Q4 composite damage 改善 `>=10%`，证明排序有独立增益；
4. far/Q1 组任一结构指标相对 P2 恶化不超过 `10%`；
5. target attack retention median `>=0.70`，且相对 P1-R 降幅 `<=0.05`；
6. CICR cosine median 相对 P1-R 降幅 `<=0.02`，`D_pattern >= 0.80 * P1-R`；
7. `max dot(g_atk,g_protect,c) <= 1e-5`、null dimension `>0`、显式保护 norm ratio 位于 `[0.20,0.30]`；
8. backtrack+skip ratio `<0.50`，所有梯度、SVD 与候选 finite；P1-R 对历史
   `arms.P1` 的冻结标量、逐类 probability drop、逐步路由结构与数值无回归。
   数值比较固定采用 `abs_tol=1e-6`、`rel_tol=1e-4`；历史指标文件与
   `p1_state.pt` 均须 SHA256 绑定，任一结构变化或超差即阻断 P4 state。

### E20 Success Signal

全部满足：

1. person AP50 drop `>=0.55`，且相对历史 P1 target drop `0.643759` 减少不超过 `0.08`；
2. non-target macro drop `<=0.07`，且相对历史 P1 E20 `0.095900` 改善 `>=0.025`；
3. person-cooccur non-target drop `<=0.09`，且相对历史 P1 E20 `0.132122` 改善 `>=0.04`；
4. person-free non-target drop 不超过历史 P1 的 `0.047899 + 0.01`；
5. 至少 `16/19` 非目标类 drop `<=0.10`，无新类别 drop `>0.15`；
6. poisoned count、Linf、support、PSNR、LPIPS、fresh init、manifest 与 clean-val 门禁通过。

### Failure Signal

任一成立即不推进 E200：

1. D0 correlation `<0.20` 或 Q4/Q1 composite ratio `<1.2`；
2. P4 target attack retention `<0.50`、null dimension 为 0 或 backtrack+skip `>=0.70`；
3. P4 相对 P3 的 Q4 composite damage 没有改善，说明动态排序没有额外价值；
4. E20 person AP50 drop `<0.40`；
5. E20 cooccur drop 相对历史 P1 改善 `<0.015`；
6. 至少 5 个非目标类相对历史 P1 额外下降 `>0.05`；
7. support/Linf/6095/labels/manifest/fresh-init 任一不匹配或运行不是 fresh E20。

Success 与 Failure 均未触发时记为 `inconclusive_divergence_protection_tradeoff`。

### Metric & Split

- D0/mechanism：固定 calibration/held-out person-cooccur images，按 GT instance、class、Q1-Q4、overlap/near/far 报告 JS/KL/probability/IoU/alignment；held-out 不参与标定。
- primary victim：clean VOC val person AP50 drop、19 类 non-target macro drop、person-cooccur non-target drop。
- secondary：person-free、20 类逐类 AP50/drop/retention、target retention、CICR、D-pattern、SVD rank/null、backtrack/skip。
- quality：poisoned count、Linf、PSNR、LPIPS、support area；无 EOT/JPEG/blur/gray。

### Stop Condition

- Spec 未批准、CSV/实现/本地验证/pre-run review 未完成：不启动 GPU。
- D0 未通过：停止动态加权，不启动四臂 mechanism。
- NaN/Inf/OOM/Traceback、10 分钟无有效进度或 mechanism 20 分钟 hard cap：保留证据并关机。
- mechanism gate 未通过：不物化新 state、不训练 victim。
- E20 hard cap `1.5` 小时；异常诊断超过 20 分钟或任一 fatal signal：保存证据并关机。
- 无论 Success/Failure/Inconclusive，均保留并报告结果。

### Claim Boundary

- D0 通过只说明 divergence 是 surrogate 响应损伤的有效排序信号，不等于 AP50 改善。
- mechanism PASS 只说明 frozen surrogate 上保护改善，不能替代 fresh-victim AP50。
- 单 seed E20 只能形成 tentative 消融证据；不声称 E200、多 seed、跨架构或鲁棒性成立。
- 若 P4 E20 改善，只归因于完整 DG-CAIP；P2/P3/P4 mechanism 对比支持短程组件解释，但没有各自 fresh victim 时不声明独立 AP50 因果贡献。
- 不把历史四个高损类别硬编码为方法先验。

## 8. Pre-run Review 与结果落盘

- reviewed branch / commit：`pending`
- exact D0/P1-R/P2/P3/P4/E20 commands：`pending`
- CLI/config → TAL/JS/ranking/CAIP/NLA/CGR/backtracking/state/materializer/metric sink：`pending`
- feature-off exact fallback、fixed-total-budget、data/state/checkpoint/hash、fresh roots、成本和自动关机：`pending`
- review result：`pending`
- mechanism artifacts / frozen state / E20 artifacts / H→E→N / ledger：`pending`
- 未经用户批准，不更新 `STATE.md` Current Best。
