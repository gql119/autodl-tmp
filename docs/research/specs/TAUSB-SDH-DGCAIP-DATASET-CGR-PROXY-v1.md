---
spec_id: TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1
title: 数据集级 DG-CAIP、最终更新约束与代理轨迹校准
status: approved
parent_specs:
  - TAUSB-SDH-DGCAIP-CGR-E20-v2
  - TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1
created: 2026-08-31
approved: 2026-08-31
---

# 数据集级 DG-CAIP、最终更新约束与代理轨迹校准

## 1. 本轮范围

本轮只解决三个已确认问题：

1. DG-CAIP 由 batch 内 JS 排名改为训练集全局、类别内 KL/JS 排名；
2. CGR 约束最终实际更新，而不是只约束目标攻击分量后再叠加未约束保护梯度；
3. 用多阶段 clean-proxy 快照与短程 fresh-victim audit 校准代理保护位置和真实训练损害之间的偏差。

“victim 是否因果地学习了载体捷径”不属于本轮门禁，不增加 secret 恢复、错误 secret、宿主打乱或归因实验。旧 P4/R4 代码、配置、状态和实验 artifacts 保持只读可复现；新语义只能由本 SpecID 显式启用。

## 2. 数据集级 DG-CAIP 风险库

### 2.1 数据单元

只扫描训练集内同时含 person 与非目标 GT 的图像。每个有效实例使用稳定主键：

```text
(image_id, gt_index, class_id)
```

响应仍使用 clean real-TAL 对齐 anchors；clean 分支 detached，poison 分支使用同一 anchors，不允许 poison TAL 重选位置。记录：

```text
JS, clean_to_poison_KL,
assigned_probability_drop, IoU_drop, TAL_alignment_drop,
positive_count, geometry_risk, surrogate_snapshot_id
```

### 2.2 排名

排名只由 KL/JS 决定，其他损害量用于诊断和验证，不进入主排序：

```text
r_js(i) = percentile_rank_within_class(JS_i)
r_kl(i) = percentile_rank_within_class(log1p(KL_i))
risk(i) = 0.7 * r_js(i) + 0.3 * r_kl(i)
```

- tie 使用稳定 mid-rank；排序键追加稳定实例主键，保证跨进程确定性；
- 每类独立排名，禁止高频类别淹没低频类别；
- 每个代理快照先独立排名，多快照风险取 `max`，另报告 mean 与 rank variance；
- dataset bank 冻结一个 carrier optimization cycle，禁止每个 step 重排；
- bank 不完整、主键重复、哈希不匹配或某类覆盖不足时 fail closed。

### 2.3 重点处理

- 每类 top 25% 为 high-risk queue，每个有覆盖类别至少保留 1 个实例；
- 图像风险取其非目标实例风险最大值；
- optimization batch 固定为 50% uniform cooccurrence + 50% high-risk replay；
- 实例权重使用 `h=1+2*risk^2`，与 geometry risk 相乘后在当前有效实例内有界均值归一化到 `[0.5, 2.0]`；
- 不复制物化数据集，不利用 val/test AP50 排名，不硬编码 bicycle、bottle、dog、horse。

产物：

```text
dgcaip_risk_records.jsonl
dgcaip_risk_bank.json
dgcaip_risk_manifest.json
dgcaip_replay_manifest.json
```

全部包含 schema、SpecID、数据/模型/carrier hashes、生成时间以外的 canonical SHA256 和类别覆盖统计。

## 3. 最终整体更新约束

令 `d` 为优化器实际采用的扁平更新梯度，参数更新为 `omega_new = omega - eta*d`。

### 3.1 安全约束

未超过 tolerance、但处于 active/near-boundary 的非目标约束构成行矩阵 `G_safe`：

```text
G_safe * d = 0
```

目标梯度先投影到该零空间。任何显式保护或修复分量也必须留在该零空间内，不能在投影后叠加未投影梯度。

### 3.2 已违反约束

对 pre-update 已超过 tolerance 的约束，实际更新必须满足一阶非恶化并产生最小修复：

```text
g_j^T d >= repair_floor_j
```

因为更新采用 `-eta*d`，正内积表示对应损失下降。使用稳定的循环半空间投影，在 `null(G_safe)` 内寻找距离投影目标梯度最近的可行方向；有限次迭代仍不可行、null dimension 为 0、更新范数超过预算或数值非有限时 skip。

### 3.3 非线性验证

最多五次回溯验证完整渲染、裁剪和检测前向：

- safe：candidate 不超过固定 tolerance/baseline；
- violated：所有值不高于 pre-update baseline，且至少一个严格改善；
- 任一条件不满足则缩步；五次后仍失败则恢复原参数。

核心审计量改为：

```text
max_safe_final_row_dot <= 1e-5
min_violated_final_row_dot >= repair_floor - 1e-6
```

旧 `protection_ratio=0.25` 不再表示额外相加梯度，只作为最大修复范数预算；旧 P4 的 route 函数和历史配置不得改变。

## 4. 代理轨迹与 short-victim 校准

### 4.1 多快照来源

先训练 matched clean C0 E20，并保存 epoch `1/5/20` 快照；该训练本来就是 paired experiment 的组成部分，不另起完整代理训练。三个快照共同构成保护代理轨迹：

- stable TAL anchors：至少两个快照出现；
- volatile anchors：仅一个快照出现，记录 assignment churn 并提高审查优先级；
- 风险取三个快照的 class-wise rank 最大值；
- 约束行来自三个快照，重复/共线行由 SVD 去重。

目标攻击目标仍由冻结主 surrogate 计算，避免把三个快照同时用于全部目标损失而使成本失控。

### 4.2 fresh-victim audit

在完整 M1 E20 前，使用独立 seed 的 fresh YOLOv8n，在固定训练子集上训练 3 epochs：

- 只使用 train split，不读取 clean val AP50 参与排序或调参；
- 计算同一稳定实例主键的 victim KL/JS、probability/IoU/alignment damage 和 TAL churn；
- 比较 proxy risk 与 short-victim risk 的 Spearman、每类 top-25% overlap 和宏平均 top-25% overlap；
- 只允许单调校准或取 proxy/victim 风险最大值，不拟合高容量校准器。

进入完整 M1 E20 的门禁：

```text
macro Spearman >= 0.40
macro top-25% overlap >= 0.50
finite matched-key coverage >= 0.90
```

低于门禁时停止 M1，不在 GPU 上修改方法；保留结果并回到 risk/anchor 定义审查。

## 5. 分阶段执行

### L0：本地无卡基础实现

- dataset risk bank 的稳定主键、类别内 mid-rank、KL/JS composite、top-q 和 canonical hash；
- dataset rank 注入 DG-CAIP，feature-off 精确保持历史 batch ranking；
- strict final constrained route 与 mixed nonlinear backtracking；
- proxy/victim 风险一致性指标和 fail-closed gate；
- focused tests、compile 和旧 P4 回归。

### G0：dataset risk scan

- 使用冻结 carrier 与三个 C0 快照扫描完整 person-cooccurrence train subset；
- GPU hard cap 60 分钟；任何终态自动关机；
- 只生成 risk artifacts，不优化 carrier、不物化 poison、不训练 M1。

### G1：strict mechanism

- 只运行一个 dataset-ranked + strict-CGR arm；
- 8 optimization steps，20 分钟硬上限，所有终态自动关机；
- 通过最终约束、finite、coverage 和 target-retention 门禁才冻结 candidate state。

### G2：short-victim audit

- 独立 seed、固定训练子集、3 epochs；
- 45 分钟硬上限，所有终态自动关机；
- 通过 proxy-victim agreement gate 才允许完整 M1 E20。

### G3：paired E20

- C0 与 M1 均 fresh YOLOv8n，matched config/init/data/eval；
- 保留 person、19 类逐类 AP50、宏平均、下降量、保持率和 cooccur/person-free split；
- 无论成功、失败或 inconclusive 都保留；GPU 总硬上限 9 小时并自动关机。

## 6. 判据

### 机制门禁

1. bank 主键唯一、canonical hash 可复算、每类排名确定性完全一致；
2. high-risk replay 每类 coverage 达到配置要求，uniform/high-risk 比例精确；
3. 最终而非仅 projected-target 满足 safe row dot；violated row 满足 repair floor；
4. nonlinear trace 对 safe/violated 使用正确的两类接受条件；
5. target attack retention median `>=0.60`，null dimension `>0`；
6. backtrack+skip ratio `<0.70`；所有梯度、SVD、候选和风险值 finite。

### E20 判据

沿用父 Spec 的输出口径，不因方法修订改变评价协议：

- person AP50 drop 至少 `0.30`；
- 非目标宏平均 drop 不超过 `0.05`；
- 至少 `16/19` 非目标类 drop 不超过 `0.10`。

单 seed 只形成 tentative 结论。

## 7. Stop 与 Claim Boundary

- L0 未通过不得启动 GPU；G0/G1/G2 任一失败不得进入下一阶段；
- NaN/Inf/OOM/Traceback、无有效进度或硬上限触发时保留证据并关机；
- 任何 GPU bug 诊断超过 20 分钟立即停止并关机；
- dataset ranking 通过只证明能稳定定位代理/short-victim 高风险实例，不等于 AP50 改善；
- strict route 通过只证明当前候选更新满足局部一阶与非线性约束；
- short-victim gate 通过只说明训练早期风险排序具有转移一致性；
- 本轮不声明 victim 已学习捷径，不声明多 seed、跨架构或鲁棒性。
