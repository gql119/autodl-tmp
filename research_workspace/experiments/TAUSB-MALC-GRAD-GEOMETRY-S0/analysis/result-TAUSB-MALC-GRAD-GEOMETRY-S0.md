# TAUSB-MALC-GRAD-GEOMETRY-S0 H→E→N 分析

## H — 预注册假设

- **Hypothesis**：v2 中 MALC 无法使 A1 与 A0 分离的首要原因发生在 CGR 和 carrier 更新之前，即单一全局 prototype 不稳定、MALC 梯度跨 batch 不一致，或 MALC 与 easy-cls/RMS 存在系统性方向冲突；组合梯度 norm calibration 掩盖了该问题。
- **Success Signal**：16 个 calibration batch、24 个 held-out residual batch 和 8 步 matched microtrajectory 全部完整且 finite；`first_bad_boundary` 唯一落在 `prototype_incoherence`、`cross_batch_malc_conflict` 或 `objective_gradient_conflict`，且 median `rho_malc>=0.20`。
- **Failure Signal**：若 first boundary 是 `cgr_selective_suppression` / `carrier_update_sink` 则反驳“首要问题在 CGR 之前”；若为 `unresolved_by_probe` 则测量不足；若 primary metric 缺失、batch 不完整、非 finite、A0/A1 不匹配或只读语义被破坏，则 probe 无效，不能支持任何机制结论。
- **Metric & Split**：64-image calibration / 96-image held-out，primary 为 prototype resultant/LOO、跨 batch MALC cosine、MALC-vs-easy/RMS cosine、per-component CGR retention；secondary 为 8-step `D_theta`、`D_pattern` 与更新路由。
- **Stop Condition**：输入/hash 不一致、旧 root 已存在、NaN/Inf/OOM、图跨 batch 滞留、batch 不完整、旧 v2 回归失败或 10 分钟无进度时停止；无论结果如何都不进入 victim 训练。
- **Claim Boundary**：单 seed、surrogate-only 的 calibration/held-out 机制诊断；不支持 AP50、fresh-victim UE、鲁棒性、迁移性、最终因果贡献或 SOTA 声明。

## E — 证据

### 完整性与运行边界

- Artifact pull：10/10 required files，`missing_required=[]`、`failed=[]`；清单见 [transfer-report.json](../remote_artifacts/transfer-report.json)。
- 计算计数：16 calibration batches、24 held-out batches、8 matched steps，全部 JSON 数值有限。
- 状态：[status.json](../remote_artifacts/geometry/status.json)（SHA-256 `20f30a13b9038df247cb331b3b5401b2ac25759a292176878df45a605014d83c`）为 `stopped / diagnostic_decision / valid=false / first_bad_boundary=null`。
- 控制日志 [geometry-seed0.log](../remote_artifacts/control/geometry-seed0-18304b96-r1/geometry-seed0.log) 显示 probe exit 0、证据快照完成和 shutdown 请求；未见 NaN/Inf/OOM/Traceback。
- 无 `viz/` 且未 materialize 数据，故不作视觉、support localization、PSNR/LPIPS 声明。

### Primary validity gate

三个 calibration scale 的 coverage 分别为 `0.270677 / 0.323308 / 0.661654`，三个 held-out scale 为 `0.168478 / 0.396739 / 0.690217`，均低于预注册阈值 `0.80`。因此 effective scale count 为 `0`，effective resultant/LOO medians 为 `null`。冻结 decision artifact [diagnostic_decision.json](../remote_artifacts/geometry/diagnostic_decision.json)（SHA-256 `4b2e6b963b6ed6c7c5a7f0b57402102e1f26c3dcddfc07bf973bc09ea04b4943`）给出的 validity issues 是：

- `no_effective_prototype_scale`
- `missing_or_nonfinite_primary_metric`

这满足预注册 Failure Signal 的“probe 无效”分支；不能跳过 validity gate 指定 first boundary。

### 描述性几何信号（不可作为正式 boundary）

- Cross-batch MALC cosine：median `0.003894`，Q25 `-0.129668`。
- MALC-vs-easy median：`0.151405`；MALC-vs-RMS median：`-0.140263`。
- Per-component CGR retention median：easy `0.988075`、MALC `0.981178`、RMS `0.992081`。
- `D_theta=0.537088`，`D_pattern=0.000365778`。

若 primary validity 成立，cross-batch Q25 与 MALC-vs-RMS median 会越过冻结的冲突阈值；但当前只允许把它们记录成复核线索。高 MALC retention 不支持 CGR 选择性抑制。`D_theta` 已高于 carrier-sink 的 `<0.25` 条件，因此“参数分离、pattern 分离很小”只能描述为渲染压缩迹象，不能登记为预注册 carrier sink。

原始逐尺度、逐 batch、逐 component、逐步数据及哈希完整保存在 [prototype_geometry.json](../remote_artifacts/geometry/prototype_geometry.json)、[gradient_geometry.json](../remote_artifacts/geometry/gradient_geometry.json) 和 [microtrajectory.json](../remote_artifacts/geometry/microtrajectory.json)；汇总见 [diagnostic_summary.md](diagnostic_summary.md)。

## 判定

- **Research Contract：Failure Signal 命中（invalid probe）**。
- **假设状态：inconclusive，不支持也不反驳**。原因不是运行失败，而是 prototype primary metric 在冻结 coverage gate 下不可用。
- **first_bad_boundary：不存在（`null`）**。所有正式 trigger flags 为 false 是 validity-first 分类的结果，不代表所有候选机制均已排除。
- **Current Best：不变**。本轮没有 victim/AP50 证据，也不是可与 current best 比较的方法运行。
- **成本闭环**：GPU probe 已自动关机；产物在无卡模式拉回后，无卡实例的 `/usr/bin/shutdown` 返回成功，随后一次有界重连得到 `Connection refused`。无需再次开启实例。

## N — 唯一最小后续实验

新建一个 `prototype coverage measurement audit`，只做一项改变：为每个 person 实例和每个 P3/P4/P5 尺度记录残差 pooling 无效原因，并预注册一个与 TAL/尺度分配语义一致的 coverage 分母；随后按同一 64/96/8、同一 split、surrogate、MALC、CGR、carrier、loss 权重、LR 与 no-EOT 协议重跑 read-only probe。

在该测量产生至少一个有效 primary scale 前，不进入多原型、signed carrier、PCGrad、权重扫描或 victim 训练。这样下一次运行只回答一个问题：当前无效结论来自 prototype 本身覆盖不足，还是 coverage 统计口径与检测器分配语义不匹配。
