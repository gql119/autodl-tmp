# TAUSB-MALC-GRAD-GEOMETRY-S0 机制诊断摘要

## 结论

本次短 GPU probe 完成了预注册的 `16` 个 calibration batch、`24` 个 held-out batch 和 `8` 步 A0/A1 匹配微轨迹，10 个最小证据文件均已拉回并通过 SHA-256 校验，数值字段中未发现 NaN/Inf。但结果必须判为 **invalid / inconclusive**：三个尺度的 prototype coverage 均未达到冻结阈值 `0.80`，所以有效尺度数为 `0`，两个 prototype primary median 为 `null`。冻结分类器因此正确输出 `valid=false`、`first_bad_boundary=null`，不能事后跳过门禁指定一个机制瓶颈。

这不是运行崩溃，也不是 MALC 有效或无效的 fresh-victim 证据。本轮没有训练 victim、没有 materialize 投毒数据、没有 AP50、PSNR 或 LPIPS。

## 协议与完整性

- Spec / Exp / Run：`TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1` / `TAUSB-MALC-GRAD-GEOMETRY-S0` / `geometry-seed0-r1`
- Branch / reviewed code：`codex/tausb-malc-grad-geometry-audit-v1` / `18304b96c45360cfba5168d97d21d2961a13f390`
- 协议：64 calibration images（16 batches）、96 held-out images（24 batches）、batch 4、warm-up 4、microtrajectory 8、no EOT。
- 拉回：10/10 required files；`missing_required=[]`、`failed=[]`、`credentials_persisted=false`。
- 运行：probe exit 0，cost guard 随后请求关闭实例；最终状态为 `stopped / diagnostic_decision`。
- 原型只读：before/after hash 都是 `c7d9d6512dc8542351fa02da3b6ee93bbf124866f39a20ecb6154670990ba0b2`。

完整清单见 [transfer-report.json](../remote_artifacts/transfer-report.json)，机器可读摘要见 [diagnostic_summary.json](diagnostic_summary.json)。

## Prototype geometry

| Split | Scale | Valid / Total | Coverage | Resultant | LOO Q25 | Held-out reference cosine |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 0 | 36 / 133 | 0.270677 | 0.219525 | 0.963757 | — |
| calibration | 1 | 43 / 133 | 0.323308 | 0.250375 | 0.982473 | — |
| calibration | 2 | 88 / 133 | 0.661654 | 0.208774 | 0.991098 | — |
| held-out | 0 | 31 / 184 | 0.168478 | 0.348714 | 0.994414 | 0.403915 |
| held-out | 1 | 73 / 184 | 0.396739 | 0.251836 | 0.994972 | 0.619097 |
| held-out | 2 | 127 / 184 | 0.690217 | 0.218182 | 0.996070 | 0.907621 |

每个已获得方向的子集内部 LOO 稳定性较高，但 coverage 只有 `0.168–0.690`；因此不能将这些 resultant 直接包装成全体 person 实例的稳定原型。原始逐尺度及逐 batch LOO 数组保存在 [prototype_geometry.json](../remote_artifacts/geometry/prototype_geometry.json)（SHA-256 `d09f9419…bf331`）。

## Gradient geometry（描述性证据）

| Statistic | Value | 冻结阈值关系 |
|---|---:|---|
| cross-batch MALC median | 0.003894 | 高于 `<=0` 触发线 |
| cross-batch MALC Q25 | -0.129668 | 低于 `<-0.10` 触发线 |
| MALC vs easy median | 0.151405 | 不构成冲突 |
| MALC vs RMS median | -0.140263 | 低于 `<-0.10` 触发线 |

若 validity gate 已通过，后两项中的 Q25 与 MALC–RMS 信号会达到预注册冲突阈值；但本轮门禁未通过，所以它们只能作为后续测量修复后的复核线索，不能登记为 `cross_batch_malc_conflict` 或 `objective_gradient_conflict`。

| Component | Raw norm median | Projected norm median | CGR retention median | Q25 |
|---|---:|---:|---:|---:|
| easy-cls | 0.214339 | 0.210406 | 0.988075 | 0.981031 |
| MALC | 0.235889 | 0.224444 | 0.981178 | 0.958943 |
| RMS | 0.018861 | 0.018712 | 0.992081 | 0.971301 |

MALC 的 median retention 约 `0.981`，没有 CGR 选择性压掉 MALC 的证据。16 个 batch、120 个跨 batch pair、逐 component 原始/投影 48 维梯度和 row-space 统计保存在 [gradient_geometry.json](../remote_artifacts/geometry/gradient_geometry.json)（SHA-256 `e3be9ce1…3fba`）。

## 8-step matched microtrajectory

- `D_theta=0.537088`：A1 与 A0 的 coefficient 更新已有明显分离。
- `D_pattern=0.000365778`：归一化到 `eps` 后，实际渲染 pattern 的分离极小。
- 第 8 步 update cosine 为 `0.827041`，A0/A1 update norm 分别为 `0.003640` / `0.004141`。
- 第 8 步 A0/A1 CGR retention 分别为 `0.995483` / `0.998798`，两者均接受。

这提示“参数空间发生分离但渲染空间变化被强烈压缩”值得后续观察；然而冻结的 `carrier_update_sink` 需要 `D_theta<0.25` **且** `D_pattern<0.01`，本轮不满足前一项，而且整个 decision 仍因 prototype primary metric 缺失而 invalid。逐步 actual update、hash、loss、route 与 pattern snapshots 保存在 [microtrajectory.json](../remote_artifacts/geometry/microtrajectory.json)（SHA-256 `5b5bd2b6…90bd`）。

## 冻结判定与 validation gaps

- `valid=false`
- validity issues：`no_effective_prototype_scale`、`missing_or_nonfinite_primary_metric`
- `first_bad_boundary=null`
- 所有正式 trigger flags 均为 false；这是 validity-first 语义，不等于所有机制都已排除。
- `mAP50_target`、`mAP50_non_target`：`N/A_no_victim_training`
- `PSNR`、`LPIPS`、`poisoned_count`：`N/A_no_materialization`
- 单 seed surrogate probe；不支持 UE、鲁棒性、迁移性或 SOTA 声明。

## 最小后续动作

先做一次 **prototype coverage measurement audit**：只增加每个 person/尺度残差无效原因和分母分解，预注册一个与 TAL/尺度分配语义一致的有效覆盖定义，再原样重跑同一 64/96/8 read-only probe。MALC、CGR、carrier、loss 权重、LR、split、surrogate 和 no-EOT 均保持不变；在有效 primary metric 产生前，不选择多原型、signed carrier、PCGrad 或 victim 训练。
