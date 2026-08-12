# TAUSB-SDH-E2E-V0-S0-E20 H→E→N 分析

## Hypothesis

预注册假设是：相对 matched C0，当前 r2 fixed semantic carrier 经
D-LFC、CICR、CGR 与 NLA 联合更新后，能够在 seed-0、20-epoch fresh
victim 上降低 clean-val person AP50，同时避免其余 19 类出现同量级崩塌。

Success 必须由 paired E20 同时证明 person AP50 下降至少 0.10、非目标宏平均
下降不超过 0.08，并满足逐类保持、6095 个 poisoned images、Linf 和协议一致性。
Failure 也必须基于 paired E20。Spec 明确规定：smoke 或成本门禁停止不能改写成
科学 Failure，smoke 只证明数据流。

## Evidence

- R4 controller 终态为 `cost_gate_stop`，不是代码异常。
- `PRECHECK`、P1/binding 复核、`SMOKE_C0` 和 `SMOKE_M1` 均完成；两臂退出码
  都是 0，GPU 进程均被观察到。
- C0 smoke 用时 145.24 秒；M1 smoke 用时 135.25 秒。两臂均完成
  generate、train 和 clean evaluate。
- M1 materialization 产生 40/200 个 poisoned images；manifest 中 40 个 poisoned
  rows 的 `secret_source_sha256` 唯一且一致，证明 R3 暴露的 provenance 缺口已经修复。
- M1 smoke 的实际 Linf max 为 0.0627445，PSNR 为 34.8598，LPIPS 为
  0.02080；这些只描述 40-image smoke materialization，不是完整数据集或最终方法质量。
- 数据流门禁通过，但 paired E20 估算为 213,440.76 秒（约 59.29 小时），高于
  批准的 8 小时上限；投影需要约 29.79 GB，实际剩余约 13.16 GB，低于 1.5×
  安全余量要求。因此 controller 没有创建 E20 或 comparison roots，并自动关机。
- C0/M1 smoke 的 20 类 AP50 都是 0。原因是 victim 只训练 1 epoch；二者没有
  区分能力，不能解释成 person collapse、非目标保持、方法有效或方法失败。
- 19 个拉回文件共约 365 KB；三份 transfer report 均为
  `missing_required=[]`、逐文件 SHA-256 通过，且不包含数据集、图片、checkpoint、
  权重或凭据。

关键原始证据：

- `remote_artifacts/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R4/controller_status.json`
- `remote_artifacts/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R4/smoke_review.json`
- `remote_artifacts/TAUSB-SDH-E2E-V0-S0-R4-SMOKE-C0/.../metrics/metrics.json`
- `remote_artifacts/TAUSB-SDH-E2E-V0-S0-R4-SMOKE-M1/.../metrics/metrics.json`
- `metrics-summary.json`

## Judgment

当前科学假设仍是 `inconclusive / not evaluated`。本轮取得的是端到端机械闭环：
单一 secret → P1 state → person bbox materialization → fresh victim train → clean eval →
provenance/metric sink 已经贯通。它没有提供目标类不可学习或非目标类保持的 E20 证据。

本轮也不是 scientific failure：E20 没有启动，预注册 Success/Failure 条件均不可计算。
终态应归类为 `cost_gate_stop`，且自动止损逻辑按批准 Spec 正常工作。

## Next

下一步不应继续调 carrier，也不应再运行相同的 200-image/1-epoch smoke。最小判别实验是
重新冻结一个“成本可承受且 AP50 已脱离全零区”的 paired victim pilot：使用同一 P1、
同一 C0/M1、同一 clean validation 和 seed0，只缩小训练集或 epoch，预先规定硬 GPU
时长与磁盘预算，并要求 clean C0 的 person/非目标 AP50 达到可解释下限后才比较 M1。
这属于评测协议与成本门禁变更，必须先形成并批准新 Spec；不能直接绕过本次 8 小时和
磁盘止损条件启动完整 E20。

不建议更新 `STATE.md` 的 Current Best；用户仍保留最终科研判断权。
