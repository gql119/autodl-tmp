---
spec_id: TAUSB-SDH-HIDING-SB-v1
title: SDH single-secret Haar spectral-bottleneck hiding retry
status: approved
experiment_type: ablation
csv: issues/TAUSB-SDH-HIDING-SB-v1.csv
created: 2026-08-11
approved: 2026-08-11
approval_evidence: user explicitly approved the RMS-descriptive spectral-bottleneck revision
---

# SDH 单一 secret 的 Haar 频谱瓶颈门禁修订

## 1. 问题锚点

- STATE 关联：当前主线仍是固定高语义 secret、person bbox 宿主、sample-adaptive
  deep hiding；不恢复旧 Fourier/ALCE/PAG/late-repair 路线。
- 触发证据：`HIDING-S0-R2` 在 reviewed commit `20c35b6` 上正常完成，但预注册
  hiding gate 失败。高频能量为 `0.671792 > 0.40`，每通道 RMS CV 为
  `0.018599/0.010226/0.015434 < 0.05`。
- 本轮问题：一个固定的 Haar 高频子带缩放，能否减少当前 carrier 对高频隐写通道的
  依赖，同时保留 unseen-primary-secret 恢复、同一 carrier 身份和已经观察到的有限
  像素纹理差异？本轮不要求不同宿主产生更大的能量差异。
- 非目标：不改 D-LFC、CICR、CGR、NLA、TAL、victim、数据集、secret 筛选、support、
  epsilon、EOT 或 JND；不运行 mechanism/victim/AP50。

## 2. Idea Source 与方案比较

- 来源类型：预注册 mechanism gate 的真实失败信号。
- 证据链：
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/analysis/hiding_gate.md`
  及其引用的 hash-verified artifacts。
- 方案 A——只增加 hiding steps：不采用。r2 的恢复已通过、扰动已达到 epsilon 边界，
  更多步数不能直接约束高频比例或 RMS CV，且可能强化现有高频捷径。
- 方案 B——同时加入 FFT 比例损失和 RMS-CV hinge：不采用。它一次修改两个 loss，
  直接优化门禁统计量，并会把本轮并不需要的跨宿主能量差异强加给 carrier，可能削弱
  容易学习的类级共享信号。
- 方案 C——固定 Haar 频谱瓶颈：采用。它只有一个参数、具有精确 feature-off 回退，
  并直接检验能否在不破坏共享 carrier 身份的情况下减少高频依赖。
- 不运行的代价：当前 checkpoint 被 gate 明确禁止进入 mechanism；直接训练 victim
  会浪费 GPU 并污染机制结论。

## 3. 机制与接入

- 核心机制：在 `SemanticHidingCarrier.forward` 中，`raw_residual` 进入 `tanh` 之前做
  一次已有 `FixedHaarDWT`：保留 LL，将 LH/HL/HH 统一乘
  `hiding.hf_subband_scale=0.25`，再 inverse DWT 得到 filtered residual。
- canonical 接入点：
  `ue_project/ue_framework/methods/semantic_hiding_carrier.py` 中
  `raw_residual -> delta` 的唯一活动路径。
- 参数链：retry config `hiding.hf_subband_scale` → `run_hiding_pilot` →
  `SemanticHidingCarrier(..., hf_subband_scale=...)` → `forward` filtered residual →
  held-out `compute_hiding_metrics`。
- feature-off：`hf_subband_scale=1.0` 必须与旧路径在浮点容差内输出等价；旧 r2 artifacts
  是 frozen baseline，不重跑。
- 预期收益：降低 `[64,+inf)` FFT 能量，同时保持同一 secret 在不同 person 上的
  高度一致载体身份与非完全相同的具体像素纹理。
- 可能副作用：恢复 SSIM/L1 margin 下降；pixel cosine 退化到固定模板区间；低频幅值
  增大但仍受 `eps=16/255` 和 support 门禁约束。RMS CV 只记录，不作为副作用或成败判据。

## 4. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | `semantic_hiding_carrier.py` | 新增一个有界 `hf_subband_scale`，在 residual 上缩放 Haar 高频子带 | `scale=1.0` 输出/梯度回退等价；`0.25` 高频子带能量下降；finite gradient |
| 2 | `sdh_experiment.py` 与 retry config | 显式绑定 config → constructor；缺失/非法值 fail closed | config parse、参数 sink probe、feature-off test |
| 3 | focused tests | 只扩展 carrier/config/hiding validation 测试 | 原 67 项 SDH focused regression 全部通过 |
| 4 | no-card pre-run review | 审查精确 checkout、输入 hash、全新 root/session、成本脚本 | `pass` 前不得开 GPU |

不改变 reveal/cover loss，不加入 RMS/host-diversity loss，不改变 120 steps、batch 8、
learning rate、secret bank 训练/primary held-out 语义或 64/96 split。

## 5. 远程运行

- 入口：`python -m ue_framework.tools.run_tausb_sdh --config ue_framework/configs/tausb_sdh_hiding_sb25_v1.yaml --stage hiding`。
- Method / Steps / Seed：`tausb_sdh / 120 / 0`。
- Stage：仅 `hiding`；显式禁止 `mechanism`、victim、EOT、JND。
- Remote project root：clean detached/worktree checkout at reviewed commit。
- ExpID / RunID：`TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25` / `HIDING-S0-SB25-R1`。
- Artifact root：`/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25`；tmux
  session 与 control root 在 pre-run review 绑定 reviewed commit，并且必须与 r1/r2 全部
  不同且预先证明不存在。
- 预计时长：r2 为 17.4 秒；仍保留外层 20 分钟 hard cap、10 分钟无进度 watchdog、
  成功/失败/超时自动关机。
- Resume：不 resume、不覆盖；失败证据原样保留。rollback 为 scale `1.0`，但不为省钱
  重跑已存在的 r2 baseline。

## 6. Research Contract

- **Hypothesis**：当前 carrier 已形成可恢复、类级一致且像素纹理非完全相同的共享
  secret 信号，但过度依赖高频编码。将 Haar LH/HL/HH 固定缩放为 `0.25`，在不改变
  loss、数据、secret、RMS 分布目标或训练步数的前提下，会使 held-out 高频比例降至
  `<=0.40`，同时保留原有恢复、非完全固定像素纹理和非目标泄漏门禁。
- **Success Signal**：固定 seed 0、同一 64 calibration/96 held-out split、同一 secret bank
  下，high-frequency median `<=0.40`；retrieval `>=0.90`；primary SSIM `>=0.50`；
  relative L1 margin `>=0.20`；pairwise pixel cosine `<0.98`；cooccur balanced accuracy
  `<=0.60`；non-target macro-AUROC `<=0.60`；finite/support/Linf 全部通过。每通道 RMS CV
  必须报告，但没有硬阈值，也不要求相对 r2 提升。
- **Failure Signal**：以下任一项独立否定本假设：高频比例仍 `>0.40`（固定瓶颈没有
  解决目标问题）；恢复 SSIM `<0.50` 或 L1 margin `<0.20`（瓶颈破坏隐藏容量）；
  pairwise pixel cosine `>=0.98`（退化为预注册定义下的近固定像素模板）；cooccur
  balanced accuracy 或 macro-AUROC `>0.60`（低频通道引入 collateral semantics）；
  输出非 finite、support 泄漏或超出 Linf。RMS CV 低不单独构成 Failure Signal。
- **Metric & Split**：baseline 为 hash-verified `HIDING-S0-R2`；variant 只运行 seed 0，
  使用 split hash `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`；
  primary metrics 为 revised hiding gate 统计量；RMS CV 作为 descriptive secondary metric；
  不产生或填充 AP50/PSNR/LPIPS。
- **Stop Condition**：本地回退等价或参数 sink 不通过则不远程运行；远端任一 revised
  hiding hard gate 失败、NaN/Inf、OOM、输入/hash 漂移、root 已存在、超时或自动关机
  失效时立即停止，不进入 mechanism/victim。RMS CV 单独偏低不触发停止。
- **Claim Boundary**：这是一个单 seed、hiding-only、mechanical ablation。PASS 只允许
  新 checkpoint 进入独立 mechanism pre-run review；不支持目标类不可学习、非目标保持、
  AP50 改善、鲁棒性、迁移性或 SOTA 声明。FAIL 必须保留并停止。

## 7. Pre-run Review

- reviewed branch / commit：`pending after approval and implementation`。
- exact command：`pending`。
- parameter sink probe：`pending`。
- baseline/disable-path evidence：r2 hash-verified artifacts + `scale=1.0` equivalence test。
- output non-overwrite check：`pending`。
- result：`pending`。

## 8. 结果落盘

- Remote artifacts：`pending`。
- H→E→N analysis：`pending`。
- Experiment ledger：hiding-only 不写 AP50 ledger；如需索引只记录 mechanical gate。
- STATE update decision：仅在用户审核 evidence 后决定，不自动更新 Current Best。
