---
spec_id: TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1
title: MALC 梯度几何与载体更新瓶颈审计
status: approved
approved: 2026-08-10
experiment_type: probe
csv: issues/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1.csv
created: 2026-08-10
source_exp: TAUSB-SIRC-MALC-CGR-MAP50-S0
---

# MALC 梯度几何与载体更新瓶颈审计

## 1. 问题锚点

- STATE 关联：当前可信主线仍是已有 TAUSB 单 seed best。本 Spec 只诊断失败的
  SIRC-MALC-CGR 支线，不提升其优先级，也不修改 Current Best。
- 触发证据：
  `research_workspace/experiments/TAUSB-SIRC-MALC-CGR-MAP50-S0/analysis/mechanism_gate.md`。
  在 reviewed commit `fcf26cc24dc7e6943234cc3cdf7943fd957cb6cc` 上，A1 相对 A0 的
  residual cosine median 只提高 `0.004839`，Q25 仍为 `-0.047634`，
  log-energy MAD 比值为 `0.99632`；但组合梯度经 CGR 后保留率为 `0.969713`。
- 本轮问题：MALC 没有使 A1 与 A0 分离的第一处瓶颈，究竟位于：
  1. 单一全局残差原型本身不稳定；
  2. MALC 梯度跨 batch 不一致或与 easy-cls/RMS 冲突；
  3. CGR 选择性移除了 MALC 分量；
  4. SIRC 正调制参数化无法把有效梯度转成不同 pattern。
- 非目标：不训练 C0/M1 victim，不生成投毒数据集，不计算 AP50，不改变 MALC、CGR、
  carrier、学习率或 loss 权重，不引入 EOT/PCGrad/新原型，不做参数扫描、鲁棒性或迁移实验。

## 2. Idea Source

- 来源类型：失败机制实验的梯度方向缺口。
- 证据链：上述 mechanism report、A0/A1 diagnostics、`malc_calibration.json`，以及：
  - `ue_project/ue_framework/methods/malc_calibration.py`：当前只标定 gradient norm；
  - `ue_project/ue_framework/methods/constraint_gradient_router.py`：已暴露
    `target_gradient` 和 `projected_target_gradient`；
  - `ue_project/ue_framework/methods/semantic_residual_carrier.py`：载体使用
    `1+tanh(theta)` 正调制；
  - `ue_project/ue_framework/methods/sirc_probe.py:455`：已有梯度余弦工具语义。
- 为什么现在做：现有结果已证明“继续增大 MALC 权重或重复 40 步”缺乏依据；先定位第一处
  方向瓶颈，成本远低于 victim 训练，并能防止一次修改多个模块。

### 2.1 候选方案比较

| 方案 | 能回答的问题 | 主要代价/风险 | 决策 |
|---|---|---|---|
| A. 直接增大 `lambda_malc` 或延长优化 | 只回答更强压力是否改变结果 | 无法区分梯度冲突、CGR 与 carrier；容易继续盲试 | 不采用 |
| B. 立即换成多原型或 signed carrier | 可能提高表达能力 | 同时改机制与载体，失败后仍无法归因 | 不采用 |
| C. 只读梯度几何 + 8 步匹配微轨迹 | 依次定位 prototype、raw gradient、CGR、update sink | 需要一次短 GPU probe，但不训练 victim | **采用** |

不运行的替代方案是保持 v2 失败结论并停止该支线；它不会产生新的错误声明，但也无法决定
下一次只能修改哪一个模块。

## 3. 机制与接入

### 3.1 固定输入与控制

- 复用 v2 的 VOC、target id `14`、surrogate、8 张授权背景源、semantic bank recipe、
  split hash、SIRC 16 bases/48 coefficients、4 variants、`eps=16/255`、batch `4`、
  warm-up `4`、no-EOT 与 CGR 配置。
- calibration 固定为 64 images / 16 batches；held-out 固定为 96 images / 24 batches。
- held-out 只计算残差/原型泛化统计，不更新原型、梯度权重或载体。
- 旧 v2 tool/config/正式 root 均只读；新 probe 使用独立 artifact root，禁止覆盖。

### 3.2 G0：原型与梯度几何审计

对 calibration batch `b`、尺度 `l` 和 component
`k in {easy_cls, malc, rms}` 记录：

```text
R_l       = || mean_i normalize(r_i,l) ||
S_l^LOO   = cosine(mu_l^full, mu_l^(-b))

g_k^b     = grad_theta L_k^b
C_kk'^b   = cosine(g_k^b, g_k'^b)
C_malc^bb'= cosine(g_malc^b, g_malc^b')

rho_k^b   = ||P_b g_k^b|| / (||g_k^b|| + eps)
```

`P_b` 必须来自该 batch 当前 clean non-target constraints 的同一 SVD 行空间。
对每个 component 单独记录投影前/后的 norm 与 cosine；不得用组合梯度 retention 代替。
所有统计只使用 detached CPU 副本，禁止保存 detector autograd graph。

### 3.3 G1：8 步匹配微轨迹

- A0 与 A1 从同一 warm coefficients 出发，使用 calibration 前 8 个 batch 的完全相同顺序；
  A0 为 easy-cls+RMS+CGR，A1 仅多 MALC。
- 沿用 v2 的 frozen prototype、frozen norm weights、learning rate 和 nonlinear backtracking；
  不改变任何科学参数。
- 每步记录 coefficient hash、接受模式、实际 update、A0/A1 coefficient distance、update
  cosine；第 0、4、8 步记录渲染 pattern。

```text
D_theta = ||theta_A1^8 - theta_A0^8|| /
          max(||theta_A0^8 - theta_warm||, eps)

D_pattern = RMS(delta_A1^8 - delta_A0^8) / epsilon
```

microtrajectory 结束后丢弃载体；不得产生 `a1_frozen_carrier.pt`，不得设置
`allow_fresh_victim=true`。

### 3.4 预注册的 first-bad-boundary 分类

先检查所有 validity gates，再按以下顺序选择唯一的 `first_bad_boundary`；同时保留所有原始
触发信号，避免把“最早”写成“唯一因果”。

1. `prototype_incoherence`：在 coverage `>=0.80` 的有效尺度中，median `R_l<0.20`
   或 median LOO-Q25 `<0.80`。
2. `cross_batch_malc_conflict`：pairwise `C_malc^bb'` median `<=0` 或 Q25 `<-0.10`。
3. `objective_gradient_conflict`：batch-wise `cos(g_malc,g_easy)` 或
   `cos(g_malc,g_rms)` 的 median `<-0.10`。
4. `cgr_selective_suppression`：median `rho_malc<0.20`，或在 `rho_easy>=0.20` 时
   `median(rho_malc/rho_easy)<0.50`。
5. `carrier_update_sink`：前四项均未触发，但 `D_theta<0.25` 且 `D_pattern<0.01`。
6. `unresolved_by_probe`：前五项均未触发。

此分类只决定下一份方法 Spec 的单一修改对象，不直接宣称因果已经完全证明。

### 3.5 canonical 接入与回退

- 新入口：`ue_framework/tools/probe_tausb_malc_geometry.py`。
- 新实现优先放在独立 `malc_geometry_audit.py`；只复用现有 observation、MALC、CGR 与
  carrier，不改变 v2 运行语义。
- 参数链：
  `new config -> geometry tool -> SIRC observation -> MALC components -> component gradients
  -> CGR projector -> geometry sinks -> 8-step matched microtrajectory -> decision.json`。
- feature-off / baseline：A0 是 MALC-off 精确控制；关闭 `run_microtrajectory` 时 G0 仍应
  产生完整只读 geometry artifacts。旧 `probe_tausb_sirc_malc.py` 的行为与输出 hash 不得改变。

## 4. 实施与本地验证

| Step | 文件/入口 | 原子改动 | 本地证据 |
|---|---|---|---|
| 1 | `malc_calibration.py` 或独立 audit module | 计算 resultant/LOO，不改变 frozen prototype | 合成单峰、双峰、零向量、leave-one-batch 测试 |
| 2 | `malc_geometry_audit.py` | component gradient cosine 与同一 projector 的 per-component retention | 正交、冲突、rank-0/full-rank、disconnected/nonfinite 测试 |
| 3 | geometry workflow | 8 步匹配 A0/A1，记录 coefficient/pattern separation | 同初值/同 batch、A0 回退、held-out 不写、graph 不滞留测试 |
| 4 | tool/config | 新独立入口、fresh root、no-EOT、固定 hashes | config parse、CLI dry-run、Python 3.8 AST/import |
| 5 | decision sink | 按固定优先级输出唯一 boundary 和全部触发项 | 六类合成 decision-table 测试 |

本地验证只证明实现和参数链，不证明真实梯度几何结论。

## 5. 远程运行

- Branch：`codex/tausb-malc-grad-geometry-audit-v1`。
- Reviewed source base：`fcf26cc24dc7e6943234cc3cdf7943fd957cb6cc`；实现后绑定新 commit。
- 入口：

```bash
cd /root/tausb-malc-geometry-wt/ue_project
python -u ue_framework/tools/probe_tausb_malc_geometry.py \
  --config ue_framework/configs/exp_voc_person_malc_grad_geometry_audit_v1.yaml \
  --device 0
```

- ExpID / RunID：`TAUSB-MALC-GRAD-GEOMETRY-S0 / geometry-seed0`。
- Artifact root：`/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry`。
- 预计资源：单张 RTX 4090 D；目标运行时间 `<10 min`。使用 tmux 和独立日志/status。
- 成功、异常、或 10 分钟无 artifact/log 进度均请求 AutoDL shutdown。
- Resume / rollback：fresh root fail-closed；不默认 resume；失败产物归档而不覆盖；回退是删除新
  probe 的调用，不修改旧 v2 入口或证据。

预期最小 artifacts：

- `status.json`
- `config_resolved.json`
- `input_audit.json`
- `prototype_geometry.json`
- `gradient_geometry.json`
- `microtrajectory.json`
- `diagnostic_decision.json`

## 6. Research Contract（首次远程运行前冻结）

- **Hypothesis**：v2 中 MALC 无法使 A1 与 A0 分离，首要原因发生在 CGR 和 carrier
  更新之前，即单一全局 prototype 不稳定、MALC 梯度跨 batch 不一致，或 MALC 与
  easy-cls/RMS 存在系统性方向冲突；单纯的组合梯度 norm calibration 隐藏了该问题。
- **Success Signal**：输入、16 个 calibration batch、24 个 held-out residual batch 和
  8 步 matched microtrajectory 全部完整且 finite；`first_bad_boundary` 唯一落在
  `prototype_incoherence`、`cross_batch_malc_conflict` 或
  `objective_gradient_conflict`，同时 median `rho_malc>=0.20`。这支持“方向问题发生在
  CGR/carrier 之前”的假设，但不支持 UE 效果声明。
- **Failure Signal**（独立定义）：
  1. `first_bad_boundary` 为 `cgr_selective_suppression` 或 `carrier_update_sink`，直接反驳
     “首要问题在 CGR 之前”；
  2. `unresolved_by_probe`，说明当前测量不足，禁止凭直觉选择新方法；
  3. component gradient disconnected/nonfinite、有效 calibration batch `<16`、held-out
     residual batch `<24`、A0/A1 初值或 batch 顺序不一致、held-out 回写、旧 v2 sink 变化，
     均使 probe 无效而不是支持任何机制结论。
- **Metric & Split**：
  - primary：calibration prototype resultant/LOO stability、cross-batch MALC gradient
    cosine、MALC-vs-easy/RMS cosine、per-component CGR retention；
  - secondary：8-step `D_theta`、`D_pattern`、update cosine、CGR mode/acceptance；
  - validation：固定 calibration/held-out split；不使用 VOC val AP50；
  - quality：本轮不 materialize 数据，PSNR/LPIPS/poisoned_count 不适用并明确记为 N/A。
- **Stop Condition**：任一 hash/input 不一致、旧 artifact root 已存在、NaN/Inf/OOM、
  autograd graph 跨 batch 滞留、batch count 不完整、旧 v2 回退测试失败、日志/artifact
  10 分钟无进度时立即停止并关闭实例。无论诊断分类为何，都不进入 victim 训练。
- **Claim Boundary**：本轮只是 seed0 surrogate calibration/held-out 诊断；不声称 AP50、
  fresh-victim UE、鲁棒性、迁移性、MALC 最终因果贡献或 SOTA。多个 boundary 同时触发时，
  只把预注册顺序中的最早项用于选择下一次单变量修改。

## 7. Pre-run Review

- reviewed branch / commit：`pending`
- exact command：`pending`
- parameter sink probe：`pending`
- baseline/disable-path evidence：`pending`
- output non-overwrite check：`pending`
- Python 3.8 / remote import：`pending`
- cost guard：`pending`
- result：`pending`

## 8. 结果落盘（运行后填写路径，不事后改判据）

- Remote artifacts：`pending`
- Diagnostic decision：`pending`
- H->E->N analysis：`pending`
- Experiment ledger：`pending`
- STATE update decision：`pending`

## 9. Spec 自审

- 没有修改 v2 的失败判据，也没有将诊断包装成 AP50/UE 实验。
- A0、raw component、per-component CGR 和 microtrajectory 分别对应四个候选瓶颈。
- 每个阈值和 first-bad-boundary 顺序均在远程运行前冻结；不允许看结果后重排。
- 新入口、新 root、旧入口回归和无覆盖策略可验证。
- 用户已于 2026-08-10 批准本 Spec；后续仍须通过执行 CSV、本地验证和独立 pre-run
  review，才允许启动短 GPU probe。
