---
spec_id: TAUSB-SDH-E2E-V0-ONEBOOT-v2
title: SDH E2E V0 单次 GPU 开机、逐阶段硬门禁与条件关机修订
status: approved
experiment_type: orchestration_amendment
parent_spec: TAUSB-SDH-E2E-V0-MAP50-v1
exp_id: TAUSB-SDH-E2E-V0-S0-E20
created: 2026-08-11
approved: 2026-08-11
approval_evidence: user explicitly approved TAUSB-SDH-E2E-V0-ONEBOOT-v2 Spec
---

# SDH E2E V0 单次 GPU 开机、逐阶段硬门禁与条件关机修订

## 1. 问题锚点

- 父 Spec：`docs/research/specs/TAUSB-SDH-E2E-V0-MAP50-v1.md`，其方法、数据、指标、阈值与声明边界保持不变。
- 当前状态：本地与 AutoDL 无卡 pre-run review 已通过；方法实现绑定到
  `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3@3a7a1aaff912d0904794a91a4d3512d18b5c69fa`；尚未运行真实 GPU mechanism、smoke 或 E20。
- 冲突：已生成的 `mechanism_run_contract.json` 是 mechanism-only 合约，成功、失败或超时都会关机，并明确禁止 smoke/E20；它不能执行用户要求的连续流程。
- 本轮问题：能否在一次 GPU 开机中，依次执行 mechanism → P1 核验与配置绑定 → paired smoke → CPU-only 数据流审查 → 条件继续 paired E20，并在任一硬错误或最终完成时自动关机？
- 非目标：不修改 r2 carrier、D-LFC、CICR、CGR、NLA、扰动预算、数据 split、victim 超参数、AP50 判据或正式 200-epoch 协议；不做 EOT、JPEG、blur、gray、transfer 或多 seed。

## 2. 方案比较与决策

| 方案 | 优点 | 风险/代价 | 决策 |
|---|---|---|---|
| 五次独立 GPU 开机，阶段间本地拉取 | 隔离最强 | 多次开关机、等待与人工衔接；E20 无法在 smoke 后自动继续 | 不采用 |
| 一个无中间校验的长 shell 串行运行 | 简单 | mechanism/P1/smoke 错误可能直接污染昂贵 E20 | 不采用 |
| 一个 tmux controller，阶段状态落盘、每步 fail-closed、条件关机 | 兼顾费用、连续性与可审计性 | 需要增加 orchestration controller 与 review | **采用** |

不做本修订的代价不是科学结论改变，而是继续承担多次 GPU 启停和人工衔接成本。

## 3. 冻结身份与输入

### 3.1 代码与环境

- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`
- Method implementation base commit：`3a7a1aaff912d0904794a91a4d3512d18b5c69fa`
- Execution/orchestration commit：`pending`；必须包含上述 method commit，且只能增加本 Spec 所需的 controller/gate/contract/tests/state。
- Remote clean checkout：`pending`；实现后按 execution commit 建立新的 detached clean worktree，禁止在旧 checkout 上直接写脚本。
- Project root：`<new-clean-checkout>/ue_project`
- 旧 `/root/autodl-tmp` 只提供经 hash 核验的数据和 checkpoint，禁止作为可执行代码目录。
- Python：`/root/miniconda3/bin/python`
- Device：`cuda:0`
- Controller session：`tausb-sdh-e2e-v0-oneboot-s0-r1`

### 3.2 数据与固定 hash

- Dataset：`/root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready`
- Train images/labels：16,551 / 16,551；含 person 图像 6,095；target id 14。
- Train image manifest：`4954727df8686532a788668fd815092112ac3e3ee1414eba83b616e683708fbd`
- Train label manifest：`3cd05ad1ab6a546bf2afd5e63cb6c3ff6667064d80af129dd819325625b9d848`
- Surrogate：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`
- r2 hiding metrics：`c7d1b120ffbadeb7385be41669dda704b00a2cee60940e3c3d97112e24e59246`
- r2 checkpoint：`a765e27a62bb1a1939aaae487ff6e61ec405f457056d2329c1c49f91e02c9f36`
- Secret manifest：`a25277499e07310e68a39277461f176dd0d8666e69a4b890328d7b913601ac3e`
- Primary tensor：`cb4b9bba70701c118cc869738ca2713a049ec23b86e89238f32353b06bd4adaf`
- Mechanism config：`ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml`
- Mechanism config hash：`46f757afa7f0a57944af2bec84cab72549230aa431d41bf99e3ff8a25ab4dc56`

### 3.3 新输出根（全部 fresh，禁止 resume/覆盖）

- Controller/control：`/root/tausb-sdh-control/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R1`
- Logs：`/root/tausb-sdh-logs/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R1`
- Mechanism：`/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH`
- Binding：`/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1`
- Arm prefix：`/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0`
- Smoke roots：后缀 `SMOKE-C0`、`SMOKE-M1`
- E20 roots：后缀 `E20-C0`、`E20-M1`
- Final comparison：`/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-COMPARISON-R1`

任何上述新输出根已存在都 fail closed；不得清理、覆盖或默认 `--force_resume`。

## 4. 单次开机状态机

```text
PRECHECK
  └─ fail → SHUTDOWN
MECHANISM (GPU, <=20 min)
  └─ fail/timeout/invalid P1 → SHUTDOWN
P1_VERIFY_AND_BIND (CPU-only, GPU instance stays online)
  └─ hash/schema/config/data mismatch → SHUTDOWN
SMOKE_C0 (GPU, 1 epoch)
  └─ fail/invalid artifacts → SHUTDOWN
SMOKE_M1 (GPU, 1 epoch)
  └─ fail/invalid artifacts → SHUTDOWN
SMOKE_DATAFLOW_REVIEW (CPU-only, GPU instance stays online)
  └─ dataflow/cost/disk gate fail → SHUTDOWN
E20_C0 (GPU, 20 epochs)
  └─ fail/invalid artifacts → SHUTDOWN
E20_M1 (GPU, 20 epochs)
  └─ fail/invalid artifacts → SHUTDOWN
COMPARE_AND_FINALIZE
  └─ success or failure → SHUTDOWN
```

“无卡审查”表示该阶段不调用 CUDA、只做 JSON/YAML/hash/log/路径审查；为避免再次付出启停和排队成本，AutoDL 实例仍保持 GPU 模式在线。

Controller 必须：

1. 在首个可执行语句注册幂等 `EXIT/INT/TERM` shutdown trap；trap 先解除自身再调用 `/usr/bin/shutdown -h now`，避免递归；controller 任意异常退出都必须关机。
2. 每阶段写原子 `stage_status.json`，包含开始/结束时间、exit code、输入/输出 hash、gate 结果和下一阶段。
3. 运行于一个持久 tmux session；SSH 断开不影响流程。
4. 不并行运行两个训练；GPU 阶段严格串行。
5. `nvidia-smi`、进程、日志增长和 status 文件共同作为健康证据，不能只看 tmux 是否存在。

## 5. 各阶段精确协议

### 5.1 PRECHECK

- checkout HEAD、clean status、config hash、数据/label/surrogate/r2/secret hash 全部匹配第 3 节。
- 新输出根全部不存在；GPU 可见；磁盘剩余空间、tmux、`timeout`、`shutdown` 可用。
- 不允许旧 `mechanism_run_contract.json` 直接启动；它仅保留为历史审计证据。
- 任一项失败立即关机，不创建科学结果。

### 5.2 MECHANISM

```bash
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s \
  /root/miniconda3/bin/python -u -m ue_framework.tools.run_tausb_sdh \
  --config ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml \
  --stage mechanism
```

- 内部 `mechanism.max_seconds=900`，外层总上限 1,200 秒。
- 真实运行 T0/T1/P0/P1；不运行 hiding、materialization、victim 或 evaluate。
- 必须存在并核验：`status_mechanism.json`、`mechanism/mechanism_metrics.json`、
  `mechanism/p1_state.pt`、`mechanism/p1_feasibility_sdh_state.pt`。
- `hiding_gate_passed=false` 与真实 `mechanism_gate_passed` 允许保留为 diagnostic；NaN/Inf、OOM、support/Linf/hash/schema/provenance 错误不允许继续。
- 成功后**不关机**；失败或超时立即关机。

### 5.3 P1_VERIFY_AND_BIND

```bash
/root/miniconda3/bin/python -u -m ue_framework.tools.bind_tausb_sdh_e2e_v0 \
  --mechanism-root /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH \
  --mechanism-config ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml \
  --base-config ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml \
  --dataset-root /root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready \
  --output-dir /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1 \
  --run-root-prefix /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0
```

通过条件：

- binder 再次验证 P1 state 与 mechanism/config/r2 hashes；
- `smoke_train_selection.json` 为确定性的 200 张：40 person + 160 person-free；
- 四份 config 均由 `load_config` 通过并写入 canonical/file hash；
- smoke 为 1 epoch、预期 poisoned count C0=0/M1=40；E20 为 20 epoch、C0=0/M1=6095；
- 四臂 `seed=0`、`steps=40`、`imgsz=640`、`batch=36`、SGD、bbox support、no EOT/JND；
- 四个 run root 互异且 fresh。

本阶段可把最小 P1/binding JSON/hash 拉到本地复核，但不得中断 controller 或关闭实例。

### 5.4 Paired GPU smoke

按顺序运行 C0 后 M1：

```bash
/root/miniconda3/bin/python -u ue_framework/launch_one.py \
  --config /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1/smoke-c0.yaml \
  --method tausb_sdh --steps 40 --seed 0 --stage all --gpu_id 0 --run_tag C0

/root/miniconda3/bin/python -u ue_framework/launch_one.py \
  --config /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1/smoke-m1.yaml \
  --method tausb_sdh --steps 40 --seed 0 --stage all --gpu_id 0 --run_tag M1
```

- 每臂依次贯通 materialization → fresh victim 1 epoch → full clean VOC val AP50。
- `--stage all` 不等于 aggregate；smoke 不运行科学 success gate。
- 必须得到完成状态、victim checkpoint、20 类 finite AP50、配置/selection/clean-val hashes。
- M1 `poisoned_count=40`、C0 `poisoned_count=0`；两臂训练选择与 clean-val identity 必须匹配。
- 任一臂首个有效进度 5 分钟内未出现、连续 20 分钟无有效进度、Traceback/OOM/NaN/Inf 或链路产物缺失，立即关机。
- smoke 总 wall cap 为 120 分钟；它是防失控上限，不是预期耗时。

### 5.5 SMOKE_DATAFLOW_REVIEW 与 E20 成本门禁

审查只判断链路正确性，不以 1-epoch AP50 高低判断方法有效性。必须同时满足：

1. C0/M1 的 config、selection、seed、模型结构、优化器、clean-val hash 匹配，只有 arm/poison ratio/expected count/run root 不同。
2. C0/M1 均按同一保存协议 materialize；没有 poisoned root、checkpoint 或 victim 继承。
3. 20 类 AP50 均 finite 且 person 映射到 class id 14；M1/C0 poisoned count 精确。
4. P1 state 的 schema/hash/protocol 一直传到 materialization manifest 与 metrics。
5. 日志中无 silent fallback 到 `tausb_mask`、SIRC、carrier-only、pseudo mask 或 COCO 类空间。
6. smoke 产物与剩余磁盘能支持 paired E20；不能靠删除既有证据腾空间。

E20 成本估计使用真实分阶段 wall time，设 `s=16551/200=82.755`：

```text
estimated_arm = s * smoke_generation
              + 20 * s * smoke_train_one_epoch
              + smoke_full_clean_evaluation
hard_cap_arm  = ceil(1.5 * estimated_arm + 600 seconds)
```

- 该估计是保守费用上界，不作为科学指标。
- 若 paired E20 估计总时长超过 8 GPU-hours，或按 smoke 实测产物增长外推后安全余量不足 1.5 倍，自动关机并等待用户重新批准费用；这不记为方法失败。
- 若所有数据流、磁盘与成本门禁通过，controller 不关机，直接进入 E20。

### 5.6 Paired E20

按顺序运行独立 C0 后 M1：

```bash
/root/miniconda3/bin/python -u ue_framework/launch_one.py \
  --config /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1/e20-c0.yaml \
  --method tausb_sdh --steps 40 --seed 0 --stage all --gpu_id 0 --run_tag C0

/root/miniconda3/bin/python -u ue_framework/launch_one.py \
  --config /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R1/e20-m1.yaml \
  --method tausb_sdh --steps 40 --seed 0 --stage all --gpu_id 0 --run_tag M1
```

- C0 必须先完成 20 epochs、`poisoned_count=0`、20 类 finite AP50；否则不启动 M1。
- M1 独立 fresh victim，6,095 张含 person 图像使用同一 frozen P1 V0 state，其他训练图像 clean。
- 每臂 wall cap 使用第 5.5 节动态值；首进度 5 分钟、连续 20 分钟无有效进度为 bug guard。
- 正常长训练允许超过 20 分钟；20 分钟限制只针对无进度/故障诊断。
- M1 完成后用 `ue_framework.tools.compare_tausb_sdh_e2e_v0` 读取两个显式 metrics 路径，输出 `comparison.json` 和 `per_class_ap50.csv`。
- comparison 成功或任何 E20 失败后都执行最终关机；不自动启动 200 epochs、载体调优或新 seed。

## 6. Research Contract

### Hypothesis

科学假设沿用父 Spec：r2 fixed semantic carrier 经当前 D-LFC、CICR、CGR 与 NLA 联合更新后，相对 matched C0，可在 seed-0 20-epoch fresh victim 中降低 clean-val person AP50，而 19 类非目标宏平均不会同量级崩塌。单次开机只改变运行编排，不应改变该因果比较。

### Success Signal

E20 directional feasibility PASS 必须同时满足：

1. `AP50_person(C0)-AP50_person(M1) >= 0.10`；
2. 非目标 19 类宏平均 AP50 drop `<=0.08`；
3. 至少 15/19 个非目标类 AP50 drop `<=0.15`；
4. M1 `poisoned_count=6095`、actual `Linf<=16/255+1/255`、20 类 AP50 finite；
5. C0/M1 配置、seed、训练/验证 manifests、victim 初始化协议匹配。

### Failure Signal

以下任一项独立否定当前 V0 推进价值：

1. person AP50 drop `<0.03`；
2. 非目标宏平均 drop `>0.15`；
3. 至少 5/19 个非目标类 drop `>0.20`；
4. M1 `poisoned_count!=6095`，或 Linf/support/hash/materialization 非确定；
5. 任一 arm 未完成 20 epochs、评估非 finite、类别映射或 clean-val identity 不一致。

Success 与 Failure 之间标记 `inconclusive_tradeoff`。mechanism/smoke/成本门禁失败是 `operational_failure` 或 `cost_gate_stop`，不能改写成科学 Failure。

### Metric & Split

- Dataset：VOC2007+2012；train 16,551，含 person 6,095；clean VOC val。
- Victim：两臂独立 YOLOv8n-style from scratch；seed0；E20；imgsz640；batch36；SGD。
- Primary：person AP50 drop、19 类 non-target macro AP50 drop。
- Secondary：20 类逐类 AP50/drop/retention、mAP50_all、person-free/cooccur non-target AP。
- Quality：poisoned count、actual Linf、PSNR；LPIPS 缺失必须标 `validation_gap`。

### Stop Condition

- precheck、mechanism、binding、smoke 或 smoke dataflow review 任一失败：不进入 E20，并立即关机。
- 输入 hash 漂移、旧输出根存在、NaN/Inf、OOM、Traceback、GPU 无目标进程、连续 20 分钟无有效进度：终止当前流程并关机。
- paired E20 预计超过 8 GPU-hours 或磁盘安全余量不足：成本门禁停止并关机。
- E20 C0 失败：不启动 M1；M1 完成或失败：最终关机。

### Claim Boundary

- smoke 只证明端到端数据流，不证明方法有效。
- E20 只有单 seed、20 epoch、failed-scientific-gate feasibility state，只能形成 tentative directional evidence。
- 不得声称 hiding/mechanism 已验证、shortcut 机制成立、正式 UE 成功、鲁棒性/迁移性/SOTA 或多 seed 稳定性。
- 单次开机编排与分阶段开机在科学协议上等价；如果输入/config/hash不同，比较无效。

## 7. 实施与本地验证

| Step | 入口 | 原子改动 | 必需证据 |
|---|---|---|---|
| 1 | 新 one-boot controller/contract | 用状态机替代旧 mechanism-only 执行入口；旧文件保留 | shell syntax、payload hash、trap/shutdown unit test |
| 2 | controller gate helpers | P1/binding/smoke/E20 JSON/hash/路径/finite AP 校验 | focused unit tests |
| 3 | cost estimator | 从 smoke 分阶段 wall time计算动态 E20 cap和磁盘投影 | 边界/缺字段/fail-closed tests |
| 4 | issues CSV/review | 将原来的多次 shutdown/pull 行改为一个连续远程 run 与最终 pull | CSV validator、dependency audit |
| 5 | pre-run implementation review | 复核 CLI/config → P1 → materializer → victim → metrics → compare | review result=`pass` 后才允许开 GPU |

实现不得修改方法损失或 config 语义；只允许新增 controller、gate、contract、测试和执行状态更新。

## 8. Pre-run Review

- reviewed branch/commit：`pending; implementation 后重新冻结`
- exact controller payload/hash：`pending`
- old mechanism-only contract superseded marker：`pending`
- input/config/output non-overwrite：`pending`
- shutdown trap success/fail/timeout paths：`pending`
- smoke → E20 dataflow/cost gate：`pending`
- result：`pending`

在本节为 `pass` 前不要开启 GPU。

## 9. 结果落盘

- Controller state/log：`pending`
- Mechanism/P1 evidence：`pending`
- Binding report：`pending`
- Smoke C0/M1 metrics与审查：`pending`
- E20 cost estimate：`pending`
- E20 C0/M1 metrics：`pending`
- VOC20 comparison：`pending`
- Local pull/ingest/H→E→N：GPU 自动关机后执行，`pending`
- STATE decision：仍由用户批准，`pending`
