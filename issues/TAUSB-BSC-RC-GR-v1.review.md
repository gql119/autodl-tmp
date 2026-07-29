# TAUSB-BSC-RC-GR-v1 Review

## PRERUN-REVIEW-01

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-PROBE-01`；禁止启动，等待 `PRERUN-INPUTS-01` 与新的 pre-run review。
- Code snapshot: branch `codex/tausb-bsc-rc-gr-v1`；commit `6c56bf7b2d4bc8861aa52464c90b7271edd8314b`。
- Intent: 只运行 surrogate-only Phase A/B/C mechanism probe；不生成 poisoned dataset，不训练 fresh victim，不更新 Current Best。
- Code location: CLI `ue_project/ue_framework/tools/probe_tausb_bsc_rc_gr.py`；active workflow `ue_project/ue_framework/methods/bsc_rc_gr_probe.py`；carrier integration `ue_project/ue_framework/methods/tausb_universal.py`；R+/R- 与 non-target constraints `ue_project/ue_framework/methods/shadow_tal.py`。
- Parameter data flow: CLI → frozen YAML → `BSCProbeWorkflow` → C0/C1-L/C2-L/C2-LM carrier → clean real TAL/PAG gate → YOLO Detect `cv3`/`cv2` pre-final features → CICR + mutually exclusive R+/R- → per-class one-sided constraints → coefficient-space SVD router/backtracking → Phase A/B/C JSON/status sinks。
- Runtime state: VOC20；target class id `14`；`eps=16/255`；seed `0`；surrogate frozen；唯一 optimizer parameter 为 `ProbeCarrier.coefficients`；prototype 为 detached train-only EMA state，不属于 optimizer。
- Sink effect: Phase A 未 PASS 不进入 B；Phase B 未 PASS 不进入 C；独立 failure signals 可阻断；结果写入 `protocol.json`、`phase_a_metrics.json`、`phase_b_metrics.json`、`phase_c_metrics.json` 与 `status.json`；非有限 JSON 数值转换为 `null`，同时由 finite/gate 判据失败关闭。
- Baseline/disable path: 现有 `tausb_mask` 默认仍为 `carrier_basis_mode=synthetic_fourier` 且 `background_basis_path=""`；数值等价测试覆盖旧 Fourier dispatch；新 probe 是独立入口，不改 `launch_one` 正式阶段。
- Local validation: `py_compile` 通过；配置 `--validate-only` 通过，config hash `031dca3390a5d970ccfa7caecc47687a797fd52fd1aa9f6d52f5947ac20be771`；完整 `tests/` 为 `58 passed`。
- Minimal probe: 本地真实 surrogate `3014748` 参数且 trainable 参数为 `0`；`cv3/cv2` P3/P4/P5 为 `64x8x8`、`64x4x4`、`64x2x2`。真实 surrogate + synthetic annotated batch 的完整 sink probe 得到 person positive count `8`、active non-target class `11`、finite coefficient gradient norm `0.0307333`。该结果仅为 mechanical PASS。
- Run command binding: provisional inner command 为 `cd /root/autodl-tmp/ue_project && python -u -m ue_framework.tools.probe_tausb_bsc_rc_gr --config ue_framework/configs/exp_voc_person_tausb_bsc_rc_gr_probe.yaml --phase all --device 0`。正式 tmux/session/log 命令未冻结，不得执行。
- Experiment validity: 配置声明 VOC20、target `14`、seed `0`、single-GPU、surrogate-only、独立 artifact root `/root/autodl-tmp/ue_project/runs_research/TAUSB-BSC-RC-GR-v1`。本 probe 不运行 clean validation，也没有 robustness transform；victim/clean mAP 均 not_applicable。
- Output non-overwrite: workflow 在任何外部输入读取前检查 artifact root 不存在；存在即 fail closed；不删除、不覆盖、不 resume。外部输入和 surrogate 验证完成后才创建 run root。
- Recoverability/secrecy: repo manifest 禁止绝对路径；本地 source map 不提交；未记录 hostname、端口、用户名、密钥或 token。正式 tmux session、driver log 与远程状态检查待下一次 review 冻结。
- Blockers:
  1. `research_workspace/sources/bsc_background_manifest.json` 尚不存在；8 个授权 person-free source 的 SHA256、尺寸、license note 未冻结。
  2. 本地私有 `bsc_background_local_map.json` 尚未提供；因此 source hash、C1/C2 basis hash/rank 无法计算。
  3. `TAUSB-ALCE-CTX-AUDIT-v1` 当前仍为 draft，要求复用的共享 split manifest 尚未落盘；split hash 无法冻结。
  4. 正式远程 dataset/checkpoint/input 路径和 unique run root 尚未做只读存在性检查；完整 tmux 命令、session 和日志路径尚未绑定。
- Validation gaps: 本地无 VOC dataset；尚未运行真实 VOC calibration/held-out forward；没有 Phase A/B/C mechanism metrics；没有 victim UE 或 clean mAP 证据。

结论：代码路径与最小 sink 已达到进入输入冻结阶段的条件，但 pre-run 硬门禁未满足。不得启动 GPU probe。

## PRERUN-INPUTS-01 Progress

- Split decision: 用户授权 Codex 选择当前更合适方案；采用按
  `TAUSB-ALCE-CTX-AUDIT-v1` 冻结协议生成一个 shared manifest，并让 ALCE/BSC
  复用其 immutable hash。生成器不得覆盖既有 manifest。
- Background authorization: 用户声明“本人持有并授权本研究使用”。
- Intake result: 已收到并核验 `8/8` 张 person-free 背景图；均可解码且 SHA256
  互不重复。

| Stable source id | Dimensions | SHA256 |
|---|---:|---|
| `bg-waves-01` | 960×540 | `02d5976d61ca704bb9cbd547fd6bf9bbecd3baf28cf936716c4b59aad35ee778` |
| `bg-bubbles-02` | 960×720 | `75ff3a5039c8f1d2c5f74262d9259d7373f9a78af5e2f489452b46496b29f374` |
| `bg-beach-03` | 960×640 | `a37692584c6208dd108613a0bd4e08b087e28edf01c20ba5057de34bdf948981` |
| `bg-field-04` | 960×643 | `42cb21ff8b0605a25932871dd63a4b340092c54810bb31e747aaad442f9a092f` |
| `bg-landscape-05` | 960×641 | `55f4e32732d4f1c17ea794433694e090a251ec901fdb61fb2f95cd410f091830` |
| `bg-windmills-06` | 1920×1280 | `92ca0356776751506088728ae624d40c791c93ef7a035083091b45e82939c878` |
| `bg-cliff-beach-07` | 1920×1440 | `5faf6e55afe212a008e3d621fefffc37f67792941d681340dd9c446594ca546b` |
| `bg-tree-08` | 960×638 | `dc8ffacadaa07cae3aa38658099574f81787b8803a56b1962c6ccc68a332f310` |

- Repository manifest:
  `research_workspace/sources/bsc_background_manifest.json`；不含本机路径，
  canonical hash
  `3a13b0f38b06006fd7f68ae03c7206b4b047d4b6129ee7357b05b966641d47af`。
- Local-only map:
  `C:/Users/20272/.local/share/tausb/bsc_background_local_map.json`；保持仓库外。
- Basis validation: `640×640`、16 bases、seed 0 下三组 basis 均 rank 16、
  finite、近似正交。basis hashes：
  - C1-L：`6b88ea983ff292e51571c1ca13f9383c46df9abca5b0d2e14760fe758dbaf267`；
  - C2-L：`5755233f16d00684987d40307b35e792bdd9b67b8267969899d89fdfc8fad636`；
  - C2-LM：`0395c41541d6bcb51ce81805a96271cd099253eee20c94945968d6e1b0f881c1`。
- Shared split:
  `research_workspace/sources/TAUSB-ALCE-CTX-AUDIT-v1.json`；split hash
  `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1`；
  label hash
  `0c8b6f6424061bc31b84ddf42b7370dcbd074f26805433d0ba275c24815e3248`；
  calibration 为 32 person-only + 32 cooccur，held-out 为 32 + 64，
  无重叠且 `validation_gaps=[]`。
- Local checkpoint:
  `voc20_surrogate.pt` SHA256
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`。
- Remaining input blocker: 远程路径与远程同源 hash 尚未只读核验；由于本地真实
  VOC Phase A 已触发科学 failure signal，当前不再推进远程 pre-run。

## LOCAL-PHASE-A-FEASIBILITY-01

- Scope: 用户要求先直接验证方案可行性；使用本地完整 VOC train、真实
  `voc20_surrogate.pt`、批准的 shared split、8-source basis 和 seed 0，执行
  frozen-carrier Phase A。未运行 Phase B/C、未生成 poisoned dataset、未训练 victim。
- Command:
  `python -u -m ue_framework.tools.probe_tausb_bsc_rc_gr --config ue_framework/configs/exp_voc_person_tausb_bsc_rc_gr_probe.local.yaml --phase A --device cpu`。
- Artifact root:
  `ue_project/runs_research_local/TAUSB-BSC-RC-GR-v1-local-phaseA-20260729`。
- Result: `FAIL`；`status.json` 为 `state=stopped`，
  `stop_reason=phase_a_failure_signal`，符合 approved Spec 的停止条件。
- Core evidence:

| Carrier | held-out CICR median | Q25 | non-target/target energy | box residual | source correlation |
|---|---:|---:|---:|---:|---:|
| C0 | 0.575505 | 0.389817 | 0.404013 | 3.089932 | 0.015123 |
| C1-L | 0.390414 | 0.219753 | 0.430254 | 2.471952 | 0.571231 |
| C2-L | 0.353652 | 0.111970 | 0.403262 | 2.487122 | 0.191186 |
| C2-LM | 0.523650 | 0.302853 | 0.354418 | 3.136600 | 0.213312 |

- Gate diagnosis:
  - C2-LM 的 finite、Q25、non-target ratio、box leakage、intended-band、
    protocol hash 与 zero-norm 检查均 PASS；
  - 唯一主失败是 `cicr_improvement`：C2-LM 相对 C0 为 `-0.051855`，
    而冻结阈值要求 `>=+0.10`；
  - C2-L 相对 C0 为 `-0.221853`，且 intended-band 为 `0.691128`，
    略低于 `0.70`；
  - `semantic_dependence=false`、`low_only_unstable=false`，因此失败不是
    raw 背景语义依赖或数值不稳定，而是背景 basis 未产生更一致的目标 residual。
- Paired held-out audit:
  - C2-LM vs C0：`n=89`，paired median delta `-0.050573`，
    bootstrap 95% CI `[-0.069847,-0.027142]`，win rate `0.3483`；
  - C2-L vs C0：`n=93`，paired median delta `-0.195101`，
    bootstrap 95% CI `[-0.238114,-0.163345]`，win rate `0.1505`。
- Scientific verdict:
  当前冻结 carrier 假设不成立，不能进入 R+/R−、gradient routing 或 victim
  阶段。C2-LM 降低 non-target leakage 是可保留信号，但不足以支持“比 C0
  更稳定的统一 residual”声明。若继续，应另立 Spec 检验 matched
  coefficient optimization 后的 C0 vs C2-LM，而不是查看结果后放宽 Phase A 阈值。
