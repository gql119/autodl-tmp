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
