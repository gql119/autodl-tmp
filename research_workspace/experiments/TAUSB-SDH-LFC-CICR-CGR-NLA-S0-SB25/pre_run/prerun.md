# TAUSB-SDH-HIDING-SB-v1 pre-run implementation review

## PRERUN-REVIEW-01

- Result: `pass`
- Decision: `allow_run`, but only by executing the exact uploaded launch gate after the user
  enables GPU mode. The no-card review itself does not authorize an alternate command.
- Gated run: seed-0, 120-step, hiding-only `HIDING-S0-SB25-R1`; no mechanism, victim,
  EOT, JND or AP50 stage.
- Code snapshot: branch `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`; implementation commit
  `7f59ab84483f02207594fe6bf89ff870035685cd`; exact clean run commit
  `d244c3270eb24d7a6515e79ff643cb015ebb0bb9`; detached worktree
  `/root/autodl-tmp-sdh-sb25-d244c32` was clean during no-card review.
- Intent: test only whether a fixed `0.25` Haar high-subband bottleneck reduces the hiding
  carrier's high-frequency reliance while preserving the revised hiding gates. RMS CV remains
  descriptive and no RMS/host-diversity objective is introduced.
- Code location: canonical active path is
  `ue_project/ue_framework/methods/semantic_hiding_carrier.py` at
  `raw_residual -> _filter_residual_subbands -> tanh -> delta`. Config validation, construction,
  gate selection and checkpoint reload are in `methods/sdh_experiment.py`; frozen-state reload is
  in `methods/sdh_materializer.py`.
- Parameter data flow: CLI `--config ue_framework/configs/tausb_sdh_hiding_sb25_v1.yaml
  --stage hiding` -> YAML `hiding.hf_subband_scale=0.25` -> fail-closed
  `validate_sdh_experiment_config` -> `run_hiding_pilot` constructor -> `SemanticHidingCarrier`
  field -> Haar LL unchanged and LH/HL/HH multiplied by `0.25` -> inverse Haar -> `tanh` ->
  held-out `compute_hiding_metrics` -> revised gate v2. `hiding.rms_cv_gate_enabled=false`
  excludes only `rms_diversity` from `required_checks`; its metric and diagnostic check remain.
- Runtime state: hiding pretraining still optimizes the existing carrier with only reveal plus
  `0.01 * cover` loss for 120 steps, batch 8, learning rate `0.0002`. The filter has no trainable
  state and is differentiable. D-LFC leakage is appended explicitly to `required_checks`.
- Sink effect: local and AutoDL no-card probes preserved LL, scaled all three Haar high
  subbands by `0.25`, and produced finite backward gradients. The active variant architecture
  hash differs from scale 1, so a misbound checkpoint cannot silently pass the hash gate.
- Baseline/disable path: `hf_subband_scale=1.0` returns `raw_residual` directly without a DWT
  round trip. Exact output, input-gradient and parameter-gradient rollback tests pass. The new
  default architecture descriptor remains byte-compatible with r2; AutoDL loaded the frozen r2
  checkpoint and matched architecture hash
  `8812eb926f7b39637f9562a189d2aa001f3e45336e4e0aeb203954ac9929e7e6`.
- Local validation: Python compile and Python-3.8 AST parse passed. The final focused regression
  command passed `81` tests, including carrier, hiding gate, detector LFC, CICR, multi-parameter
  CGR, mechanism objective, materializer, config, host crop, evaluation, selective routes, NLA
  and VOC20 reporting. `git diff --check` and the scoped credential scan passed.
- Minimal probe: AutoDL no-card Python `3.8.10`, Torch `2.0.0+cu118`, Ultralytics `8.4.33`.
  The approved config parsed, the four-secret bank loaded as `(4,3,256,256)` with primary index
  3, scale/RMS bindings were `0.25/false`, the Haar sink and finite backward passed, and the
  scale-1 r2 checkpoint compatibility probe passed. No optimization or GPU process ran.
- Run command binding: after GPU mode is enabled, the only approved entry is
  `bash /root/verify_and_launch_sdh_hiding_sb25.sh` on the authorized AutoDL instance. Uploaded
  launch-gate SHA-256 is
  `80020b1544fb50405243b76ef24c49f6bb7564f83923f9876a64d85d62548b5e`; wrapper SHA-256 is
  `1ab2898db01763b69c84bf84c4bd4a11349571bf7607fc075ebc879de6f0c6a6`; input-audit SHA-256 is
  `83aa3db3430eabc10e88a2f1b880cc177873fd6c640ffbd5f104f13b243daad9`.
- Experiment validity: VOC train image count `16,551`, person-image count `6,095`, target id
  `14`, seed `0`, deterministic split hash
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`, surrogate SHA-256
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`, epsilon `16/255`,
  secret hashes and no-EOT/no-JND protocol match the approved Spec.
- Output non-overwrite: artifact root
  `/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25`, control root
  `/root/tausb-sdh-control/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/hiding-d244c32-r1`, and tmux
  `tausb_sdh_hiding_sb25_r1` were all absent. The launch gate rechecks all three immediately
  before launch. r1/r2 roots are never deleted, moved, resumed or reused.
- Recoverability/secrecy: the gate launches a detached tmux session and the wrapper records a
  log, JSON cost-guard state and minimal ready evidence. A 1,200-second hard timeout, 10-minute
  idle watchdog, and success/failure/preflight shutdown traps call `/usr/bin/shutdown`. Uploaded
  files contain no credentials; SSH details remain outside the repository.
- Blockers: none in implementation or no-card preflight. GPU mode is intentionally not enabled,
  so the gate has not been executed and no run is currently active.
- Validation gaps: CUDA availability, empty compute-app list, first GPU progress and final hiding
  metrics can only be observed after GPU mode is enabled. The launch gate fails closed and
  requests shutdown if these checks fail. No detector mechanism, victim efficacy, AP50,
  robustness, transfer or perceptual-quality claim is supported by this review.
