# TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1 pre-run review packet

## PRERUN-REVIEW-0

- Result: blocked
- Decision: do_not_run
- Gated run: `python -u ue_framework/tools/probe_tausb_malc_geometry.py --config ue_framework/configs/exp_voc_person_malc_grad_geometry_audit_v1.yaml --device 0`
- Code snapshot: `codex/tausb-malc-grad-geometry-audit-v1@18304b96c45360cfba5168d97d21d2961a13f390`
- Intent: locate the first failed boundary among prototype coherence, MALC cross-batch direction, objective conflict, component-wise CGR survival and carrier update separation. This is surrogate-only and cannot authorize victim training or an AP50 claim.
- Code location: independent `malc_geometry_audit.py`, `sirc_malc_geometry.py`, `probe_tausb_malc_geometry.py` and a unique audit config/root. The old v2 entrypoint and formal artifact root are not called.
- Parameter data flow: CLI loads the frozen YAML; audit validation freezes 64/96 images, batch 4, warm-up 4, microtrajectory 8 and no EOT; the workflow reuses SIRC observation, forms easy/MALC/RMS losses, constructs one active CGR row space per batch, projects all three component gradients through it, then writes geometry and the pre-registered decision.
- Runtime state: only the 16x3 carrier coefficients are trainable. Surrogate and prototype bank remain frozen. G0 stores JSON scalars/lists and detached CPU float64 vectors; G1 alone applies the existing nonlinear route to disposable A0/A1 carriers. `allow_fresh_victim` is always false.
- Sink effect: unit tests reach raw component gradients, the single shared projector, rank-0/full-rank/selective suppression, cross-batch summary, A0/A1 coefficient/pattern separation and ordered decision. The pre-review removed a redundant three-router implementation and added a call-count test enforcing exactly one CGR projector construction per batch.
- Baseline/disable path: A0 omits only MALC. `run_microtrajectory=false` leaves complete G0 artifacts. Existing v2 config validation and v2-related regression tests pass.
- Local validation: 31 focused routing/config tests and 159 full repository tests pass; CLI help, compile, Python 3.8 grammar parse and diff whitespace checks pass. The Windows pytest temp permission error was environment-only and disappeared when using a fresh workspace-local base temp.
- Minimal probe: synthetic component gradients prove shared-projector semantics and JSON/graph detachment. No full local 64/96 real-VOC run was used as scientific evidence.
- Run command binding: command is bound to code commit `18304b96c45360cfba5168d97d21d2961a13f390`, seed 0, target class 14, VOC20, surrogate hash `8de8a0c...`, split hash `e254251...`, and unique root `/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry`.
- Experiment validity: no victim, materialization, AP50, robustness transform, EOT, parameter sweep or carrier freeze is reachable from this tool.
- Output non-overwrite: code fails closed if the artifact root already exists, but the actual remote root has not yet been inspected.
- Recoverability/secrecy: planned execution requires tmux, external log/status monitoring and a ten-minute no-progress shutdown. No credential is stored in the repository or command packet.
- Blockers: the authorized AutoDL endpoint returned `Connection refused` on the single 2026-08-10 availability check. Remote branch/commit, Python/torch/ultralytics, VOC/split/source/checkpoint hashes, GPU, disk and fresh-root state remain unverified.
- Validation gaps: real CUDA graph lifetime and wall-clock cost; exact remote input hashes; output non-overwrite; tmux/watchdog/shutdown command. Formal `pass / allow_run` is forbidden until these are resolved.

## PRERUN-REVIEW-01

- Result: blocked
- Decision: do_not_run
- Gated run: `tmux new-session -d -s tausb-malc-geometry-s0 'bash /root/run_tausb_malc_geometry_cost_guard.sh'`
- Pre-run code commit: `18304b96c45360cfba5168d97d21d2961a13f390`
- Remote source: `/root/tausb-malc-geometry-wt-v2/ue_project` has exact HEAD, 131 tracked source/config files and empty source-scoped working/index/untracked diffs. This sparse exact-source repository intentionally omits unrelated historical artifact blobs, so the review does not claim repository-global `dirty0`; `/root/autodl-tmp` remained untouched.
- Runtime: Python 3.8.10, torch 2.0.0+cu118 and ultralytics 8.4.33 import successfully. Python 3.8 compile/import/config validation passes on the reviewed source.
- Inputs: VOC has 16,551 train images and 6,095 target images; frozen calibration/held-out counts are 64/96. Label, split, source-manifest and surrogate hashes were recomputed and match the Spec. Semantic-bank and C2LM recipe hashes match the prior v2 audit over the same split/source/surrogate and remain fail-fast checks in the formal workflow.
- Non-overwrite: formal artifact root, run-specific control root and tmux session are absent. Disk free at audit time was 26,685,173,760 bytes.
- Wrapper: local and remote SHA-256 `3b132fcfb85c4ef76228bf13a5bf0f0d06d7329b951a1599126f7862757087cf`; remote `bash -n` passes. It exposes only the geometry probe, preserves partial evidence and requests `/usr/bin/shutdown` on success, error or ten minutes with no log/artifact progress while probe CPU/GPU are idle.
- Regression/claim boundary: no victim training, poisoned materialization, AP50, EOT, resume or old-v2 artifact path is reachable. `allow_fresh_victim` remains false.
- Evidence: `research_workspace/experiments/TAUSB-MALC-GRAD-GEOMETRY-S0/pre_run/remote_input_and_runtime_gate.md`, `remote_input_audit.json`, and `prior_v2_input_audit.json`.
- Sole runtime blocker: no-card mode reports `torch.cuda.is_available() == False` and device count 0. Therefore no command was launched.
- Required follow-up: `GPU-ENABLE-01`, then a new `PRERUN-REVIEW-02`. The second review must confirm one idle RTX 4090 D, unchanged commit/source scope, unchanged wrapper hash, fresh roots and absent tmux session before it may emit `pass / allow_run`.
