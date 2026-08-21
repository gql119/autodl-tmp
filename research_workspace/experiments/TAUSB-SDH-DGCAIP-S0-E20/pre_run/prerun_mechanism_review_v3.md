# PRERUN-MECHANISM-03

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-MECHANISM-01`
- Code snapshot: `96fc0f7017f043eadf38e8fd26c988c2445e4435`
- Intent: run the matched P1-R/P2-CAIP/P3-DIST/P4-DGCAIP arms under the
  preregistered 20-minute hard cap, without changing research parameters.
- Code location: the exact snapshot contains the bound mechanism config,
  mechanism-only controller, and mechanism-only tmux launcher. The CLI reaches
  `run_dgcaip_pilot` and its four-arm metric/state sink.
- Parameter data flow: bound config -> validated D0/P1/split/Spec hashes -> shared
  calibration and held-out batches -> shared adapter initialization -> four arms
  -> `mechanism_metrics.json` -> optional P4 state only when the full gate passes.
- Runtime state: exact detached checkout is clean at the reviewed commit. Python
  3.8.10, PyTorch 2.0.0+cu118, Ultralytics 8.4.33, and PyYAML are available;
  CUDA is false as expected in no-card mode.
- Sink effect: all D0/P1/surrogate/hiding input hashes, VOC counts, secret
  manifest, and split binding passed. Both exact-checkout control scripts passed
  remote `bash -n`.
- Baseline/disable path: P1-R remains current NLA+CGR; P2/P3/P4 differ only in
  the approved CAIP/distribution/ranking switches. The previous 28 focused tests
  passed and no research code changed in this snapshot.
- Local validation: 28 focused tests passed before snapshot; local and remote
  Bash syntax checks passed for both control scripts.
- Minimal probe: read-only remote input audit passed with 16,551 images, 16,551
  labels, 6,095 person images, passed D0, and exact P1 state/metrics hashes.
- Run command binding: **blocked**. The launcher pins Windows working-tree config
  SHA256 `aef8b3c0...`, but the committed LF-normalized config bytes in the exact
  Linux checkout have SHA256
  `a5de2322f40c090103895d869d5aeb528379ced58be285017d51a615a592119d`.
  The current formal launcher would fail before tmux handoff.
- Experiment validity: VOC20, target id 14, seed 0, 16/24 batches, 8 steps,
  no EOT/JND, fixed budget, and mechanism gates are correctly frozen.
- Output non-overwrite: all formal checkout/artifact/control/cache/tmp/log paths
  and the tmux session were confirmed absent.
- Recoverability/secrecy: timeout, status, terminal evidence, and all-terminal
  shutdown are present; no credential is persisted. The hash mismatch prevents
  an exact reproducible launch.
- Blockers: pin the launcher to the LF-normalized exact-checkout config SHA256,
  create and push a new scoped snapshot, then repeat the exact review.
- Validation gaps: no mechanism arms, victim training, or AP50 evidence exists;
  no GPU process was started.
