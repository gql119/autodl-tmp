# TAUSB-MALC-GRAD-GEOMETRY-S0 remote input and runtime gate

## Reviewed source snapshot

- Branch: `codex/tausb-malc-grad-geometry-audit-v1`
- Reviewed code commit: `18304b96c45360cfba5168d97d21d2961a13f390`
- Remote project root: `/root/tausb-malc-geometry-wt-v2/ue_project`
- Verification: exact HEAD; 131 tracked files under `ue_framework/` and `configs/`;
  source-scoped working, index, and untracked diffs are empty.
- Limitation: this is a sparse exact-source repository. A repository-global `git status`
  would require unrelated historical artifact blobs exceeding 1 GiB, so no global
  `dirty0` claim is made. The executable source/config scope is exact and clean.
- The pre-existing dirty repository at `/root/autodl-tmp` was not modified.

## Read-only input audit

- Remote evidence: `remote_input_audit.json`
- Same-input prior recipe evidence: `prior_v2_input_audit.json`
- VOC train images: 16,551
- Images containing target class 14 (`person`): 6,095
- Frozen calibration / held-out images: 64 / 96
- Free disk at audit time: 26,685,173,760 bytes
- Formal artifact root: fresh
- Formal tmux session: absent

Verified hashes:

- label: `0c8b6f6424061bc31b84ddf42b7370dcbd074f26805433d0ba275c24815e3248`
- split: `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1`
- source manifest: `3a13b0f38b06006fd7f68ae03c7206b4b047d4b6129ee7357b05b966641d47af`
- surrogate: `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`
- semantic bank recipe: `0b8a94efc55155bea20a1ec799bfac14c8a6f11fd6530538f3e0437b37c0dd4b`
- C2LM basis recipe: `8350c0a608150839c98a8dad8db862d0c9dfaeca4714f05d1714afac0f30cfa5`

The first four hashes were recomputed during the current no-card audit. Rebuilding the
semantic bank and C2LM basis on CPU exceeded the five-minute guard, so those two recipe
hashes were cross-checked against the prior v2 input audit that used the same split,
source manifest, and surrogate. The formal workflow independently fails fast if either
recipe hash differs during GPU initialization.

## Cost-guarded command packet

- Wrapper: `run_geometry_cost_guard.sh`
- Local/remote SHA-256:
  `7720af582f914b74fb63babea2d85fdf85dc711c167f95536a97346604cf464a`
- Remote wrapper: `/root/run_tausb_malc_geometry_cost_guard.sh`
- Bash syntax: pass
- tmux session: `tausb-malc-geometry-s0`
- log:
  `/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96/geometry-seed0.log`
- guard status:
  `/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96/cost-guard-status.json`
- formal artifacts:
  `/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry`

Gated launch (not yet executed):

```bash
tmux new-session -d -s tausb-malc-geometry-s0 \
  'bash /root/run_tausb_malc_geometry_cost_guard.sh'
```

The wrapper checks the reviewed commit, source-scoped clean state, frozen input audits,
fresh roots, one CUDA device whose name contains `4090`, and executable shutdown command.
It runs only `probe_tausb_malc_geometry.py`; victim training, poisoned-dataset
materialization, AP50 evaluation, EOT, and resume are not reachable. It snapshots any
available diagnostic JSON and calls `/usr/bin/shutdown` after success, failure, or ten
minutes with no log/artifact progress while both the probe CPU and GPU are idle.

## PRERUN-REVIEW-01 result

- `pre_run_result`: `blocked`
- `decision`: `do_not_run`
- Source/config/input/cost-guard checks: pass, subject to the sparse-scope limitation above
- External blocker: no-card mode reports `torch.cuda.is_available() == False` and zero
  CUDA devices.
- Required next gate: enable the GPU instance, then run `PRERUN-REVIEW-02` to confirm one
  idle RTX 4090 D, unchanged exact commit, fresh roots, absent session, and identical
  wrapper hash before launch.

## PRERUN-REVIEW-02 result

- `pre_run_result`: `pass`
- `decision`: `allow_run`
- GPU: one idle `NVIDIA GeForce RTX 4090 D`; utilization 0%, memory 0 MiB;
  no compute applications.
- CUDA: available; device count 1; torch `2.0.0+cu118`; ultralytics `8.4.33`.
- Reviewed source commit and source/config diff: unchanged and clean. The only untracked
  paths are 36 Python import caches matching `__pycache__/*.pyc`; the gate allows exactly
  that pattern and blocks every other untracked source/config path.
- Formal/control roots: fresh. Tmux session: absent. Shutdown: executable.
- Final wrapper SHA-256:
  `7720af582f914b74fb63babea2d85fdf85dc711c167f95536a97346604cf464a`.
- The wrapper snapshots all seven pre-registered JSON artifacts, including
  `config_resolved.json`, before requesting shutdown.
