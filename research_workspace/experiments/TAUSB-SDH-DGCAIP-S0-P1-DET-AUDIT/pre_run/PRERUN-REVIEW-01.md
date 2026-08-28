# TAUSB P1 Determinism Audit — Pre-run Review

## Review status

- SpecID: `TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT`
- Pre-run review: `PASS`
- Remote no-card gate: `PASS`
- GPU gate: `READY, AWAITING USER GPU MODE`
- Reviewed implementation commit:
  `067fd35c3a3a71f4905bcfc613d8492a301796a9`
- Frozen config SHA256:
  `064f4ee3a9cbfeacdd141c59e754cf1ca926249952cb3773014582a0402d1679`

This PASS authorizes only the already approved, bounded GPU P1 determinism
audit after the user opens GPU mode. It does not authorize P2/P4, victim
training, AP50, E20, E200, method tuning, or an automatic retry.

## Scope reviewed

- one first P1 batch and one candidate only;
- no accepted parameter update;
- no P2/P3/P4, poisoned-dataset export, victim training, AP50, E20, or E200;
- normal `shared/reset/fresh` pairs and strict `fresh` pair;
- one controller boot with a 300-second global watchdog and automatic shutdown;
- all growing remote paths constrained to the mounted data disk.

## Critical issues found and corrected during review

1. Fresh engines now restore the same calibrated engine counters and last real
   TAL assignment as reset replays. This makes fresh/reset initial-state hashes
   scientifically comparable.
2. The preparation path now replays the exact R4 read-only prelude: 16 `dist`
   calibration batches, four DG-CAIP/target warm-up batches, and 24 held-out
   observations. The expected engine observation count is frozen at 44 before
   the first P1 observation.
3. Strict deterministic-operator evidence no longer falsely claims that strict
   input/state validation completed before the operator error.
4. Tensor traces now preserve the original device in addition to CPU hash,
   shape, dtype, finite status, and numerical comparison fields.
5. Fresh/replay evidence now includes NLA total gradient, small row-space Gram,
   zero-update mutation names, full RNG/backend manifests, and CUDA/PyTorch/
   cuDNN/GPU identity.
6. The launcher now wraps the entire controller—not only child lanes—in a
   300-second timeout. All terminal states still invoke shutdown.

## Local evidence

- Python `compileall`: PASS for all new and modified audit Python files.
- Bash syntax: PASS for controller and tmux launcher.
- `git diff --check`: PASS for tracked changes (line-ending warnings only).
- Python AST parse: PASS.
- CSV schema: PASS, 28 columns and 17 unique task IDs.
- Scope scan: PASS; the runner contains no victim/AP50/E20/E200, checkpoint
  save, optimizer step, or multi-candidate backtracking call.
- R4 prelude contract scan: PASS.
- Focused P1 audit pytest on the exact remote checkout: `22 passed in 4.57s`.
- Adjacent regression pytest on the exact remote checkout: `59 passed in
  6.32s` (constraint router, semantic hiding carrier, DG-CAIP experiment,
  mechanism objective, E2E config, and non-target logit alignment).
- Both pytest commands returned exit code 0 under `OMP_NUM_THREADS=1` and
  `MKL_NUM_THREADS=1`. Without those bounds, a tiny carrier observation test
  exceeded 60 seconds because the large no-card host over-subscribed PyTorch
  CPU threads; the same test passed in 4.41 seconds with one thread. This is a
  test-runtime scheduling issue, not a method-code failure.

## Exact remote no-card evidence

- Detached sparse checkout:
  `/root/autodl-tmp/tausb-dgcaip/preflight-checkouts/cc55bf2-p1-det-audit`
- Remote HEAD exactly matched the reviewed implementation commit and
  `git status --porcelain --untracked-files=all` was empty.
- The sparse checkout contains only `ue_project/ue_framework`, focused tests,
  this pre-run bundle, and the tracked secret assets needed for input hashing.
- Frozen artifacts, surrogate checkpoint, secret manifest/tensors, hiding
  checkpoint, training image/label manifests, P1 state, P1 metrics, and D0
  report all passed the production validation path.
- Frozen dataset binding passed: 16,551 images, 16,551 labels, and 6,095
  person-containing images. The audit split contains 160 images and matched
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`.
- Frozen first batch IDs: `000321`, `000777`, `001362`, `001686`.
- The planned artifact root was absent; no `tausb-p1-det-audit` tmux session
  existed; `nvidia-smi` reported no devices, so no GPU work ran.
- Storage at review: system disk 21 GiB available; data disk 4.2 GiB
  available (92% used). The audit evidence cap is 100 MiB and all growing
  paths remain bound to the data disk. No cleanup was performed.

## Dirty-worktree isolation

Pre-existing R4 CSV/review/state/analysis evidence remains dirty or untracked.
It is user work and must not be included in the P1 audit snapshot. The future
selective snapshot may include only the files named by this Spec plus the four
default-off trace-hook modifications.

## Required next gate

1. Wait for the user to open GPU mode.
2. Reconfirm the same exact commit/config hash, absent artifact/control roots,
   mounted data disk, and available CUDA device.
3. Launch the reviewed controller exactly once. Its global hard cap is 300
   seconds and every terminal state requests shutdown.
4. Preserve and report the result even if the audit fails. Do not retry or
   continue into any effectiveness experiment automatically.
