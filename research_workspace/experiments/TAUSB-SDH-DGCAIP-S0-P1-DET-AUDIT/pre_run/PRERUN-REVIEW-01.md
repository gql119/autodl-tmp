# TAUSB P1 Determinism Audit — Pre-run Review

## Review status

- SpecID: `TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT`
- Local review: `CONDITIONAL PASS`
- GPU gate: `BLOCKED`
- Blocking item: the local bundled Python runtime does not contain PyTorch,
  PyYAML, or pytest, so the focused no-card pytest suite has not yet executed
  in the real project environment.

`CONDITIONAL PASS` only means the implementation is ready to be snapshotted
for an exact remote no-card checkout. It is not permission to launch GPU work.

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
- Focused pytest: NOT RUN locally because required packages are absent.

## Dirty-worktree isolation

Pre-existing R4 CSV/review/state/analysis evidence remains dirty or untracked.
It is user work and must not be included in the P1 audit snapshot. The future
selective snapshot may include only the files named by this Spec plus the four
default-off trace-hook modifications.

## Required next gate

1. Re-verify that the selective exact HEAD contains only the audit whitelist
   and excludes all R4 evidence.
2. Obtain explicit authorization before normal non-force push.
3. On an exact clean remote checkout, run the focused audit tests plus existing
   DG-CAIP/SDH regressions in no-card mode.
4. Recompute config/commit hashes and change this review to `PASS` only if all
   remote tests and path checks pass.
5. Request a separate GPU gate; do not launch automatically from this review.
