# TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1 execution handoff

- Spec: `docs/research/specs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1.md`
- CSV: `issues/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1.csv`
- ExpID: `TAUSB-MALC-GRAD-GEOMETRY-S0`
- Branch: `codex/tausb-malc-grad-geometry-audit-v1`
- Base commit: `fe8697ab6fe00310db33182951fd8563dc301efc`
- Code snapshot: `354ad58c067968e3e4b6dd220bfdf262fea1fa7a`
- Remote branch: pushed normally to `origin/codex/tausb-malc-grad-geometry-audit-v1`
- Approval: user explicitly approved on 2026-08-10.
- Active row: `REMOTE-INPUTS-01`.

## Frozen scope

This is a calibration-only diagnostic of the failed SIRC-MALC-CGR mechanism branch. It measures prototype stability, component and cross-batch gradient geometry, component-wise CGR survival, and one matched eight-step A0/A1 microtrajectory. It does not train a victim, materialize a poisoned dataset, compute AP50, add EOT, or alter MALC/CGR/carrier hyperparameters.

The ordered output is exactly one `first_bad_boundary`: `prototype_incoherence`, `cross_batch_malc_conflict`, `objective_gradient_conflict`, `cgr_selective_suppression`, `carrier_update_sink`, or `unresolved_by_probe`.

## Execution gates

1. Keep the existing v2 implementation and evidence read-only; add an independent probe path and artifact root.
2. Run focused local tests, compile/import/config/CLI validation, and graph-lifetime checks before any remote execution.
3. Create an exact-code Git snapshot and pass `pre-run-implementation-review` before starting AutoDL GPU work.
4. The remote probe is bounded to the approved 64-image calibration, 96-image held-out, warm-up 4, microtrajectory 8 protocol. Stop and shut down on non-finite values, invariant failure, or ten minutes without progress.
5. Surrogate-only probe evidence cannot support a fresh-victim or AP50 claim and cannot change Current Best.

## Worktree boundary

The worktree already contains unrelated user modifications and untracked research assets. Preserve them: no reset, stash, clean, bulk deletion, or broad staging. Only files attributable to this Spec may be committed.
