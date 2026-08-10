# TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1 execution handoff

- Spec: `docs/research/specs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1.md`
- CSV: `issues/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1.csv`
- ExpID: `TAUSB-MALC-GRAD-GEOMETRY-S0`
- Branch: `codex/tausb-malc-grad-geometry-audit-v1`
- Base commit: `fe8697ab6fe00310db33182951fd8563dc301efc`
- Reviewed code snapshot: `18304b96c45360cfba5168d97d21d2961a13f390`
- Initial implementation snapshot: `354ad58c067968e3e4b6dd220bfdf262fea1fa7a` (superseded by the single-projector pre-run fix).
- Remote branch: pushed normally to `origin/codex/tausb-malc-grad-geometry-audit-v1`
- Approval: user explicitly approved on 2026-08-10.
- Active row: `PRERUN-REVIEW-03` (awaiting GPU mode).

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

## Current remote state and blocker

The authorized AutoDL no-card instance is reachable. `REMOTE-INPUTS-01` is complete: the
remote executable source/config scope is an exact clean copy of reviewed commit
`18304b96c45360cfba5168d97d21d2961a13f390`; frozen VOC counts and input hashes match;
the formal/control roots and tmux session are absent; and the cost-guard wrapper has matching
local/remote SHA-256 and passes remote `bash -n`. Detailed evidence is in
`research_workspace/experiments/TAUSB-MALC-GRAD-GEOMETRY-S0/pre_run/`.

`PRERUN-REVIEW-01` recorded the no-card blocker. The user then enabled GPU mode and
`GPU-ENABLE-01` passed with one idle NVIDIA GeForce RTX 4090 D, CUDA device count one, zero
reported memory use and no compute applications. `PRERUN-REVIEW-02` is now `pass / allow_run`:
the exact reviewed commit, source-scoped clean state, inputs, fresh roots, absent session,
shutdown executable and final wrapper hash
`7720af582f914b74fb63babea2d85fdf85dc711c167f95536a97346604cf464a` all pass. The wrapper
allows only import-generated `__pycache__/*.pyc` as benign untracked paths and snapshots all
seven required JSON files.

`REMOTE-GEOMETRY-01` then failed before the probe: the wrapper duplicated the prior-audit
schema check using nonexistent `semantic_bank_sha256` / `c2lm_basis_sha256` keys. The actual
prior JSON and the standalone gate use `semantic_bank_hash` / `c2lm_basis_hash`. The tmux
launch was accepted, SSH closed within the eight-second health window, and the single bounded
reconnect returned `Connection refused`, showing that the automatic shutdown path protected
GPU cost. No geometry, victim, materialization or AP50 stage was reached.

The correction changes only those two key names and switches retry control/session to an `r1`
suffix, preserving the old failure evidence. Corrected wrapper SHA-256 is
`06fd902397867482cbdb0fc12a9261455be06e8c5dd0b1dd9724be4f2dc8187d`.

`COST-GUARD-FIX-01` is now complete. In no-card mode, the old status/log were pulled and
confirm `failed/preflight` plus `KeyError: 'semantic_bank_sha256'`; the formal root and new r1
control/session remain fresh; the corrected wrapper passes bash/schema/hash checks against the
actual remote JSON; and CUDA is false with zero devices, proving no GPU probe was started. The
active row is `PRERUN-REVIEW-03`, which requires GPU mode to recheck the idle device and then
gate `REMOTE-GEOMETRY-02`.
