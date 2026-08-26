# TAUSB-SDH-DGCAIP-R3-DIAG-v1 review handoff

## Workflow state

- SpecID: `TAUSB-SDH-DGCAIP-R3-DIAG-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-R3-DIAG`
- User approval: explicit approval received in the active task on 2026-08-26.
- CSV: `issues/TAUSB-SDH-DGCAIP-R3-DIAG-v1.csv`
- CSV SHA256: `C9A0711098EEEEEB4276ACEDCF2AECEC6548EB178E3FA47028ABED4A14331857`
- CSV validation: header exact; 14 rows; no overflow columns; legal initial state enums; `PRERUN-REVIEW-01` precedes `REMOTE-DIAG-01`; final row is `REVIEW-01`.
- Current executable row: `SPEC-01`.
- Remote GPU: off and previously verified disconnected.

## Current blocker

The Codex Windows filesystem sandbox helper currently fails before reading any
existing file:

```text
windows sandbox failed: helper_unknown_error: setup refresh had errors
```

New files can be added with `apply_patch`, but every update of an existing file
fails before patch verification. This prevents truthful CSV state transitions
and prevents safe implementation changes to the existing DGCAIP code. No shell
write fallback was used because repository instructions require auditable patch
editing and the CSV must remain the single durable state source.

## Work completed

1. The approved Spec was read completely.
2. The mission routing, approved-doc, CSV execution, and coding-guideline skills
   were read completely.
3. The canonical CSV template and `ue_project/AGENTS.md` were read.
4. The 14-row execution CSV was generated with a structured serializer and
   validated.
5. No method code, runtime config, test, remote state, or historical artifact was
   changed.

## Recovery

After the local sandbox helper is restored:

1. re-read the approved Spec, this handoff, and the CSV;
2. update the Spec status from `draft_pending_user_approval` to `approved`
   with the approval date;
3. mark `SPEC-01` as in progress and then complete it with evidence;
4. continue rows in CSV order;
5. do not request GPU until `PRERUN-REVIEW-01` records a pass and an exact
   branch/commit.

## Claim boundary

No R3 implementation or experiment has run. No scientific or mechanism claim is
made by this handoff.

## Execution log — 2026-08-27

### Workspace recovery

- The original user workspace is preserved at `F:\autodl-tmp -1`.
- A clean worktree was created at `F:\tausb-r3-clean` on branch
  `codex/tausb-sdh-dgcaip-r3-diag-v1`, based on `6f2e44d`.
- The old task path is a junction to the clean worktree so the task-bound patch
  tool remains usable.
- Only the approved Spec, CSV, and this handoff were copied into the clean
  worktree. No user dataset, weight, artifact, or dirty source file was moved or
  deleted.

### Local implementation

- `SPEC-01`, `TRACE-01`, `ATTR-01`, `REPLAY-01`,
  `BINDING-01`, and `TEST-01` are closed.
- The heterogeneous backtracker has a default-off observational trace.
- R3 uses only P1-A, P1-B, P2-CAIP, and P4-DGCAIP.
- H1 emits exactly one pre-registered rejection label.
- H2 reuses the frozen `abs=1e-6`, `rel=1e-4` replay comparison and fails
  closed on initial-state or batch-sequence hash mismatch.
- The R3 config freezes seed 0, 16/24 calibration/held-out samples, batch 4,
  eight steps, `eps=16/255`, no EOT/JND, a 600-second cap, and a unique
  artifact root.
- The R3 path returns before the legacy P4 state-save block. No state is promoted.

### Local evidence

```text
python -m compileall -q <four changed method modules>
exit=0

validate_sdh_experiment_config(tausb_sdh_dgcaip_r3_diag_v1.yaml)
TAUSB-SDH-DGCAIP-R3-DIAG-v1 r3_diag 600

pytest -q test_dgcaip.py test_dgcaip_cgr.py test_dgcaip_diagnostics.py
          test_dgcaip_experiment.py test_dgcaip_r3_diagnostics.py
          test_bind_dgcaip_mechanism_config.py
41 passed in 3.66s

git diff --check
exit=0 (line-ending notices only)
```

### Current boundary

- These checks prove local mechanics and config binding only.
- No GPU mechanism run has occurred.
- H1 and H2 do not yet have experimental labels.
- `PRERUN-REVIEW-01` remains mandatory before requesting GPU.
