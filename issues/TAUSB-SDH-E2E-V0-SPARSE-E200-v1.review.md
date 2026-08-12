# TAUSB-SDH-E2E-V0-SPARSE-E200-v1 Review Handoff

## Approved objective

Run one matched, fresh, seed0 paired E200 experiment using the frozen SDH P1 and sparse mixed-list materialization to determine whether the E20 person collapse persists at the full victim-training horizon while preserving the other 19 VOC classes.

## Frozen scientific boundary

- No change to secret, carrier, P1, D-LFC, CICR, CGR, NLA, person-bbox support, eps, dataset split, victim optimizer or AP50 computation.
- Surrogate: frozen VOC20 YOLOv8n checkpoint `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`.
- Victims: matched fresh YOLOv8n-style C0/M1, random initialization under seed0, no surrogate/E20 checkpoint inheritance and no resume.
- Frozen SDH state: `c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168`.
- P1 state: `2e102026a9356116de38acb1f5056bf5728afcd453e3447b516d4222f4d70b81`.
- Data: VOC train 16,551; M1 poisons exactly 6,095 person images and directly references 10,456 original JPEGs; clean VOC val 4,952.

## User amendments included in approval

- Overall GPU wall hard cap is 9 hours (`32400s`).
- Success, Failure and Inconclusive scientific outcomes must all be retained, pulled, entered into the ledger and reported to the user.
- Operational failure or timeout must retain every readable status, log, partial metric and reason before shutdown.
- Failure to meet the Success Signal is never authorization to delete results.

## Storage and cost boundary

- All growing paths, checkout, caches and temp files must be on mounted data disk `/root/autodl-tmp`.
- Data-disk reserve is 8 GiB after projected output; system disk must start with at least 4 GiB free and may not grow by more than 1 GiB across a stage.
- C0 and M1 train+evaluate each have a 3.5-hour hard cap; M1 materialization has a 20-minute cap.
- Ten obsolete system-disk worktrees were audited as removable, but repository safety rules require the user to remove directories manually.

## Current workflow state

- Spec: approved.
- Execution CSV: generated and structurally validated, 14 rows.
- `APPROVAL-01`: complete.
- `CONTROLLER-EPOCHS-01`: next.
- No E200 code snapshot, pre-run pass or GPU run exists yet.

## Dirty-worktree boundary

The local worktree contains unrelated user modifications and untracked research files. Preserve them without cleaning, stashing, resetting, overwriting or bulk deletion. Any future commit must explicitly stage only this Spec/CSV/review and task-specific implementation/tests.

## Append-only execution log

- 2026-08-12: user approved the Spec with the GPU wall hard cap revised from 8 to 9 hours and mandatory preservation/reporting of all terminal experiment outcomes.
- 2026-08-12: approved amendments were written into the Research Contract without changing Success/Failure thresholds.
- 2026-08-12: generated and validated the 14-row execution CSV; pre-run review precedes remote execution, and artifact pull/ingest/STATE/final review rows are present.
- 2026-08-12: implemented explicit E20/E200 binder/controller contracts, E200 IDs and roots, the 9-hour overall cap, per-arm 3.5-hour caps, and the frozen E200 scientific thresholds while retaining E20 compatibility.
- 2026-08-12: implemented pre-train fresh victim tensor hashes, C0/M1 hash pairing, C0 full-horizon sanity, data-disk/cache/tmp/system-disk gates, and an all-terminal-outcomes evidence manifest.
- 2026-08-12: added the one-shot data-disk E200 wrapper with an early shutdown trap and evidence flush before shutdown.
- 2026-08-12: local validation passed 106 tests, Python 3.8 AST, Git Bash syntax, a real VOC dataloader batch, and a real YOLOv8n init-hash probe. These are mechanical readiness evidence only, not E200 AP50 evidence.
