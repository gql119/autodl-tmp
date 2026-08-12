# TAUSB-SDH-E2E-V0-SPARSE-E20-v3 execution handoff

## Approval and scope

- Status: approved by the user on 2026-08-12.
- Spec: `docs/research/specs/TAUSB-SDH-E2E-V0-SPARSE-E20-v3.md`.
- ExpID: `TAUSB-SDH-E2E-V0-S0-E20-SPARSE`.
- RunID: `SPARSE-E20-S0-R1`.
- Branch: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`.
- Starting HEAD: `fbaa19d289f253994ad0ef15de67f93bc3693062`.

## Frozen scientific protocol

- Reuse the already verified P1 state; do not rerun P1/mechanism or paired smoke.
- Keep the fixed semantic secret, person-GT-bbox support, D-LFC, CICR, CGR and NLA unchanged.
- C0 and M1 are independent fresh YOLOv8n-style victims, seed 0, 20 epochs, clean VOC20 AP50.
- M1 poisons all and only 6,095 person-containing train images; 10,456 person-free images remain byte-identical source JPEG references.
- This is a single-seed, 20-epoch feasibility experiment, not a robustness, multi-seed or SOTA claim.

## Implementation boundary

- Add a V0-only sparse materialization protocol and explicit mixed-image path list.
- C0 must not duplicate image files. M1 may store only target PNGs and necessary target labels.
- Keep legacy directory-based data generation/training unchanged.
- Add manifest provenance, saved-reload perturbation metrics, list/label/count/hash gates, cost/disk accounting and auto-shutdown orchestration.
- Do not modify carrier, P1, loss weights, D-LFC, CICR, CGR, NLA, target class, split, optimizer, AP50 computation or formal 200-epoch code paths.

## Existing evidence and rationale

- R4 proved the old P1→materialize→fresh victim→clean VOC20 path mechanically works.
- R4 did not run E20 because the former estimator incorrectly multiplied fixed smoke overhead by dataset scale and epochs, yielding 59.29 hours.
- Historical full-VOC evidence places a 20-epoch victim curve near 15.96 minutes per arm; the approved paired cap is 2 GPU-hours.
- Current generation writes every image as PNG. Sparse mixed lists remove the unnecessary 10,456 clean-image rewrites per M1 arm and all C0 image duplication.

## Dirty-worktree boundary

- The starting worktree contains unrelated modified and untracked user files.
- Preserve them; do not clean, stash, reset, overwrite or bulk-delete anything.
- Any commit must explicitly stage only this Spec/CSV/review and task-specific code/tests.

## Current execution state

- `APPROVAL-01`: complete.
- `SPARSE-IMPL-01` through `COST-CONTROLLER-01`: complete.
- `LOCAL-VALIDATION-01`: complete; remote no-card audit remains a truthful validation gap because the instance refused the SSH connection.
- `GIT-SNAPSHOT-01`: in progress.
- No GPU run has been authorized by a passing pre-run review for this Spec.

## Append-only execution log

- 2026-08-12: user explicitly approved the Spec.
- 2026-08-12: approved status recorded; 13-row execution CSV and this handoff created and validated.
- 2026-08-12: implemented V0-only sparse mixed-list materialization, train-list consumption, saved-reload quality/provenance gates and an E20-only controller.
- 2026-08-12: real local VOC audit found 16,551 train images, 6,095 person images, no missing/corrupt labels, and a valid Ultralytics batch.
- 2026-08-12: 64-image PNG round-trip gate passed; full SDH focused regression reached 91 passed.
- 2026-08-12: actual target-subset disk sampling projects 3,727,519,779 new bytes and 6,948,745,251 bytes including the 3 GiB reserve, replacing the invalid 29.79 GB estimate.
- 2026-08-12: 92/92 SDH tests, Python 3.8 AST for eight active files, Python 3.12 compile, controller CLI, Bash syntax and diff checks passed; no GPU job was run.
- 2026-08-12: added and tested the C0 all-zero AP50 stop gate; an uninterpretable clean control now prevents M1 training and exits through the guarded shutdown path.
- 2026-08-12: AutoDL no-card compatibility probe returned connection refused; recorded as a pre-run validation gap rather than retried or treated as pass.
- 2026-08-12: exact execution snapshot `b70fc87ecfcda8c2adb5f40b86a1147dbe738633` was committed and ordinarily pushed; the GitHub branch resolved to the same SHA and unrelated dirty files remained uncommitted.
- 2026-08-12: `PRERUN-REVIEW-01` found the local implementation and active data/metric chain ready, but recorded `blocked / do_not_run` because the AutoDL endpoint remained offline and remote input/fresh-root/GPU/disk/shutdown gates could not be observed.
- 2026-08-12: after GPU enablement, exact commit fetch and a clean detached worktree succeeded. Remote CUDA/runtime/P1/disk/GPU/fresh-root checks and the full 16,551-image Ultralytics 8.4.33 sparse-list probe passed; `PRERUN-REVIEW-02` is `pass / allow_run` for the exact recorded controller contract.
