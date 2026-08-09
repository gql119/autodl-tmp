# TAUSB-SIRC-MALC-CGR-MAP50-v2 Draft Handoff

## Current state

- Research scope: authorized academic machine-learning research on public YOLOv8/Pascal VOC
  and user-controlled code/AutoDL. It does not involve unauthorized access, network scanning,
  exploitation, credential collection, malware, persistence, access-control bypass, or service
  disruption.
- Current Git branch: `codex/tausb-sirc-lfc-cgr-map50-v1`.
- Base HEAD: `6262d918f6d2355757c1b8e98e1d6728cf005b69`.
- New draft Spec:
  `docs/research/specs/TAUSB-SIRC-MALC-CGR-MAP50-v2.md`.
- Approval status: `approved` by explicit user message on 2026-08-09.
- The v2 execution CSV is generated after this approval and becomes the only active durable
  execution source; the old v1 CSV remains paused for audit.
- No v2 method code, local validation, AutoDL training, evaluation, artifact generation,
  deletion, or overwrite has been performed.
- The previous approved v1 Spec remains on disk for audit but has
  `execution_state: paused_for_method_revision` and points to this v2 candidate.
- The previous v1 CSV row `CARRIER-01` remains `进行中` only as historical durable state;
  its notes contain `do_not_execute:pending user approval of TAUSB-SIRC-MALC-CGR-MAP50-v2`.

## Why v1 was revised

The original Semantic Deep Hiding LFC is a classification-oriented module. It computes
pairwise cosine distance between flattened final-convolution ResNet-18 features of perturbation
maps `x_pm=x_ue-x_c`, with paper weight `omega_3=1e-4`. It does not model object instances,
clean TAL assignment, P3/P4/P5 scale, co-occurring classes, or detector head structure.

Directly adding an external ResNet-18 LFC would create task mismatch, an extra checkpoint,
and an objective partly duplicated by the existing detector residual CICR. The user requested
a detector-native redesign rather than a backbone substitution.

## Frozen v2 proposal: MALC

MALC means **Multi-scale Assignment-aware Latent Concentration**.

1. Keep one shared SIRC carrier family: 16 bases / 48 RGB coefficients, radial range `[2,24]`,
   `eps=16/255`, instance-canonical rendering, deterministic JND, and forced pseudo fallback
   support.
2. Freeze the existing YOLOv8 surrogate. Only carrier coefficients `theta` are updated.
3. Capture P3/P4/P5 `cv3` classification-tower features immediately before their final class
   convolutions. The `cv2` box tower is monitoring-only.
4. Clean real TAL supplies foreground, assigned target scores, and `target_gt_idx`. PAG selects
   person-relevant detector locations.
5. For each person GT and scale, pool the clean-to-poison `cv3` residual with normalized clean
   assigned-score weights.
6. Fit per-scale direction prototypes, median residual energy, and `0.5 * Q25` energy floors
   only on the fixed calibration split; freeze all of them before optimization/held-out use.
7. MALC uses scale-balanced, instance-balanced terms:
   - residual direction concentration;
   - log residual-magnitude concentration;
   - non-zero energy floor.
8. MALC replaces both external ResNet-18 LFC and standalone CICR. Existing
   `instance_cicr.py` is an implementation starting point, not a second simultaneous loss.
9. Loss weights are not copied from the paper or all set blindly to 1.0. A fixed warm-up
   calibration matches median coefficient-gradient norms to the easy-classification route,
   clips ratios to `[0.1,10]`, then freezes them without using victim mAP.

## Non-target protection

CGR remains the only non-target protection mechanism:

- clean TAL real non-target foreground positives only;
- one class-balanced assigned-class probability-drop constraint per active class;
- tolerance and near-boundary both `0.005`;
- row-normalized constraint gradients and SVD relative threshold `1e-4`;
- null-space projected target step near the boundary;
- repair-only when violated;
- at most five nonlinear backtracks, then skip;
- no scalar non-target distillation, non-target feature loss, ALCE, or late repair;
- non-target box/CIoU is monitoring-only.

## Minimum experiment

Mechanism gate, not UE evidence:

- A0: shared SIRC + easy-classification route + CGR, MALC off.
- A1: exactly matched A0 with MALC on.
- Fixed calibration/held-out split; held-out never updates prototypes or weights.
- A1 must pass the frozen direction, magnitude, energy, scale-coverage, leakage and CGR
  retention gates before M1 is allowed.

Fresh-victim experiment after the mechanism gate:

- C0: clean VOC train, fresh YOLOv8n victim.
- M1: A1 carrier materialized on all 6,095 person-containing train images, followed by an
  independent fresh victim.
- Matched protocol: seed 0, 200 epochs, image size 640, batch 36, SGD, original clean VOC val.
- No EOT or JPEG/blur/gray robustness evaluation in this first run.

## Main success criteria

- Mechanism A1 vs A0: held-out level-balanced residual cosine median gain at least `0.10`;
  log-energy MAD at most `0.90x`; coverage at least `0.80`; zero-norm at most `0.20`;
  floor-pass at least `0.80`; at least two scales remain valid; CGR attack-retention median
  at least `0.20`; repair-plus-skip below `0.50`.
- Fresh victim: person AP50 drop at least `0.30`; non-target macro AP50 drop at most `0.05`;
  at least 16/19 non-target classes drop no more than `0.10`; poisoned count exactly 6,095.
- Single-seed evidence remains tentative. A0/A1 mechanism evidence cannot prove UE efficacy.

## Dirty-worktree boundary

The worktree contains pre-existing user changes and many untracked research files/artifacts.
Do not reset, stash, clean, bulk-delete, or stage unrelated paths. Tracked changes already
exist in `launch_one.py`, `paths.py`, `runtime.py`, `stages/aggregate.py`,
`stages/evaluate.py`, and `stages/train_victim.py`; audit and preserve them.

## Instructions for the next window

1. Read `AGENTS.md`, `ue_project/AGENTS.md`, `research_workspace/STATE.md`, this handoff,
   the approved v2 Spec, the v1 review, and the old v1 CSV.
2. Do not resume the v1 CSV. The approved v2 CSV is the only active execution source.
3. Continue with `mission -> mission-csv-execute` on the v2 CSV; do not mutate the old v1
   CSV into v2.
4. Continue on `codex/tausb-sirc-malc-cgr-map50-v2` while preserving the dirty worktree.
5. Implement in order: carrier/render -> MALC -> gradient calibration -> CGR integration ->
   A0/A1 mechanism gate -> pipeline/config -> named VOC20 AP50 -> local/no-card validation ->
   scoped Git snapshot -> remote-input audit -> pre-run review -> GPU execution.
6. Never launch formal training before `pre_run_result=pass`, exact branch/commit and commands
   are frozen, all input hashes exist, roots are fresh, and a GPU is available.

## Pre-run decision

`not_applicable_yet` -- implementation and local validation have not reached the pre-run row.

## Final claim/evidence review

`pending` -- no v2 implementation or experiment evidence exists.

## Append-only log

- 2026-08-09: user stopped/cancelled the parallel ResNet-18 work and requested a detector-native
  improvement rather than a direct LFC transplant.
- 2026-08-09: original paper equations and the active YOLO tower/TAL/CICR/router code paths were
  audited.
- 2026-08-09: draft `TAUSB-SIRC-MALC-CGR-MAP50-v2` created and self-reviewed; old v1 execution
  explicitly paused and gated against accidental recovery.
- 2026-08-09: cross-window handoff created; no method code or remote run had started.
- 2026-08-09: user explicitly approved `TAUSB-SIRC-MALC-CGR-MAP50-v2`; Spec status changed
  to `approved`, the dedicated branch was created, and CSV-driven execution was authorized.
- 2026-08-09: generated and structurally validated the dedicated 21-row v2 execution CSV;
  `CARRIER-01` was selected first. Carrier audit found the existing shared 48-coefficient,
  four-variant implementation consistent with the frozen SIRC design. Added fail-closed
  validation for non-finite/non-positive carrier parameters and deterministic-JND regression;
  the focused carrier/render/support suite passed `17 passed`. `MALC-01` is now active.
- 2026-08-09: implemented detector-native MALC primitives in `malc.py`: clean-score-weighted
  per-instance residual pooling, frozen normalized scale prototypes, equal-scale direction,
  log-magnitude and non-zero-floor losses, and explicit coverage/energy/scale diagnostics.
  No external ResNet or duplicate CICR loss was introduced. The combined MALC, legacy CICR,
  tower-hook and carrier regression suite passed `30 passed`; `CALIBRATION-01` is now active.
- 2026-08-09: implemented deterministic MALC prototype and gradient-norm calibration. Prototype
  updates accept only the calibration split and freeze per-scale direction/median/Q25-floor
  values with a reproducibility hash. Loss weights are immutable after one-time calibration;
  disconnected, zero/non-finite and excessive-clipping paths fail closed. The focused suite
  passed `18 passed`; `CGR-INTEGRATION-01` is now active.
- 2026-08-09: added the MALC-CGR integration boundary. Only per-class classification-probability
  margins become normalized CGR rows; box/CIoU margins remain diagnostics. Composite target
  gradients use SVD projection or repair-only and every class-constrained candidate is checked
  by the actual nonlinear evaluator with five backtracks before skip. The combined routing,
  selective-route and MALC suite passed `29 passed`; `MECHANISM-HARNESS-01` is now active.
- 2026-08-09: implemented the matched A0/A1 held-out mechanism harness. It enforces a single
  MALC-switch difference, aggregates level-balanced cosine/energy/coverage/scale/CGR/leakage
  metrics, retains size and person-only/co-occurrence groups, applies every frozen Success and
  Failure Signal independently, and writes a no-overwrite JSON explicitly scoped as mechanism
  evidence only. The focused suite passed `24 passed`; `PIPELINE-01` is now active.
- 2026-08-09: completed `PIPELINE-01`. The canonical entrypoints now bind the approved chain
  `config -> frozen SIRC bank -> YOLO cv3 -> clean TAL/PAG -> score-weighted MALC -> per-class
  CGR -> 48 coefficient update -> gate-passed A1 state -> deterministic materializer`. The
  mechanism runner starts A0/A1 from one route-warm state and one frozen prototype/gradient
  calibration; the arm configs may differ only at `method.enable_malc`. It never calls the EOT
  path. Formal M1 rejects a failed gate, A0 state, malformed/tampered content, mismatched
  semantic-bank/source/split hashes, partial feature switches, EOT, support leakage and budget
  overflow. All v2 switches off dispatch the exact legacy `tausb_mask` method config. Fresh
  entrypoints now fail closed on existing roots instead of deleting them. Formal victim settings
  are seed 0, 200 epochs, imgsz 640, batch 36 and SGD. The focused pipeline/config/MALC/CGR/SIRC
  suite passed `55 passed`; direct frozen-state/materializer tests passed `15 passed`. No
  mechanism or fresh-victim experiment has been run. `EVAL-01` is now active.
- 2026-08-09: completed `EVAL-01`. Train/eval dataset YAMLs now use the canonical VOC20 class
  names and order. Full-val AP50 is mapped through explicit Ultralytics `ap_class_index`; missing,
  duplicate, out-of-range, non-finite, unit-invalid or map50-inconsistent mappings fail closed.
  Metrics contain the 20-name AP50 mapping, person AP50, all 19 non-target AP50 values, the
  non-target macro, all-class mean, poisoned count, actual Linf and explicit PSNR/LPIPS gaps.
  Aggregation writes an exact C0/M1 per-class table with both delta directions and retention,
  checks C0 count 0/M1 count 6095, and evaluates the frozen single-seed success rules while
  labeling the result `tentative_single_seed`. M1 provenance requires one carrier-state hash,
  matching bank/split hashes, valid deterministic variants and only forced-pseudo fallback
  support. The focused implementation/metrics suite passed `61 passed`; no validation metric or
  fresh-victim result has been generated. `LOCAL-VALIDATION-01` is now active.
- 2026-08-09: tightened the formal M1 evidence chain before closing local validation. Every
  poisoned manifest row now carries the frozen source-manifest hash in addition to the state,
  semantic-bank and split hashes, and evaluation rejects any mismatch. M1 `metrics.json` now
  embeds the immutable A0/A1 held-out diagnostics only after validating the report schema,
  evidence scope, split hash, passing gate and `allow_fresh_victim=true`; C0 and legacy arms
  record the mechanism evidence as not applicable.
- 2026-08-09: completed `LOCAL-VALIDATION-01`. The entire local suite passed `134 passed`;
  the 12 active v2 entry files passed compile and Python-3.8 grammar parsing, and module import
  plus formal config parsing resolved person id 14, no-EOT and 200 victim epochs. Read-only VOC
  checks found exactly 16,551 train and 4,952 val images, and the surrogate checkpoint hash
  matched the frozen contract. A real-VOC CPU smoke exercised frozen YOLO/TAL/PAG, MALC
  calibration, A0/A1 coefficient gradients, CGR projection and held-out aggregation. All ten
  JSON artifacts were finite; projected-row dots were about `1e-8` and both updates were
  accepted. As expected for the one-step non-evidentiary smoke, the scientific mechanism gate
  did not pass (`allow_fresh_victim=false`) and no frozen carrier was written. This is not a UE
  result or a formal mechanism failure. Actual Python-3.8/AutoDL runtime availability remains a
  read-only `REMOTE-INPUTS-01` check. `GIT-SNAPSHOT-01` is now active.
- 2026-08-09: the Git-boundary audit recovered prior AutoDL evidence that Python 3.8 evaluated
  the ordinary type-alias assignment `BandRange = tuple[float, float]` at import time despite
  postponed annotations. This was the only top-level PEP-585 alias in `ue_framework`; it was
  changed to `typing.Tuple[float, float]`. The complete `134`-test suite passed again. Actual
  AutoDL import remains a mandatory read-only recheck, but the previously observed blocker is
  now corrected rather than hidden behind a grammar-only claim.
- 2026-08-09: created the scoped implementation snapshot
  `d85967ed070340718b8e805fb59021f560f1eb71` on
  `codex/tausb-sirc-malc-cgr-map50-v2`. The commit contains exactly 35 audited v2 Spec/CSV/review,
  source, formal-config and test files. Secret and binary-artifact scans passed. It excludes
  `.tmp`, the local smoke config, datasets, weights, credentials, old v1 task state and all
  unrelated dirty-worktree files. Push remains pending exact remote payload authorization.
