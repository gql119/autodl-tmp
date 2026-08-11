# TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3 Review Handoff

## Current workflow state

- Spec: `approved` on 2026-08-11 by explicit user message.
- Active CSV: `issues/TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3.csv`.
- Active row: `CGR-01`.
- Branch requested by the Spec: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`.
- The isolated hiding carrier and detector-native D-LFC are implemented locally. No GPU run,
  victim training, or AP50 result exists yet.

## Objective scientific result

Validate whether one fixed high-semantic building secret, encoded by a host-conditioned
hiding network inside every person GT box, creates a repeated detector shortcut that lowers
clean-val person AP50 while D-LFC/CICR stabilize the target effect and CGR plus explicit NLA
preserve the other 19 classes. A carrier-added person-val counterfactual and early-epoch
`R_e` audit are required before claiming shortcut learning.

## Frozen input evidence

- Exact local VOC root:
  `F:/dateset/VOC_0712_Kaggle_Ready/VOC_0712_Kaggle_Ready`.
- VOC audit:
  `research_workspace/sources/voc_input_audit_v1.json` (`overall_pass=true`).
- Split counts: train `16,551`; val `4,952`.
- Person-image counts: train `6,095`; val `2,007`.
- Image/label stems match exactly; no empty or invalid YOLO labels were found; all 20 class
  IDs occur in both splits.
- Four secret source hashes have zero exact SHA-256 overlap with all VOC train/val JPEGs.
- Secret manifest:
  `research_workspace/sources/secret_assets/manifest.json`.
- Final secret is exactly one image: `bg-building-sky-09`, source SHA-256
  `66bd89ebef12b21d578341e945d56ed315372b213e52c2cc07d2110543cc6a48`.
- The field, landscape, and tree images are pretrain-only anti-collapse inputs; they are not
  final person carriers.

## Active risks and blockers

- The TAUSB-SDH runtime entrypoint and remaining CICR/NLA/CGR integration are not implemented yet.
- The committed/remote asset policy is not yet reviewed; no asset or code has been staged.
- AutoDL data/secret/model hashes must be rechecked against the local manifests before GPU.
- Source low-frequency concentration does not prove generated-delta low-frequency energy or
  robustness.

## Frozen minimal integration map

The code audit found no design error. The smallest isolated implementation is:

1. Register `tausb_sdh` in `ue_project/ue_framework/config.py:7`, add method defaults near
   `ue_project/ue_framework/config.py:186`, and add formal fail-closed validation in
   `ue_project/ue_framework/config.py:258`.
2. Add a new `semantic_hiding_carrier.py` for DWT/coupling/reveal/bbox rendering. Do not reuse
   `instance_canonical_carrier.py`; that module is the abandoned Fourier/JND/pseudo-support
   carrier.
3. Reuse `YOLODetectTowerCapture` from
   `ue_project/ue_framework/methods/detector_tower_hooks.py:17` for frozen P3/P4/P5 cv3/cv2
   pre-logit features; add a small `detector_lfc.py` instead of an external ResNet.
4. Reuse and minimally extend `instance_classification_residuals`, calibration, and loss from
   `ue_project/ue_framework/methods/instance_cicr.py:66`, `:180`, and `:202`. Keep D-LFC and
   CICR as separate losses and diagnostics; do not route through MALC.
5. Add `non_target_logit_alignment.py` using real assignment tensors exposed by
   `ue_project/ue_framework/ultra/hijacked_loss.py:206`.
6. Preserve the existing single-tensor projector in
   `ue_project/ue_framework/methods/constraint_gradient_router.py:66`; add a flatten/unflatten
   wrapper for all `omega` parameters rather than silently projecting only the last layer.
7. Add an SDH-specific immutable state/materializer schema, following only the provenance and
   gate pattern of `ue_project/ue_framework/methods/sirc_malc_cgr.py:65` and `:225`. It must not
   import or resolve to SIRC/MALC.
8. Bind the materializer through `ue_project/ue_framework/methods/factory.py:8` and the normal
   generate loop at `ue_project/ue_framework/stages/generate.py:171`. Hiding/mechanism training
   uses a separate SDH workflow command; formal generate only loads a passing frozen state.
9. Extend existing strict per-class AP50 handling in
   `ue_project/ue_framework/stages/evaluate.py:465`; counterfactual/dynamics metrics remain
   explicitly secondary.

Existing `cicr.py`, `instance_cicr.py`, and `instance_canonical_carrier.py` appeared modified in
`git status`, but `git diff` is empty and their index/blob IDs are unchanged. They are treated as
stat-only dirty entries and will not be refreshed, reset, or staged merely to clean status.

## Pre-run decision

`pending`. No GPU command is authorized until local implementation, validation, Git snapshot,
remote input audit, and the matching pre-run review row pass.

## Final claim/evidence review

`pending`. Hiding/mechanism smoke evidence cannot support a fresh-victim UE claim. A single
seed, if eventually run, remains tentative.

## Append-only execution log

- 2026-08-11: User approved the single-building-secret Spec and requested first validation.
- 2026-08-11: Mission route selected
  `mission-doc-route -> mission-approved-doc -> mission-csv-execute`.
- 2026-08-11: Generated and structurally validated a 31-row CSV. The first generation had
  Windows pipe encoding loss and was replaced before any row execution; the valid CSV uses
  canonical status enums and contains no corrupted task fields.
- 2026-08-11: Persisted four authorized source images and deterministic center-square 256px
  assets with source/PNG/uint8/float32 hashes.
- 2026-08-11: Completed local VOC structure, label, class, person-count, and zero-overlap
  audit. No dataset file was modified.
- 2026-08-11: Completed `CODE-AUDIT-01`. Reuse is limited to detector tower capture, real TAL
  assignment, CICR, generic projection mathematics, base materialization interfaces, and strict
  AP50 reporting. Fourier/JND/pseudo carrier and MALC semantics remain excluded.
- 2026-08-11: Implemented the isolated hiding core in
  `ue_project/ue_framework/methods/semantic_hiding_carrier.py`. The formal architecture has four
  affine coupling blocks, width 64, 285,359 parameters, and architecture SHA-256
  `8812eb926f7b39637f9562a189d2aa001f3e45336e4e0aeb203954ac9929e7e6`.
  Six focused tests pass, covering Haar round trip, formal structure, host/secret dependence,
  freeze boundaries, exact bbox-union support, finite adapter backward, and overlap averaging.
  A read-only smoke on real VOC person crops `000009` and `000017` produced distinct finite
  `2x3x256x256` deltas with Linf `0.0146779`; this is mechanical evidence only.
- 2026-08-11: Added local hiding validation for SSIM/L1 retrieval, true-vs-wrong secret
  margin, cross-host pixel/RMS diversity, radial high-frequency energy, deterministic
  amplitude-preserving phase scramble, formal gate logic, and a finite pretrain step. The
  carrier+validation suite passes 11 tests and Python 3.8 AST parsing. A deterministic 60-step
  small CPU smoke reduced reveal loss from `0.3694704` to `0.3108912` with finite nonzero
  gradients. Formal recovery thresholds remain untested until the capped GPU hiding pilot.
- 2026-08-11: Implemented detector-native D-LFC on the frozen YOLO P3/P4/P5 `cv3`
  pre-logit towers. It consumes canonical perturbation crops, uses calibration-only frozen
  prototypes, weights all three scales equally, retains gradients only to the carrier, and
  fails closed on zero/constant inputs. The D-LFC and related carrier/tower suite passes 19
  tests. No ResNet-18 or ImageNet feature extractor is used.
- 2026-08-11: Added a calibration-only frozen instance-CICR bank while preserving legacy
  CICR behavior for existing methods. CICR uses clean-to-poison classification-tower
  residuals grouped by real person GT assignment, equal-weights instances, and separates
  direction alignment from a differentiable residual-energy floor. Twenty-one focused and
  regression tests pass; held-out prototype updates and degenerate calibration fail closed.
- 2026-08-11: Implemented explicit non-target logit alignment on clean real-TAL positives.
  The loss aligns only the assigned class raw logit, detaches the clean teacher, excludes
  person/background/pseudo locations, macro-averages active non-target classes, and records
  full-vector drift only as a diagnostic. A one-shot warm-up gradient-norm calibration freezes
  `lambda_nla` at a 0.25 projected-target ratio. Fifteen focused/regression tests pass.
- 2026-08-11: Extended CGR over every trainable adapter tensor. Target gradients are projected
  through the per-class NLA SVD nullspace (`rtol=1e-4`), followed by explicit NLA descent and at
  most five nonlinear backtracks; unsafe proposals skip. Seventeen focused/regression tests pass.
- 2026-08-11: Bound `tausb_sdh` to strict config, immutable frozen-state loading, GT-box-only
  materialization, hiding/mechanism pilots, and the normal generation factory. A real read-only
  VOC `000009` forward used clean TAL with 9 person positives and 40 total foreground locations;
  P3/P4/P5 produced three 64-D person residuals each, Linf was `0.0285735`, outside-box change
  was zero, and all four adapter gradients were finite/nonzero. This is mechanical evidence,
  not a fresh-victim AP claim.
- 2026-08-11: Added strict named VOC20, 19-class macro, GT-box target-carrier recovery, and
  epoch 1/5/10/20 learning-preference arithmetic. Evaluation now rejects mismatched secret,
  source-manifest, train-split, support, or failing mechanism evidence. Six focused evaluation
  tests pass; the runtime counterfactual entrypoint remains the open part of `EVAL-01`.
- 2026-08-11: Completed the counterfactual runtime entrypoint. It losslessly materializes a
  PNG target-carrier val view only inside person GT boxes, evaluates the frozen P1-V, audits
  epoch 1/5/10/20 on one deterministic balanced subset using a single clean real-TAL assignment
  for both views, and records a scale/aspect-matched person-free transplant separately. None of
  these paths backpropagates or selects checkpoints. Nine evaluation tests pass. A real VOC
  `000009` CPU smoke found 30 fixed person positives and finite losses (`clean=4.3215022`,
  `carrier=4.3113022`); this checks mechanics only and is not evidence of faster learning.
- 2026-08-11: Completed local validation. The focused SDH suite passes 55 tests and the
  touched legacy CGR/CICR/SIRC/MALC paths pass 26 regression tests (81 total). Compile/import,
  Python 3.8 AST, formal/mechanism config parsing, CLI help, 31-row CSV structure, finite
  gradients, deterministic rendering, and real VOC TAL/loss smokes pass. GPU hiding,
  held-out mechanism gates, fresh-victim training, and all AP50 outcomes remain unrun gaps.
- 2026-08-11: Created dedicated branch
  `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3` and code snapshot `a781e1f`. The commit contains
  41 approved task files, including the four user-authorized source carriers and deterministic
  256px derivatives (<1 MiB total), but excludes VOC, weights, checkpoints, smoke artifacts,
  credentials, temporary outputs, and every unrelated dirty-worktree file.
- 2026-08-11: Normal non-force push succeeded; remote branch tip was `5f61c00` and the code
  snapshot remains `a781e1f`. The first read-only AutoDL input-audit connection to the
  user-authorized endpoint returned `Connection refused`; no remote command, GPU process, or
  paid experiment started. `REMOTE-INPUTS-01` remains the recoverable active row pending an
  online instance or updated SSH endpoint.
- 2026-08-11: Resumed `REMOTE-INPUTS-01` in no-card mode. A structural CSV audit found and
  repaired five notes fields whose unquoted commas had shifted later columns; the corrected
  CSV parses as 31 rows with exactly 28 fields each.
- 2026-08-11: The remote input audit exposed a real cross-platform provenance bug: the secret
  manifest was hashed as raw bytes, so Windows CRLF and Linux LF checkouts disagreed. Commit
  `7cd60f9` now hashes canonical parsed JSON and freezes
  `a25277499e07310e68a39277461f176dd0d8666e69a4b890328d7b913601ac3e`.
  A CRLF/LF regression test was added; the focused SDH suite now passes 56 tests.
- 2026-08-11: Local and AutoDL audits independently matched all VOC counts, all 20 per-class
  instance counts, train/val person-image counts `6095/2007`, canonical label-content hashes,
  image path-size hashes, all four authorized secret source hashes, and surrogate SHA-256
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`.
  Hashing all 21,503 remote VOC images found zero secret overlap. The agent-owned worktree is
  clean at `7cd60f9366652a60df0830579a25cd2ef54a13e9`; the formal artifact root is fresh; tmux and
  `/usr/bin/shutdown` are executable. CUDA remained unavailable and no GPU job started.
  AutoDL lacks pytest, so the active Linux runtime path was validated directly: mechanism
  config, canonical manifest hash, and `(4,3,256,256)` secret-bank loading all passed.
- 2026-08-11: The first hiding pre-run review was `blocked/do_not_run`: the mechanism YAML
  still resolved VOC and surrogate inputs relative to the clean checkout, where those large
  inputs are intentionally absent. `PRERUN-FIX-PATHS-01` changed only those two values to the
  audited AutoDL absolute paths. Commit `e3f6744` passed seven local config tests and remote
  no-card existence/checkpoint-hash validation from a clean exact checkout.
- 2026-08-11: The second hiding pre-run review was also blocked before GPU: direct file
  execution of `ue_framework/tools/run_tausb_sdh.py` failed to resolve the current checkout's
  package. `PRERUN-FIX-ENTRYPOINT-01` binds the wrapper to
  `python -m ue_framework.tools.run_tausb_sdh` from the reviewed project root. Module help now
  passes locally and remotely; this avoids importing an absent or stale installed package.
- 2026-08-11: Prepared `PRERUN-REVIEW-HIDING-1` with result `pass/allow_run`, conditional on
  its launch-time GPU gate. The reviewed code commit is `e3f6744`; wrapper SHA-256 is
  `99fe8cbcac2a82d8af20b7df5f165688a1b090c586261233e3ec231e3c3f6419`; launch-gate SHA-256
  is `39a07c572554e1b0e9b4199a20405118ded9369fec9a51e2299d6c5329ca8baa`.
  Both scripts pass remote `bash -n`. The wrapper invokes hiding only, enforces an external
  1200-second timeout and idle watchdog, snapshots minimal evidence, and requests shutdown on
  success or failure. CUDA remains unavailable and no experiment has started.
- 2026-08-11: Pushed pre-run packet commit `59253ba`, then uploaded the reviewed wrapper and
  launch gate to new commit-suffixed AutoDL paths. Remote bytes match the frozen hashes and
  both files pass `bash -n`. `PRERUN-HIDING-01` is closed. The only authorized next command is
  `/bin/bash /root/verify_and_launch_sdh_hiding_e3f6744.sh` after GPU mode is enabled; executing
  any inner Python or tmux command directly would bypass the reviewed cost guard.
- 2026-08-11: User enabled GPU mode. Selected `REMOTE-HIDING-01` and recorded the frozen
  execution owner, code commit `e3f6744`, session, log, artifact root, 20-minute cap and the
  only allowed launch-gate command before remote execution. No inner command is launched
  outside the reviewed gate.
- 2026-08-11: The reviewed launch gate passed on an idle NVIDIA GeForce RTX 4090 D: exact
  commit and wrapper hash matched, both output roots were fresh, CUDA/config inputs passed,
  and tmux session `tausb-sdh-hiding-s0-e3f6744-r1` started. Both the immediate and one
  bounded follow-up health check then received SSH connection refusal. This is consistent
  with the wrapper's automatic shutdown path, but the run outcome is deliberately recorded
  as uninspected until no-card recovery can read the control log/status. No rerun was made
  and no hiding completion or metric claim is recorded.
- 2026-08-11: No-card recovery pulled and hash-verified exactly three r1 failure files with
  no missing required evidence. The cost guard records `failed / hiding_pilot` and the formal
  hiding artifact root was never created. The traceback reaches the first failed boundary in
  `_first_person_host`: `Person host crop is empty.`
- 2026-08-11: A read-only probe on the exact deterministic split found that
  `_first_person_host` passes `(image height, image width)` to `_bbox_to_pixels`, whose contract
  is `(width, height)`. This makes three of 96 held-out slices empty (`000805`, `000829`, and
  `2009_000337`) while calibration happens to have none. Passing the dimensions in the
  documented order yields zero empty crops across all 64 calibration and 96 held-out hosts.
  The failure is therefore a deterministic coordinate-order bug, not an invalid VOC label.
  `REMOTE-HIDING-01` is closed as a preserved failed attempt; `HIDING-CROP-FIX-01` is selected,
  followed by a new pre-run review and a distinct r2 remote retry. No GPU retry is authorized
  until those gates pass.
- 2026-08-11: Implemented the source-level fix as one argument-order change only; no host is
  skipped and the split/filter protocol is unchanged. Three synthetic regressions matching the
  failed portrait/landscape edge boxes pass, the focused SDH suite passes 66 tests, the shared
  support regression passes, and the three actual local VOC files now return finite
  `3x256x256` hosts. A local full-dataset scan reached its 120-second I/O cap without returning
  a result, so it is not counted as passing evidence; the exact full 160-host check remains a
  required no-card validation on the fixed AutoDL checkout before retry review.
