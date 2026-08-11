# TAUSB-SDH-E2E-V0-MAP50-v1 execution review

## Current workflow state

- Approved Spec: `docs/research/specs/TAUSB-SDH-E2E-V0-MAP50-v1.md`.
- Durable state source: `issues/TAUSB-SDH-E2E-V0-MAP50-v1.csv`.
- Active row: none; next gated row is `REMOTE-MECH-01`, which requires GPU mode.
- GPU state: no GPU job started; the required no-card audit completed and the instance is being shut down after evidence persistence.
- Branch: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`.
- Reviewed code snapshot: `3a7a1aa`.

## Objective scientific result

Obtain one real, paired, single-seed 20-epoch VOC AP50 result for the current complete SDH detector-aware method before any further carrier tuning. The result is directional feasibility evidence only, not a formal unlearnable-example claim.

## Active risks and blockers

- The r2 hiding checkpoint failed the original RMS-diversity and high-frequency scientific gates. Those failures must remain visible in every feasibility artifact.
- The formal `tausb_sdh` loader and 200-epoch protocol must remain fail closed.
- The next run must use the exact mechanism contract committed in `27e434d`; it may not substitute the old dirty repository or start victim/smoke/E20 stages.
- External prerequisite: AutoDL GPU mode must be enabled before `REMOTE-MECH-01` can launch.
- The worktree contains unrelated user changes; only explicitly scoped V0 files may be staged later.

## Pre-run decision

`pass / allow_run` for `REMOTE-MECH-01` only, bound to code commit `3a7a1aaff912d0904794a91a4d3512d18b5c69fa`, config SHA-256 `46f757af...c56`, and the clean detached worktree recorded below. This does not authorize any victim, smoke, E20, or AP50 stage.

## PRERUN-REVIEW-1

- Result: `blocked`.
- Decision: `do_not_run`.
- Gated run: mechanism only; no victim training, materialization, smoke, or AP50.
- Code snapshot: branch `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`, commit `3a7a1aa`.
- Intent: load the exact failed-gate r2 carrier, run unchanged T0/T1/P0/P1 for 8 steps, and save the actual finite P1 feasibility state while preserving real diagnostic flags.
- Code location: `ue_framework.tools.run_tausb_sdh` → `run_mechanism_pilot` → exact r2 loader → detector observation/objective/CGR/NLA → `p1_feasibility_sdh_state.pt`.
- Parameter data flow: the mechanism config fixes VOC20, person=14, 16/255, 8 steps, no EOT/JND, r2 hashes, dataset manifests, secret hashes, and surrogate hash before the observation engine is built.
- Runtime state: carrier detector adapters are the optimization parameters; surrogate is used in eval mode; D-LFC/CICR calibration banks and NLA calibration are frozen before the four-arm trajectory.
- Sink effect: V0 always preserves `hiding_gate_passed=false` and the actual `mechanism_gate_passed`; formal PASS-only state emission remains unchanged.
- Baseline/disable path: T0/T1/P0/P1 switches remain independent; non-V0 configs still reject failed hiding gates and require formal state gates.
- Local validation: 105 related tests passed; compile, Python 3.8 AST, config parse, CLI, and diff check passed.
- Minimal probe: the new manifest implementation recomputed the real local VOC train hashes exactly: 16,551 images `4954727d...8fbd` and 16,551 labels `3cd05ad1...d848`.
- Run command binding: pending current remote audit and the final cost-guarded tmux snippet.
- Experiment validity: code now verifies complete train image/label manifests, 6,095 person images, r2 metrics/checkpoint, secret source/manifest/tensor, and surrogate checkpoint before optimization.
- Output non-overwrite: code uses `mkdir(..., exist_ok=False)` for the mechanism output; current remote root freshness remains unaudited.
- Recoverability/secrecy: no credentials are stored; tmux/log/status/shutdown availability remains to be checked remotely.
- Blockers: SSH to the configured AutoDL endpoint returned `connection refused`.
- Validation gaps: current remote branch/commit, input files, output root, GPU, environment, disk, tmux, and shutdown executable.

## PRERUN-REVIEW-2

- Result: `pass`.
- Decision: `allow_run` for `REMOTE-MECH-01` only.
- Gated run: mechanism-only `MECH-V0-S0-R1`; victim, materialization, smoke, E20, evaluation, aggregate, and AP50 are forbidden.
- Code snapshot: branch `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`, code commit `3a7a1aaff912d0904794a91a4d3512d18b5c69fa`.
- Intent: unchanged r2 → T0/T1/P0/P1 8-step mechanism and truthful P1 feasibility persistence.
- Code location: `ue_framework.tools.run_tausb_sdh` → `run_mechanism_pilot` → exact input loader → observation/objective/CGR/NLA → feasibility-state sink.
- Parameter data flow: remote CPU probe exercised the real config validator, r2 loader, dataset manifests, secret source/manifest/tensor, surrogate hash, and person enumeration before GPU use.
- Runtime state: clean detached worktree `/root/tausb-sdh-checkouts/e2e-v0-3a7a1aa-worktree`; the pre-existing dirty `/root/autodl-tmp` repository is input-only and never used as executable code.
- Sink effect: mechanism writes only its unique root and preserves `hiding_gate_passed=false` plus the actual mechanism decision.
- Baseline/disable path: the four switch arms remain unchanged; formal failed-gate loading remains rejected by tests.
- Local validation: 105 related tests, compile, Python 3.8 AST, config parse, CLI, diff, and local real-VOC manifest checks passed.
- Minimal probe: remote no-card CPU execution loaded the actual r2 checkpoint and matched 16,551 image/label records, 6,095 person images, both dataset manifests, hiding split, secret manifest/tensor, and surrogate SHA-256.
- Run command binding: `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/mechanism_run_contract.json`; tmux session `tausb-sdh-e2e-v0-mech-s0-r1`; payload SHA-256 `e7d2f634...a261`.
- Experiment validity: VOC20, person=14, epsilon=16/255, seed0, 8 steps, EOT/JND off, exact r2 and surrogate, and no victim stage are fixed.
- Output non-overwrite: `/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH` is fresh and the payload refuses to run if it exists.
- Recoverability/secrecy: external log, status, metrics, P1 paths, tmux health checks, 1,200-second timeout, 900-second internal cap, and automatic shutdown are frozen; no credentials are stored.
- Blockers: none in code or remote inputs; GPU mode is an external prerequisite for launch.
- Validation gaps: mechanism metrics and P1 state do not exist until the GPU run completes; no scientific or AP50 claim is made.

## Final claim/evidence review

`pending`.

## Append-only execution log

- 2026-08-11: User approved the E2E V0 reset: collect real AP50 first and defer carrier optimization.
- 2026-08-11: CSV structure validated: canonical 28-column header, 23 rows, unique IDs, and final `REVIEW-01` row.
- 2026-08-11: Selected `FEASIBILITY-LOADER-01` as the first unmet dependency; implementation is restricted to an exact-protocol feasibility branch with formal rollback tests.
- 2026-08-11: Closed `FEASIBILITY-LOADER-01`. The loader requires the frozen r2 metrics/checkpoint hashes and exactly the two recorded failed checks; 16 focused tests, Python 3.8 AST parsing, and diff-check passed. Formal failed-gate loading remains rejected.
- 2026-08-11: Started `MECH-V0-STATE-01`; scope is limited to persisting the actual optimized P1 with truthful gate/provenance fields while leaving formal PASS-only persistence unchanged.
- 2026-08-11: Closed `MECH-V0-STATE-01`. The actual arm-tagged P1 is reloaded and checked for finite tensors and Linf before the feasibility payload is written. The payload preserves hiding FAIL and the actual mechanism decision; 51 focused SDH tests passed.
- 2026-08-11: Started `CONFIG-PAIRED-01` to bind the exact mechanism input and add isolated smoke/E20 C0/M1 runtime contracts without weakening the formal 200-epoch protocol.
- 2026-08-11: Closed `CONFIG-PAIRED-01`. A ready mechanism config and a post-mechanism binder now generate four exact C0/M1 configs. Smoke selection and labels are hash-bound; the V0 materializer verifies the state file plus seven provenance hashes. Formal E200 remains unchanged.
- 2026-08-11: Started `EVAL-COMPARE-01` to carry V0 provenance into metrics and compare two explicit full VOC20 metric files with frozen directional thresholds.
- 2026-08-11: Closed `EVAL-COMPARE-01`. Evaluation now records clean-val and paired-training hashes plus the full mechanism provenance chain. The comparator outputs all 20 AP50 values with C0-minus-M1 drop and preserves mechanism/hiding gate failures. Thirty-one focused tests passed.
- 2026-08-11: Started `LOCAL-VALIDATION-01`; no GPU work is permitted in this row.
- 2026-08-11: Closed `LOCAL-VALIDATION-01`. A 103-test SDH regression suite passed, as did scoped compile, Python 3.8 AST, CLI, CSV, and diff checks. Read-only real VOC selection produced 200 records with 40 person and 160 person-free (`ced5d8ce...bcc7`).
- 2026-08-11: Started `GIT-SNAPSHOT-01`; only the V0 Spec/CSV/review, implementation, configs, tools, and four test files are in scope.
- 2026-08-11: Closed `GIT-SNAPSHOT-01`. Sixteen scoped files were committed as `7330a76` and pushed normally (non-force) to `origin/codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`; unrelated dirty files, datasets, weights, and temporary artifacts were excluded.
- 2026-08-11: Started `PRERUN-MECH-01`; the review is bound to implementation commit `7330a76` and the mechanism-only V0 command. No GPU command is authorized until this review passes.
- 2026-08-11: `PRERUN-MECH-01` found one concrete binding gap in `7330a76`: surrogate, full dataset manifests, and secret provenance were recorded but not all content-verified at runtime. The gap was fixed without changing method equations in commit `3a7a1aa`.
- 2026-08-11: Commit `3a7a1aa` passed 105 related tests, Python 3.8 AST/compile, config/CLI checks, and a real local VOC manifest recomputation; it was pushed normally to the same branch.
- 2026-08-11: Current remote read-only audit could not start because the configured AutoDL SSH endpoint refused the connection. Pre-run remains blocked and no GPU command was launched.
- 2026-08-11: User enabled no-card mode. The existing `/root/autodl-tmp` checkout was found to be an old dirty branch, so it was preserved and excluded from execution.
- 2026-08-11: Created clean detached worktree `/root/tausb-sdh-checkouts/e2e-v0-3a7a1aa-worktree` at exact code commit `3a7a1aaff912d0904794a91a4d3512d18b5c69fa`; no existing file or artifact was deleted, reset, or cleaned.
- 2026-08-11: Remote CPU sink probe passed the real V0 loader and content binding: VOC counts/manifests, person=6095, r2 hashes/split, secret manifest/tensor, and surrogate hash all match. The unique mechanism root is fresh; tmux, timeout, shutdown, environment, and disk checks pass.
- 2026-08-11: Closed `PRERUN-MECH-01` as `pass / allow_run` for mechanism only. The audit and 20-minute auto-shutdown run contract were committed as `27e434d`; no GPU process was started.
