## PRERUN-REVIEW-1

- Review mode: scientific_review
- Reviewer: `/root/budget8_scientific_review`, independent read-only child agent
- Result: **scientifically_correct**
- Decision: **allow_run**
- Gated command: `python -m scripts.run_four_detection_baselines --sequence-id four_methods_budget8_seed0_20260930 --budget-variant --batch-size 128 --time-cap-hours 8 --code-commit 541e3389317f12e6f5f74d5c010341019c9ddb99`
- Code snapshot: `codex/detection-baselines-budget8-20260930` / `541e3389317f12e6f5f74d5c010341019c9ddb99`
- Intent alignment: correct
- Canonical code location: correct
- Critical data flow: correct
- Computation sink effect: correct
- Runtime scientific state: correct
- Baseline/disabled path: not_applicable; the budget variant does not claim strict official-method equivalence
- Experiment identity: correct
- Local validation: compilation, 13 baseline contract tests, and `git diff --check` passed.
- GPU smoke: four unique train-only two-block RunIDs; all exit 0, with finite losses and deltas, and continuation from the first block. See `smoke/*.probe.json`.
- Blockers: none
- Validation gaps: none for this pre-run gate

The reviewer confirmed the budget flag reaches the generator loop, full-precision per-image deltas and surrogate optimizer persist, EM's training-error threshold is recorded but does not stop this separate variant, and EntF retains one clean centroid and initial perturbation across 300-step Push blocks. The existing materialization audit and matched 200-epoch victim implementation remain unchanged.
