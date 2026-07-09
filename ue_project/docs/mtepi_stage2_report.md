# MTEPI Stage 2 Report

## Executive Decision
- Person-selective functional channels were not established in this run.
- The blocking issue is checkpoint legality: only a late VOC20 surrogate checkpoint is available, with no same-run early/middle checkpoints or shared initialization/training-manifest metadata.
- Stage 3 was not implemented or run because Stage 2 failed the hard gate.

## Gate
- STAGE_2_GATE: `FAIL`
- reasons: `['no legal same-trajectory early/middle/late checkpoint set', 'insufficient target-selective functional channels', 'no legal consensus pathway', 'Top-k AP ablation curve not available', 'cross-checkpoint transfer matrix not available', 'bootstrap confidence intervals not available']`

## Checkpoints
- legal same-trajectory checkpoints: `False`
- valid checkpoint count: `1`
- roles present: `['late']`

## Layer Registry
- candidate layer count: `3`
- registered layers: `[('P3', 'model.3', [1, 64, 20, 20], 8), ('P4', 'model.5', [1, 128, 10, 10], 16), ('P5', 'model.7', [1, 256, 5, 5], 32)]`

## Required Stage 2 Answers
1. person-selective functional channels: `not established`.
2. local ROI ablation vs global Top-k ablation: `not evaluated`; checkpoint hard gate failed before functional scoring.
3. most selective layer: `not determined`.
4. Top 1/5/10/20% AP effect: `not run`.
5. random/activation/gradient baseline comparison: `not run`.
6. same-trajectory checkpoint index stability: `not valid to compute`.
7. cross-checkpoint functional transfer: `not valid to compute`.
8. bootstrap confidence interval support: `not available`.
9. legal consensus pathway: `not formed`.
10. STAGE_2_GATE: `FAIL`.

## Stage 3
- Not run. Stage 2 failed the legal checkpoint hard gate, so no perturbation store, delta, poisoned dataset, ARPS loss, or victim training was created.