# TAUSB Learning Trajectory Refactor Report

## Git Baseline

- Original HEAD recorded before code changes: `e02a7528e9a074d229ed02383ecb7191425595dd`
- Legacy marker created: `legacy-best`
- Work branch created: `codex/det-crad-p1-p2`
- Pre-existing dirty tracked files before this refactor: `ue_framework/launch_one.py`, `ue_framework/paths.py`, `ue_framework/runtime.py`, `ue_framework/stages/aggregate.py`, `ue_framework/stages/evaluate.py`, `ue_framework/stages/train_victim.py`

## Preserved Legacy Modules

- `ue_framework/methods/tausb_universal.py`: preserved as the legacy-best TAUSB/ALCE-era implementation.
- `ue_framework/methods/alce_acgt.py`, `ue_framework/methods/alce_losses.py`, `ue_framework/methods/alce_metrics.py`: preserved for legacy baseline compatibility only.
- `ue_framework/methods/em.py`, `ue_framework/methods/rem.py`, `ue_framework/methods/ours.py`, `ue_framework/methods/shadow_tal.py`, `ue_framework/methods/fourier.py`: preserved as existing baselines/common utilities.
- `ue_framework/stages/*`, `ue_framework/metrics_utils.py`, `ue_framework/data_utils.py`, `ue_framework/support.py`: preserved to avoid changing the stable generate/train/evaluate pipeline.

## Moved Or Deleted Modules

- Moved modules: none.
- Deleted redundant modules: none.
- Archived modules under `legacy_archive/`: none.

No bulk delete or directory move was performed. The repository contains many untracked experiment artifacts and pre-existing tracked edits, so this pass isolates new code by namespace instead of physically relocating legacy code.

## New Modules

- `ue_framework/core/assignment_parser.py`: stable assignment result dataclass and FPN level inference.
- `ue_framework/core/class_routing.py`: protected/authorized/ambiguous routing and per-level stats.
- `ue_framework/core/detector_adapter.py`: detector adapter with parameter-scope selection, proxy assignment, and class-filtered detection loss.
- `ue_framework/core/feature_hooks.py`: reusable feature hook manager.
- `ue_framework/core/perturbation.py`: universal additive and Fourier perturbation modules.
- `ue_framework/core/logging_utils.py`: compact CSV append helper.
- `ue_framework/core/evaluator.py`: protected/authorized metric view helper.
- `ue_framework/methods/learning_trajectory/*`: P1 gradient trajectory objective, P2 virtual update objective, differentiable gradient extraction, and meta objective.
- `ue_framework/methods/baselines/*`: baseline namespace documentation without importing legacy modules into new methods.
- `runners/run_p0_diagnostics.py`: small-batch P1/P2 autograd diagnostic runner.
- `runners/generate_poison.py`: new-method diagnostic entrypoint; full VOC materialization remains separate technical debt.
- `runners/train_victim.py`, `runners/evaluate.py`: wrappers around the existing `launch_one.py` stages.
- `configs/p1_trajectory/voc_yolov8n_head.yaml`: P1 config.
- `configs/p2_virtual_update/voc_yolov8n_head.yaml`: Meta-only P2 config.
- `configs/p2_virtual_update/voc_yolov8n_head_p1_meta.yaml`: P1+Meta config.
- `tests/*`: routing, loss isolation, P1 gradient, virtual update, and parameter leak tests.

## Algorithm Notes

- Protected/authorized loss separation: `ClassConditionedRouter` filters assigned foreground units by `protected_class_id` and `authorized_class_ids`; ambiguous units are excluded by default. `compute_class_conditioned_detection_loss` computes separate cls/box/dfl components over disjoint unit masks.
- Parameter gradients: `extract_gradient_vector` uses `torch.autograd.grad` with stable parameter order. `grad is None` becomes a zero vector, and poisoned gradients use `create_graph=True` when needed.
- P1 delta update path: clean gradients are stop-gradient references; poisoned protected/authorized gradients remain graph-connected to `delta`, so the cosine trajectory loss can update the perturbation.
- Virtual update: `make_virtual_parameters` builds `theta_plus = theta - lr * grad` without assigning into the model. `functional_forward` calls the model with a merged parameter/buffer state.
- Meta gradient path: query images are clean; `delta` affects query meta loss through `delta -> support_loss -> theta_plus -> query_loss`.
- Parameter leak prevention: snapshots are compared after virtual update; tests assert `parameter_leak_max_abs_diff == 0.0`.

## Tests

Executed with `C:\Users\20272\AppData\Local\Programs\Python\Python314\python.exe tests\run_tests.py`.

| Test | Result | Key check |
| --- | --- | --- |
| class routing | PASS | protected=1, authorized=1, ambiguous=1 in synthetic assignment |
| class loss isolation | PASS | protected logit changes mainly affect protected loss; authorized logit changes mainly affect authorized loss |
| P1 gradient | PASS | finite nonzero `delta.grad`; model parameters unchanged |
| virtual update | PASS | virtual params differ; functional meta gradient reaches `delta`; leak=0 |
| no parameter leak | PASS | two consecutive virtual batches preserve original model parameters |

## Minimal Diagnostic

Executed:

```bash
C:\Users\20272\AppData\Local\Programs\Python\Python314\python.exe runners\run_p0_diagnostics.py --config configs\p2_virtual_update\voc_yolov8n_head.yaml
```

Key values:

- `protected_positive_count`: `2.0`
- `authorized_positive_count`: `2.0`
- `cos_protected_clean_poison`: `0.999999463558197`
- `cos_authorized_clean_poison`: `0.999999463558197`
- `meta_gradient_norm_to_delta`: `1.211159457170652e-07`
- `parameter_leak_max_abs_diff`: `0.0`
- `peak_gpu_memory`: `0.0`

Additional configs verified:

- `configs/p1_trajectory/voc_yolov8n_head.yaml`
- `configs/p2_virtual_update/voc_yolov8n_head_p1_meta.yaml`

## Remaining Technical Debt

- Full VOC poisoned dataset materialization for `trajectory_p1`, `meta_p2`, and `trajectory_meta_p2` is not wired into the legacy per-image generator path. This requires a batch-level dataloader and output writer rather than extending old ALCE-style per-image generation.
- The detector adapter currently uses a differentiable proxy assignment/loss for the new trajectory method. Replacing it with exact Ultralytics TAL loss internals can be done behind the same adapter interface.
- Existing legacy configs and large experiment artifacts remain in place. They were not moved or deleted because safe call-graph proof and artifact ownership were not established in this pass.
