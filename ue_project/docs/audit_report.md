# P1/P2 Refactor Audit Report

## Git State

- Current commit: `e02a7528e9a074d229ed02383ecb7191425595dd`
- Current branch: `codex/det-crad-p1-p2`
- `legacy-best`: points to the same commit.
- Important: the P1/P2 work is currently uncommitted working-tree state.

Tracked changes relative to `legacy-best`:

- `M ue_framework/config.py`
- `M ue_framework/launch_one.py` (pre-existing dirty file before this refactor)
- `M ue_framework/methods/__init__.py`
- `M ue_framework/methods/factory.py`
- `M ue_framework/paths.py` (pre-existing dirty file before this refactor)
- `M ue_framework/runtime.py` (pre-existing dirty file before this refactor)
- `M ue_framework/stages/aggregate.py` (pre-existing dirty file before this refactor)
- `M ue_framework/stages/evaluate.py` (pre-existing dirty file before this refactor)
- `M ue_framework/stages/train_victim.py` (pre-existing dirty file before this refactor)

New untracked project files from this refactor:

- `ue_framework/core/*`
- `ue_framework/methods/learning_trajectory/*`
- `ue_framework/methods/baselines/*`
- `runners/*`
- `tests/*`
- `configs/p1_trajectory/*`
- `configs/p2_virtual_update/*`
- `docs/refactor_report.md`
- `docs/audit_report.md`
- `outputs/audit/p1_p2_audit.json`

## Architecture Check

Forbidden-module search over new method runtime chain:

```text
rg "ALCE|alce|RLCP|rlcp|context prototype|context_prototype|DES-R|des_r|FDACB|fdacb|weighted ring|weighted_ring|dual carrier|dual_carrier" ue_framework/core ue_framework/methods/learning_trajectory runners configs/p1_trajectory configs/p2_virtual_update
```

Hits:

- `runners/generate_poison.py`: user-facing warning string mentioning ALCE.
- `ue_framework/core/detector_adapter.py`: docstring stating the adapter avoids ALCE/RLCP.

No import path from the new method modules to ALCE/RLCP/DES-R/FDACB/weighted-ring legacy modules was found.

## Category Routing

Synthetic batch:

- protected class: `14`
- authorized class in batch: `1`

Observed:

- `protected_labels`: `[14, 14]`
- `authorized_labels`: `[1, 1]`
- `protected_positive_count`: `2.0`
- `authorized_positive_count`: `2.0`
- `ambiguous_positive_count`: `0.0`

Masks:

```text
protected_mask = [[1, 0, 0, 0], [1, 0, 0, 0]]
authorized_mask = [[0, 1, 0, 0], [0, 1, 0, 0]]
ambiguous_mask = [[0, 0, 0, 0], [0, 0, 0, 0]]
```

## Class Loss Isolation

Initial audit found an issue: the positive-unit cls loss was filtered by unit, but still used all class logits in BCE. That allowed a person logit on an authorized unit to influence authorized loss as a negative class. This was fixed in `ue_framework/core/detector_adapter.py` by using assigned-class-only positive cls loss.

Post-fix values:

- `base_protected`: `1.1302134990692139`
- `base_authorized`: `0.5017332434654236`
- protected person-logit change delta protected: `1.116187334060669`
- protected person-logit change delta authorized: `0.0`
- authorized class-logit change delta protected: `0.0`
- authorized class-logit change delta authorized: `0.4973524212837219`
- person logit changed on authorized unit delta authorized: `0.0`

Conclusion: protected and authorized class-conditioned losses are now isolated for the synthetic assignment.

## P1 Gradient Validation

Single batch:

- `cos_protected_clean_poison`: `0.9999982714653015`
- `cos_authorized_clean_poison`: `0.9999982714653015`
- `norm_g_protected_clean`: `1.4142135381698608`
- `norm_g_protected_poison`: `1.4142135381698608`
- `norm_g_authorized_clean`: `1.4142136573791504`
- `norm_g_authorized_poison`: `1.4142135381698608`
- `p1_loss`: `1.0`
- `delta_grad_norm`: `7.077984719217056e-06`
- `surrogate_parameter_max_abs_diff`: `0.0`

Twenty delta optimization steps:

- protected cosine sequence:
  `[0.9999974966049194, 0.9962396025657654, 0.9999815821647644, 0.9970408082008362, 0.9975917935371399, 0.9999861121177673, 0.9975917935371399, 0.9975917935371399, 0.9998427629470825, 0.9990683197975159, 0.9990861415863037, 0.9997913241386414, 0.9983643293380737, 0.9991743564605713, 0.9999476075172424, 0.9975917935371399, 0.9975917935371399, 0.9998233318328857, 0.9996324181556702, 0.9982783198356628]`
- authorized cosine sequence:
  `[0.9999974370002747, 0.9962393045425415, 0.9999910593032837, 0.9970427751541138, 0.9976291060447693, 0.999997615814209, 0.9976291060447693, 0.9976291060447693, 0.9999197125434875, 0.9991714358329773, 0.9991904497146606, 0.999877393245697, 0.998439610004425, 0.9992809295654297, 0.9999921321868896, 0.9976291060447693, 0.9976291060447693, 0.9999057650566101, 0.9997355937957764, 0.9983492493629456]`
- protected start/end: `0.9999974966049194 -> 0.9982783198356628`
- authorized start/end: `0.9999974370002747 -> 0.9983492493629456`
- surrogate parameter diff after 20 steps: `0.0`

Interpretation: the objective produces nonzero gradients and reduces protected cosine on the toy batch. Authorized cosine remains high but also fluctuates; this toy diagnostic is not strong evidence of final selective victim-training behavior.

## P2 Virtual Update Validation

Meta-only synthetic batch:

- `virtual_parameter_update_norm`: `0.10361593216657639`
- `parameter_leak_max_abs_diff`: `0.0`
- protected query before update: `0.7422615885734558`
- protected query after clean update: `0.6939477920532227`
- protected query after poisoned update: `0.6941064596176147`
- authorized query before update: `0.802425742149353`
- authorized query after clean update: `0.7495175004005432`
- authorized query after poisoned update: `0.7496849298477173`
- `protected_learning_gap`: `0.00015866756439208984`
- `authorized_learning_gap`: `0.00016742944717407227`
- `meta_gradient_norm_to_delta`: `0.0008029752061702311`
- `delta_grad_norm_after_backward`: `0.0008029752061702311`

P1+Meta synthetic batch:

- `virtual_parameter_update_norm`: `0.09322895854711533`
- `parameter_leak_max_abs_diff`: `0.0`
- `meta_gradient_norm_to_delta`: `0.0012652953155338764`
- `p1_loss`: `1.0`

Functional forward check:

- meta-only original vs virtual query max abs diff: `0.09971803426742554`
- P1+Meta original vs virtual query max abs diff: `0.08809584379196167`
- leak after functional check: `0.0`

Conclusion: functional forward uses virtual parameters, and the virtual update does not write into the original model.

## Meta Gradient Path

Source check:

- `make_virtual_parameters` calls `torch.autograd.grad(..., create_graph=create_graph)`.
- `compute_p2_step` calls `make_virtual_parameters(..., create_graph=True)`.
- No `torch.no_grad()` appears in the P2 support-loss to virtual-parameter to query-loss path.
- `detach()` appears only in logging, metric extraction, parameter snapshots, and leak checks after differentiable tensors have already been used.

The numeric check confirms `meta_gradient_norm_to_delta > 0`.

## Parameter Leak Validation

Single P2:

- `parameter_leak_max_abs_diff`: `0.0`
- `surrogate_parameter_max_abs_diff`: `0.0`

Twenty consecutive P2 iterations:

- max parameter leak: `0.0`
- first five meta grad norms: `[0.0028842235915362835, 0.002687883796170354, 0.002814310835674405, 0.0027646408416330814, 0.00286908564157784]`
- last five meta grad norms: `[0.002943448955193162, 0.002887464826926589, 0.002851063385605812, 0.0029608896002173424, 0.0030181938782334328]`

## Memory Stability

Local audit environment:

- `torch.cuda.is_available()`: `False`

Twenty P2 iterations still ran on CPU. GPU memory sequence is therefore all zeros and cannot prove CUDA memory stability:

```text
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## Mode Checks

- P1 only:
  - loss: `0.9999999403953552`
  - delta grad norm: `1.51328504216508e-06`
  - protected cosine: `0.9999998211860657`
  - authorized cosine: `0.9999998807907104`
- Meta only:
  - loss: `-0.03355050086975098`
  - meta grad norm to delta: `0.0013495365856215358`
  - parameter leak: `0.0`
- P1 + Meta:
  - loss: `-0.0822468250989914`
  - meta grad norm to delta: `0.0021522243041545153`
  - p1 loss: `1.0`
  - parameter leak: `0.0`

## Legacy-Best Compatibility

Smoke checks:

- `from ue_framework.methods import build_generator`: PASS, returns callable.
- Legacy config load:
  - `legacy_best_reproduce_mode`: `True`
  - `force_pseudo_mask_fallback`: `True`

Blocked batch-level legacy smoke:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import ue_framework.methods.factory; print('factory_import_ok')
  File "F:\autodl-tmp\ue_project\ue_framework\methods\factory.py", line 1, in <module>
    from .em import EMPoisonGenerator
  File "F:\autodl-tmp\ue_project\ue_framework\methods\em.py", line 6, in <module>
    from .base import BasePoisonGenerator, PoisonResult
  File "F:\autodl-tmp\ue_project\ue_framework\methods\base.py", line 9, in <module>
    from ..support import build_support_mask
  File "F:\autodl-tmp\ue_project\ue_framework\support.py", line 3, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

The local Python used for audit has Torch but not OpenCV, so legacy one-batch execution cannot be completed in this environment.

## Unit Tests

Command:

```text
C:\Users\20272\AppData\Local\Programs\Python\Python314\python.exe tests\run_tests.py
```

Results:

- `test_class_routing_protected_authorized_ambiguous_counts`: PASS
- `test_class_conditioned_loss_isolates_protected_and_authorized_logits`: PASS
- `test_p1_loss_is_differentiable_to_delta_and_does_not_update_surrogate`: PASS
- `test_virtual_update_changes_functional_parameters_and_meta_loss_reaches_delta`: PASS
- `test_two_virtual_batches_do_not_leak_parameters`: PASS

## Unresolved Issues

- Full VOC poisoned dataset materialization for trajectory methods is still not wired into the legacy per-image generator path.
- The new detector adapter uses proxy assignment/loss rather than exact Ultralytics TAL loss internals.
- CUDA memory stability was not tested because the local audit environment has no CUDA device.
- Legacy-best batch smoke was blocked by missing `cv2`.
- P1 toy optimization reduces protected cosine but does not clearly separate it from authorized cosine over 20 steps; this should be tested on a real YOLO/VOC mini-batch before victim retraining.

## Recommendation

Do not start full victim retraining yet. The code now passes mechanism-level synthetic tests for routing, loss isolation, P1 gradient flow, P2 meta-gradient flow, functional virtual update, and parameter leak. The next gate should be a real YOLO/VOC mini-batch diagnostic in an environment with OpenCV, Ultralytics, CUDA, and VOC paths configured.
