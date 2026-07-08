# Local Validation Final Alignment

- Final HEAD: `aacab4d2b42cb27c237c87e39dcfcfb363c58807`
- Final tag(s): `det-crad-p1-p2-local-validation-pass`
- Branch: `codex/det-crad-p1-p2`
- Original local_validation_report.md commit: `e616b4ee10201d621266b7ac0bd756b938aabee8`

## Code Difference

```text
A	ue_project/docs/local_validation_report.md
A	ue_project/requirements-local-validation-freeze.txt
A	ue_project/requirements-local-validation.txt
A	ue_project/runners/run_local_validation.py
A	ue_project/ue_framework/core/yolov8_tal_adapter.py
```

## Tests On Final HEAD

```text
RUN tests.test_class_routing.test_class_routing_protected_authorized_ambiguous_counts
PASS tests.test_class_routing.test_class_routing_protected_authorized_ambiguous_counts
RUN tests.test_class_conditioned_loss.test_class_conditioned_loss_isolates_protected_and_authorized_logits
PASS tests.test_class_conditioned_loss.test_class_conditioned_loss_isolates_protected_and_authorized_logits
RUN tests.test_gradient_extractor.test_p1_loss_is_differentiable_to_delta_and_does_not_update_surrogate
PASS tests.test_gradient_extractor.test_p1_loss_is_differentiable_to_delta_and_does_not_update_surrogate
RUN tests.test_virtual_update.test_virtual_update_changes_functional_parameters_and_meta_loss_reaches_delta
PASS tests.test_virtual_update.test_virtual_update_changes_functional_parameters_and_meta_loss_reaches_delta
RUN tests.test_no_parameter_leak.test_two_virtual_batches_do_not_leak_parameters
PASS tests.test_no_parameter_leak.test_two_virtual_batches_do_not_leak_parameters
RUN tests.test_p2_inner_full_loss.test_p2_inner_update_uses_full_detection_loss
PASS tests.test_p2_inner_full_loss.test_p2_inner_update_uses_full_detection_loss
ALL_PASS
......                                                                   [100%]
6 passed in 8.05s
```