# C2 TAL / Box / DFL Implementation

`oa_lgc/yolo_diagnostics.py` adds a diagnostic-only layer over `YOLOFunctionalAdapter`. It does not alter `ue_framework/ultra/hijacked_loss.py`, `dcss/unit_partition.py`, victim training, or evaluation.

## Assignment diagnostics

For each class, the implementation records GT count, reference/clean/poison positive counts, and assignment drift. TAL assignments are recomputed for diagnostics, while gain losses continue to use the fixed clean-reference assignment.

Target localization recall is the fraction of target GT indices represented by poison-assigned target positives. Target matched score is the mean target score on poison target-positive units.

## Fixed-reference localization diagnostics

The native criterion decodes distribution logits using its real DFL projection. Per-class masks retain only fixed-reference units for that class. The native `BboxLoss` then returns class-specific CIoU and DFL losses, with the checkpoint's box/DFL gains applied.

This separates four possible outcomes:

1. classification gain changes while assignment and localization remain available;
2. target TAL positives disappear;
3. target box loss becomes unusable or drifts;
4. target DFL becomes unusable or drifts.

All target and classwise output fields are declared as fixed schema constants and verified by tests.
