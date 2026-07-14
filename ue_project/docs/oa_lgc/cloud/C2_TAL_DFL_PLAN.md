# C2 TAL / Box / DFL Plan

## Objective

Demonstrate that real-YOLO learning-gain measurements are not produced by losing target assignments or by silently disconnecting localization. Diagnostics must preserve the C1 adapter and protected historical files.

## Protocol

- Run three disjoint support/query episodes selected to share one non-target class: horse, dog, or bicycle.
- Use the real native TAL assigner for reference, clean-fast, and poison-fast states.
- Define target coverage as `poison target-positive units / max(reference target-positive units, 1)`.
- Invalidate an episode when target coverage is below 0.50.
- Define assignment overlap as Jaccard overlap of reference and poison positive-unit masks.
- Compute fixed-reference classification, box, and DFL losses per class.
- Mark absent or unmatched classes invalid; never fill them with zeros in valid-class aggregates.

The C2 Gate requires median target coverage at least 0.50, low-coverage ratio at most 0.50, available target box/DFL loss, at least one valid non-target class, complete schemas, a stable base hash, and no proxy fallback.
