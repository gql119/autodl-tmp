# VOC Detection Baselines — 8-Hour Budget Variant

This branch is a sanitized code snapshot for the four-method VOC/person budget-8 experiment. It records source corresponding to candidate commit 541e3389317f12e6f5f74d5c010341019c9ddb99, plus the pre-run review and GPU smoke evidence.

The sequence was EM-Det → REM-Det → EntF-Push-Det → SEM-Det, with distinct RunIDs ending in budget8_seed0_20260930. Each generation used an eight-hour cap and only complete passes over the 6,095 target training images. Dataset labels and locked clean validation data were audited before each independent, random-initialized YOLOv8n victim was trained for 200 epochs.

This is an explicitly identified compute-budget variant, not an unchanged official reproduction. The EM training-error threshold is logged but does not stop this variant; EntF reuses its initialization and perturbation across complete Push blocks. The source preserves the method-specific inner loops.

The branch contains source, tests, the protocol description, and small pre-run JSON/log evidence. It excludes datasets, generated images, checkpoints, surrogate/victim weights, and training outputs. Runtime inputs remain external and must match protocol_lock.json; do not commit local data or weight files.

The source candidate passed 13 contract tests and the independent pre-run implementation review recorded in experiments/baseline_budget8_20260930/PRERUN-REVIEW-1.md.
