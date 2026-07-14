# OA-LGC Next Experiment Decision

## Decision

`real YOLO engineering pass, pilot blocked`

Do not proceed to formal multi-seed, multi-target-class, YOLOv8s, Faster R-CNN, DETR, RCDS, QP, recovery, or defense experiments.

## Unblocking requirements

1. Provide local VOC2012 trainval with its authoritative split file.
2. Provide local VOC2007 test with its authoritative split file.
3. Generate exact combined train and independent test manifests, hashes, class distributions, and zero-overlap proof.
4. Train or verify a clean YOLOv8n baseline on that exact protocol.
5. Record per-epoch person AP50 and non-target mean AP50.
6. Select the smallest valid `E_pilot` from 20/40/60/80/100 using the preregistered 80% rule.
7. Run the equal-budget P0-P7 pilot without test-set tuning.

Only after those requirements pass may the project evaluate learning-gain/AP correlation and Pareto behavior. C0-C3 should be reused unchanged; there is no justification for adding RCDS or QP to conceal the missing pilot evidence.
