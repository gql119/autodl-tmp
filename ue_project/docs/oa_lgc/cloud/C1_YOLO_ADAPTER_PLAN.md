# C1 Real-YOLO Adapter Plan

## Objective

Replace the local `ObjectCropDetector` proxy only in the new real-detector path while preserving the existing carrier, episode, gain, and objective schemas. The gate requires genuine Ultralytics YOLO forward/loss/TAL, functional fast states, cloned buffers, and a non-zero protect-only mixed derivative.

## Ordered scope

1. Normalize legacy checkpoint loss arguments and instantiate the native YOLO criterion.
2. Derive parameter sets from exact registered module identities.
3. Run native detection loss through `torch.func.functional_call`.
4. Keep independent clean/poison parameter and buffer states.
5. Build a detached clean-query reference TAL assignment.
6. Evaluate positive-only classwise query losses on fixed units.
7. Prove the J=1 classification-head protect path reaches `delta_obj` without poisoned query pixels.
8. Gate detection-head J=1, selected-neck/head runnability, full-model interface, hashes, and reproducibility.

RCDS, QP, ALCE context, and feature collision remain disabled. No proxy fallback is allowed.

## Success criteria

The C1 Gate passes only if native forward and box/cls/DFL loss work; Mode A/B J=1 work; base state is unchanged; clean/poison states are independent; protect-only mixed gradient is finite and non-zero; outputs are reproducible within `1e-7`; and all legacy plus new tests pass.
