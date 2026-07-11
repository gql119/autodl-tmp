# Mini CS-DEM three-stage summary

## Results and status

| Stage | Seed/variant | target | non-target | all | free NT | cooccur NT | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Clean | seed 0 | 0.30819 | 0.08220 | 0.09350 | 0.08225 | 0.13629 | PASS |
| Stage 1 | seed 0 | 0.19116 | 0.07188 | 0.07785 | 0.07375 | 0.09787 | PASS |
| Stage 1 | seed 1 | 0.21393 | 0.07845 | 0.08522 | 0.09961 | 0.10726 | PASS |
| Stage 2 short | A | 0.06423 | 0.01698 | 0.01934 | 0.01424 | 0.01834 | PARTIAL |
| Stage 2 short | B | 0.07425 | 0.01671 | 0.01959 | 0.02017 | 0.01505 | PARTIAL |
| Stage 2 short | C | 0.11057 | 0.01980 | 0.02434 | 0.01512 | 0.02850 | PARTIAL |
| Stage 2 short | D | 0.07524 | 0.01894 | 0.02176 | 0.01538 | 0.02458 | PARTIAL |
| Stage 2 formal | C rejected, seed 0 | 0.30010 | 0.09409 | 0.10439 | 0.08635 | 0.11837 | FAIL |
| Stage 2 formal | D, seed 0 | 0.28644 | 0.07665 | 0.08714 | 0.06079 | 0.11850 | FAIL |
| Stage 2 formal | seed 1 | not run | not run | not run | not run | not run | gated |
| Stage 3 | all variants/seeds | not run | not run | not run | not run | not run | Stage 2 gate |

Stage 1 mean±population-std is target `0.20254±0.01138`, non-target `0.07517±0.00328`, all `0.08153±0.00369`, free `0.08668±0.01293`, and cooccur `0.10257±0.00469`. Stage 2/3 mean±std is unavailable because their seed-1 gate was not reached.

Stage 1 poison-train target AP is `0.58534/0.58066` versus clean-val `0.19116/0.21393`, supporting a learnability gap. Formal Stage 2 D closes that gap (`0.28804` versus `0.28644`), consistent with loss of the unlearnable effect.

All poison runs retain `L_inf=8/255`. Stage 1 materialized mean absolute perturbation is `0.003118/0.002927`; formal D is `0.002036` with area ratio `0.13152`. Formal D's last target/preserve gradient norms are `2.37113/2.43741`, cosine `0.07108`, and alignment `0.98115`; failure is not delta or gradient collapse.

## Per-class AP50

| Class | Clean | Stage1 s0 | Stage1 s1 | Stage2 D s0 |
|---|---:|---:|---:|---:|
| aeroplane | 0.03437 | 0.11128 | 0.13951 | 0.04734 |
| bicycle | 0.04003 | 0.03283 | 0.11178 | 0.03349 |
| bird | 0.00468 | 0.00135 | 0.00148 | 0.00667 |
| boat | 0.00505 | 0.03327 | 0.00266 | 0.00397 |
| bottle | 0.00650 | 0.00033 | 0.00031 | 0.00000 |
| bus | 0.03027 | 0.00875 | 0.04552 | 0.00321 |
| car | 0.23815 | 0.23939 | 0.20653 | 0.21257 |
| cat | 0.07006 | 0.02954 | 0.05859 | 0.08034 |
| chair | 0.04874 | 0.02254 | 0.01641 | 0.03390 |
| cow | 0.02680 | 0.04435 | 0.02426 | 0.07681 |
| diningtable | 0.08135 | 0.08518 | 0.08100 | 0.18802 |
| dog | 0.11090 | 0.11112 | 0.10849 | 0.11152 |
| horse | 0.23029 | 0.13166 | 0.17909 | 0.20302 |
| motorbike | 0.40028 | 0.38833 | 0.38539 | 0.29253 |
| person | 0.30819 | 0.19116 | 0.21393 | 0.28644 |
| pottedplant | 0.00043 | 0.00888 | 0.00265 | 0.00158 |
| sheep | 0.00065 | 0.00057 | 0.00020 | 0.00156 |
| sofa | 0.12987 | 0.07126 | 0.08668 | 0.09225 |
| train | 0.08635 | 0.02760 | 0.01777 | 0.03664 |
| tvmonitor | 0.01699 | 0.01756 | 0.02219 | 0.03091 |

Worst drops: Stage 1 seed 0 horse `0.09864`; seed 1 train `0.06858`; Stage 2 short D motorbike `0.25587`; formal D motorbike `0.10775`.

## Research conclusions

1. Object-aligned target-only Detection-EM stably lowers person AP (`0.30819` clean to `0.19116/0.21393`).
2. Fresh-victim hashes, clean validation, two-seed direction, poison-train/clean-val gaps, finite gradients, and perturbation checks support a genuine unlearnable trend rather than a mechanical error.
3. No Stage 2 term is formally validated. C's logits+box+DFL improves short mean AP but fails worst-class; D improves short worst-class but formal attack retention fails.
4. TAL remains ineffective/unvalidated: short drift/gradient is near zero and there is no isolated causal evidence.
5. Stage 2 does not protect merely by collapsing delta; formal D reaches `8/255` with nonzero mean magnitude.
6. Fixed weights couple non-target preservation to target learnability: the victim relearns person while collateral gains are too small.
7. Stage 3 dual concentration was not tested because the Stage 2 gate failed.
8. Gradient projection's effect on worst-class drop was not tested.
9. Projection did not weaken target EM because it was never enabled.
10. The current minimum effective method is Stage 1 object-aligned target-only Detection-EM alone.
11. Logits, box, DFL, TAL preservation, classwise duals, and projection stay outside the final method until independently validated.
12. The single highest-priority next experiment is target-learnability-gap/virtual-update validation, because Stage 2's central failure is restoration of target learnability.

Final statuses: Stage 1 **PASS**, Stage 2 **FAIL**, Stage 3 **NOT ENTERED by gate**.
