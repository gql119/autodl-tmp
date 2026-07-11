# Stage 2: non-target response preservation

## Outcome

Stage 2 is **FAIL**, not a mechanical error. Alignment, detached-teacher behavior, fresh victim initialization, and perturbation constraints remained correct, but both tested 50-epoch combinations removed most of the Stage 1 target attack. Per the gate, seed 1 was not run and Stage 3 is not permitted.

## Takeover and reproducibility

- Branch/starting HEAD: `codex/mini-csdem-3stage` / `1a71baac75465780378fc6700249f043e1df707b`.
- Stage 1 commit: `1a71baac75465780378fc6700249f043e1df707b`.
- Fresh `git fetch origin` timed out after about 74 seconds (exit 124); this was not an authentication error. Latest remote state is unverified.
- No experiment process was running at takeover. All A/B/C/D artifacts were complete, so none was repeated.
- Split: `data_splits/mini_csdem_voc_seed0.json`, SHA256 `BC1FFAD0D4A6167BB6882E1D137AB615DBF64E4F9C7589361BFB7518332216CD`, 800 train/200 validation, person ID resolved from names as 14.
- Model: `configs/voc_yolov8n_20cls.yaml`; victims record `pretrained=false`, `resume=false`, and fresh initial hashes distinct from trained surrogate final hashes.

## Implementation and mechanical verification

The clean forward is a detached teacher. Non-target units are intersected by `(batch, FPN level, global flattened anchor, matched GT index, matched GT class)` and exclude class 14. Logit MSE, decoded-xyxy SmoothL1, four-edge DFL-distribution MSE, and differentiable TAL soft-score drift have independent config weights. Detached EMA normalization uses the fixed `1e-5` floor.

The Stage 1/2 harness passed `22 passed`. Smoke alignment was about 99.97%; formal coverage remained 98.12–98.57%, GT mismatch at most 0.05 averaged units per batch, the teacher required no gradient, no NaN/Inf appeared, and materialized perturbations used `L_inf=0.03137255`.

## Short ablation: 3 generation + 15 fresh-victim epochs, seed 0

| Variant | target | non-target | all | free NT | cooccur NT | worst class/drop | target/preserve grad | cosine | coverage |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| A Stage 1 | 0.06423 | 0.01698 | 0.01934 | 0.01424 | 0.01834 | motorbike / 0.36009 | 1.31478 / 0 | n/a | 0.99039 |
| B logits | 0.07425 | 0.01671 | 0.01959 | 0.02017 | 0.01505 | motorbike / 0.37046 | 1.99657 / 0.28326 | -0.08931 | 0.98949 |
| C logits+box+DFL | 0.11057 | 0.01980 | 0.02434 | 0.01512 | 0.02850 | motorbike / 0.36402 | 0.81789 / 0.32895 | 0.07259 | 0.99588 |
| D +TAL | 0.07524 | 0.01894 | 0.02176 | 0.01538 | 0.02458 | motorbike / 0.25587 | 1.46998 / 1.01916 | 0.01198 | 0.99274 |

B does not improve mean, co-occurrence, or worst-class AP. C improves mean and co-occurrence but fails the required worst-class check. D is the only short variant that materially improves worst-class drop while keeping target rebound small, so D was the evidence-based formal candidate. Full rows are in `results/mini_csdem/stage2_ablation_seed0.csv`.

TAL's short raw gradient was only `8.40e-7`, and drift `1.60e-10`; it is not independently validated. D nevertheless differed materially from C, so D was tested formally instead of inferring equivalence from raw gradient alone.

## Formal seed 0

| Candidate | target | non-target | all | free NT | cooccur NT | poison-train target | target rebound vs Stage 1 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C, rejected selection evidence | 0.30010 | 0.09409 | 0.10439 | 0.08635 | 0.11837 | 0.32752 | +0.10895 | FAIL |
| D, final tested combination | 0.28644 | 0.07665 | 0.08714 | 0.06079 | 0.11850 | 0.28804 | +0.09528 | FAIL |

D improves non-target AP over Stage 1 seed 0 by only `0.00477`, below the required `0.015`, and target rebound exceeds the allowed `0.05`. Its worst clean-relative drop is motorbike `0.10775`. Seed 1 and mean±std are unavailable by design because seed 0 failed the entry gate.

Formal D last-epoch diagnostics: logits/box/DFL/TAL drift `0.00180376 / 9.896e-6 / 4.192e-5 / 1.278e-7`; target/preserve gradient norms `2.37113 / 2.43741` (ratio `1.028`); cosine `0.07108`; normalized combined preserve loss `1.61707`; delta tensor mean absolute `0.02091`, saturation `0.25928`; materialized mean absolute `0.002036`, area ratio `0.13152`. Delta did not collapse.

## Per-class clean-val AP50

| Class | Clean | Stage 1 seed0 | Formal D seed0 |
|---|---:|---:|---:|
| aeroplane | 0.03437 | 0.11128 | 0.04734 |
| bicycle | 0.04003 | 0.03283 | 0.03349 |
| bird | 0.00468 | 0.00135 | 0.00667 |
| boat | 0.00505 | 0.03327 | 0.00397 |
| bottle | 0.00650 | 0.00033 | 0.00000 |
| bus | 0.03027 | 0.00875 | 0.00321 |
| car | 0.23815 | 0.23939 | 0.21257 |
| cat | 0.07006 | 0.02954 | 0.08034 |
| chair | 0.04874 | 0.02254 | 0.03390 |
| cow | 0.02680 | 0.04435 | 0.07681 |
| diningtable | 0.08135 | 0.08518 | 0.18802 |
| dog | 0.11090 | 0.11112 | 0.11152 |
| horse | 0.23029 | 0.13166 | 0.20302 |
| motorbike | 0.40028 | 0.38833 | 0.29253 |
| person | 0.30819 | 0.19116 | 0.28644 |
| pottedplant | 0.00043 | 0.00888 | 0.00158 |
| sheep | 0.00065 | 0.00057 | 0.00156 |
| sofa | 0.12987 | 0.07126 | 0.09225 |
| train | 0.08635 | 0.02760 | 0.03664 |
| tvmonitor | 0.01699 | 0.01756 | 0.03091 |

## Failure diagnosis and gate

The two prior scale fixes (detached EMA normalization, then floor `1e-5`) prevented vanishing or 300x preservation gradients. In formal D, preserve/target ratio is near one, alignment is high, and delta is nonzero, yet fixed-weight preservation repairs target learnability together with non-target responses. This coupling is the principal bottleneck. TAL remains unvalidated because its drift and raw gradient are orders below logits/box/DFL.

Final Stage 2 status: **FAIL**. No Stage 2 module is promoted to the final method; all preservation switches are disabled by default after the experiment. Stage 3 entry is false, so dual constraints and projection must not mask this failure.
