# Stage 1: Object-aligned Target-conditioned Detection EM

## Method and code

Stage 1 uses one shared `3 x 64 x 64` object-coordinate perturbation. It is bilinearly resized into every person GT box, clipped to `epsilon=8/255`, and optionally removes pixels covered by a 2-pixel-dilated non-target box. Images without person are copied byte-for-byte.

The randomly initialized surrogate alternates two updates per batch:

1. update the surrogate with the standard full detection loss on poisoned images;
2. freeze surrogate parameters and update only `delta_object` by minimizing person-positive classification, box, and DFL loss.

Real Ultralytics TAL assignments define target positives. When a batch has fewer than 16 person positives, background FPN points inside person GT boxes are selected nearest the center, with up to two candidates per level. This is a GT/FPN fallback, not PAG or feature-similarity selection.

Code:

- `mini_csdem/object_aligned_perturbation.py`
- `mini_csdem/gt_conditioned_partition.py`
- `mini_csdem/selective_detection_loss.py`
- `runners/run_mini_csdem_stage1.py`
- `tests/test_mini_csdem_stage1.py`
- `configs/mini_csdem/stage1.yaml`

ALCE, PAG, context prototypes, Fourier perturbations, non-target preservation, dual constraints, and gradient projection are disabled.

## Commands and tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_mini_csdem_stage1.py
.\.venv\Scripts\python.exe runners\run_mini_csdem_stage1.py --config configs\mini_csdem\stage1.yaml
.\.venv\Scripts\python.exe runners\run_mini_csdem_stage1.py --config configs\mini_csdem\stage1.yaml --seed 1
```

Unit tests: `10 passed`. They cover no-person identity, inside/outside support, epsilon projection, multiple-instance sharing, non-target overlap exclusion, target/non-target unit isolation, empty-target handling, gradient propagation, frozen-surrogate integrity, and fresh victim initialization.

The first seed-0 attempt exposed one mechanical error: an all-person-free batch produced a zero loss without a gradient function, but the runner called `backward()`. The targeted fix skips only the delta update for such batches; the batch remains in standard surrogate training. A perturbed-area denominator error was corrected at the same time. Tests were extended and the full experiment was restarted from a fresh random initialization.

## Main results

Clean baseline: target 0.308194, non-target 0.082198, all-class 0.093498.

| Metric | Seed 0 | Seed 1 | Mean ± sample std |
|---|---:|---:|---:|
| clean-val target mAP50 | 0.191158 | 0.213928 | 0.202543 ± 0.016100 |
| target absolute drop | 0.117036 | 0.094267 | 0.105651 ± 0.016100 |
| target relative drop | 37.97% | 30.59% | 34.28% ± 5.22% |
| clean-val non-target mAP50 | 0.071885 | 0.078446 | 0.075166 ± 0.004640 |
| person-cooccur non-target mAP50 | 0.097873 | 0.107263 | 0.102568 ± 0.006639 |
| poisoned-train target mAP50 | 0.585345 | 0.580661 | 0.583003 ± 0.003312 |
| poisoned-train minus clean-val target | 0.394186 | 0.366733 | 0.380460 ± 0.019412 |

Both seeds satisfy the target criterion through relative drop >=25%; seed 0 also exceeds the 0.10 absolute-drop threshold. Both have a poisoned-train/clean-val gap far above 0.10.

## Per-class clean-validation AP50

| Class | Clean | Seed 0 | Seed 1 |
|---|---:|---:|---:|
| aeroplane | 0.034366 | 0.111281 | 0.139505 |
| bicycle | 0.040027 | 0.032832 | 0.111780 |
| bird | 0.004675 | 0.001353 | 0.001479 |
| boat | 0.005049 | 0.033267 | 0.002656 |
| bottle | 0.006500 | 0.000332 | 0.000311 |
| bus | 0.030274 | 0.008755 | 0.045519 |
| car | 0.238151 | 0.239391 | 0.206529 |
| cat | 0.070064 | 0.029543 | 0.058586 |
| chair | 0.048741 | 0.022545 | 0.016409 |
| cow | 0.026804 | 0.044349 | 0.024255 |
| diningtable | 0.081348 | 0.085182 | 0.080998 |
| dog | 0.110903 | 0.111124 | 0.108492 |
| horse | 0.230293 | 0.131656 | 0.179089 |
| motorbike | 0.400276 | 0.388326 | 0.385391 |
| **person** | **0.308194** | **0.191158** | **0.213928** |
| pottedplant | 0.000434 | 0.008881 | 0.002645 |
| sheep | 0.000648 | 0.000571 | 0.000196 |
| sofa | 0.129870 | 0.071264 | 0.086680 |
| train | 0.086349 | 0.027599 | 0.017768 |
| tvmonitor | 0.016987 | 0.017561 | 0.022192 |

Worst non-target AP drops were horse (0.098637) for seed 0 and train (0.068580) for seed 1. This is the collateral damage Stage 2 must address.

## Optimization and perturbation diagnostics

| Diagnostic | Seed 0 first → last | Seed 1 first → last |
|---|---:|---:|
| target EM loss | 13.0685 → 5.3233 | 12.9314 → 5.2134 |
| target cls loss | 4.9738 → 1.4823 | 4.8740 → 1.4702 |
| target box loss | 3.8886 → 1.7060 | 3.8431 → 1.6488 |
| target DFL loss | 4.2062 → 2.1350 | 4.2144 → 2.0944 |
| mean target positives / batch | 107.04 → 107.57 | 107.70 → 107.90 |
| fallback ratio | 1.125% → 0.375% | 0.375% → 0.750% |
| object delta L∞ | 0.031373 | 0.031373 |
| object delta mean absolute | 0.026747 final | 0.025439 final |
| materialized full-dataset mean absolute | 0.003118 | 0.002927 |
| materialized perturbed-area ratio | 0.131508 | 0.131517 |

The effective support during generation was 201,584 pixels per mean batch versus 280,401 raw target-box pixels, confirming that overlap exclusion removed direct non-target coverage. All 400 person-free images were byte-identical to clean source images. Victim initial hashes matched their same-seed fresh YAML initialization and differed from the trained surrogate final hashes.

## Conclusion

Stage 1 status: **PASS** for seeds 0 and 1. Object-aligned target-only Detection-EM produces a repeatable person AP reduction and a large poisoned-train/clean-val behavior gap. Non-target mean AP and cooccurrence AP also decline, and worst-class damage is material, so Stage 2 is justified but must preserve the Stage 1 attack rather than merely shrink delta.

Machine-readable results:

- `results/mini_csdem/stage1_seed0_metrics.json`
- `results/mini_csdem/stage1_seed1_metrics.json`
- `results/mini_csdem/stage1_seed0_per_class.csv`
- `results/mini_csdem/stage1_seed1_per_class.csv`

