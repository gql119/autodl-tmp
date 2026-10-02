# VOC/person baseline noise: eight-hour budget variant

Status: candidate schedule for a **new** experiment sequence. The original one-cycle
EM-Det, REM-Det, EntF-Push-Det and SEM-Det runs remain separate, completed evidence.
The user requested this follow-up after observing that EM had little effect and that
all four generators used less than the allowed eight hours.

## Identity and non-goals

- New sequence: `four_methods_budget8_seed0_20260930`; new per-method RunIDs end in
  `budget8_seed0_20260930`. The old RunIDs and artifacts are never reused.
- The matched clean VOC20 baseline, 16,551 train / 4,952 clean validation split,
  person class 14, seed 0, batch 128, 640-pixel surrogate inputs, frozen
  checkpoint, 8/255 person-region budget, 4/255 overlap budget, unchanged labels,
  and independent random-initialized YOLOv8n 200-epoch victim protocol are retained.
- This is a **compute-budget sensitivity experiment**. In particular EM's
  train-error threshold is measured every cycle but does not end this variant;
  the original EM stopped when its first-cycle training error was 0.009731 ≤ 0.01.
  The budget variant must never be labeled an unchanged reproduction of the
  official EM stopping rule. Clean validation is not used to decide cycle count.
- The inner update equations, losses, per-step sizes, EOT/stability/PGD draw
  counts, random initialization policy, and source data stay fixed. EntF computes
  its clean centroid once and initializes its perturbation once; additional
  complete 300-step Push blocks continue the same perturbation. This extends
  EntF's optimization horizon beyond the original fixed 300 steps and is
  reported as such.

## Pre-registered cycle rule

Each method has an eight-hour wall-clock deadline starting before data inventory
and setup. After each **complete** pass over all 6,095 person training images,
the generator measures that cycle's elapsed time. It starts the next pass only
if the remaining time exceeds 1.25 times the slower of (a) the longest
completed cycle and (b) the measured original one-cycle time, plus a 600-second
finalization reserve. A hard eight-hour limit remains in both generator and
pipeline. If a cycle unexpectedly overruns, generation fails closed and no
materialization or victim starts; partial outputs stay in that new RunID.
The maximum cycle counts are guardrails, not targets: EM 100, REM 4, EntF 4,
SEM 32. All completed cycle counts, per-cycle durations, numerical counters,
EM training-error trajectory, and the budget stop reason are saved in
`generation/status.json` and `run_spec.json`.

The first cycle uses the same target order and initialized surrogate as the
original run. Subsequent REM/EntF/SEM cycles reshuffle the training-image
order with the fixed seed stream; EM retains its ordered loader. Model optimizer
state and full-precision per-image perturbations persist in memory across
cycles. EntF retains one frozen clean centroid and does not resample its initial
perturbation. Images are exported to PNG after each completed per-batch update;
the final completed cycle is the only dataset used for victim training.

## Measured inputs and expected horizon

These are planning projections from the **completed original generations**,
not results for this variant. The full-cycle guard makes the actual count
dependent only on throughput and the eight-hour budget.

| Method | Original one-cycle seconds | Expected completed cycles | Expected generation time | Repeated complete block |
|---|---:|---:|---:|---|
| EM-Det | 531.743 | about 50–52 | about 7.4–7.7 h | 10 surrogate updates, 20 min-min noise steps/image, train-only error evaluation |
| REM-Det | 11,004.786 | 2 | about 6.1 h | defender 10 × EOT 5 × PGD 10, then adversarial surrogate PGD 10 per image |
| EntF-Push-Det | 11,301.259 | 2 | about 6.2–6.3 h | 300 projected Push steps/image, one retained reference centroid |
| SEM-Det | 1,307.754 | about 20–21 | about 7.3–7.6 h | defender 10 × 5 uniform stability draws, then surrogate PGD 10 per image |

The original one-cycle generation records supply exact forward/backward,
optimizer, PGD, stability, and Push counters. The budget run's expected total
counters are derived from its **actual complete cycle count**. EntF centroid
forward calls occur once; all repeated Push calls scale with cycles. No
partially completed cycle may be presented as a full-generation result.

## Gates and interpretation

Before this new official sequence, validate the changed source locally, run
one isolated production-reaching GPU smoke with official metrics and artifact
ingestion disabled, then obtain one independent scientific implementation
review of the new schedule and perturbation continuation. Each method then
passes the same complete materialization audit: 16,551 train images, 4,952
byte-identical clean validation images, all stems and labels, every person GT
box changed, maximum 8/255 person-only and 4/255 overlap pixel change, zero
outside person support, and byte-identical non-target-only train images.
Only after that gate does the 200-epoch victim start automatically.

Compare clean VOC20 per-class AP50, person AP50 and Recall, non-target mean
AP50, and all-class mean AP50 against the matched clean baseline **and**
the original one-cycle result for the same method. Do not use AP50-95 or
train-error as a substitute for these victim metrics. A metric change is an
effect of this longer budget variant; it does not establish causal background
misdirection.
