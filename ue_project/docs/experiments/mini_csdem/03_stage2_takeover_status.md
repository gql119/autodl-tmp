# Mini CS-DEM Stage 2 takeover status

## Repository and remote

- Branch: `codex/mini-csdem-3stage`
- Takeover HEAD: `1a71baac75465780378fc6700249f043e1df707b`
- Stage 1 commit: `1a71baac75465780378fc6700249f043e1df707b`
- The existing local tracking ref `origin/codex/mini-csdem-3stage` contains the Stage 1 commit.
- A fresh `git fetch origin` on 2026-07-11 timed out after approximately 74 seconds (exit 124). The latest remote state is therefore not verified; this is a network timeout, not an authentication failure.
- Stage 2 code and results were uncommitted at takeover. Unrelated pre-existing untracked files are preserved and will not be staged with Stage 2.

## Running tasks at takeover

- No Python, mini-CS-DEM, tmux, screen, or shell job was running.
- `nvidia-smi` showed no compute process and approximately 595 MiB display memory use on GPU 0.
- No duplicate experiment was started.

## Existing Stage 2 ablations

All four ablations are complete. Each has a metrics JSON, per-class CSV, 800-image poisoned dataset manifest, victim checkpoint hash, and a 15-row victim `results.csv`. Their victim `args.yaml` files record `epochs: 15`, `pretrained: false`, `resume: false`, and `seed: 0`. The common victim initial hash is distinct from each trained surrogate final hash.

| Variant | Status | target mAP50 | non-target mAP50 | person-free NT | person-cooccur NT | Last alignment coverage | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| A Stage 1 | complete | 0.06423 | 0.01698 | 0.01424 | 0.01834 | 0.99039 | `results/mini_csdem/stage2_ablation_A_stage1_metrics.json` |
| B logits | complete | 0.07425 | 0.01671 | 0.02017 | 0.01505 | 0.98949 | `results/mini_csdem/stage2_ablation_B_logits_metrics.json` |
| C logits+box+DFL | complete | 0.11057 | 0.01980 | 0.01512 | 0.02850 | 0.99588 | `results/mini_csdem/stage2_ablation_C_logits_box_dfl_metrics.json` |
| D all including TAL | complete | 0.07524 | 0.01894 | 0.01538 | 0.02458 | 0.99274 | `results/mini_csdem/stage2_ablation_D_all_metrics.json` |

No duplicate result directory or interrupted run marked successful was found. Formal `stage2_seed0` and `stage2_seed1` outputs did not exist at takeover.

## Shared split and next work

- Split: `data_splits/mini_csdem_voc_seed0.json`
- SHA256: `BC1FFAD0D4A6167BB6882E1D137AB615DBF64E4F9C7589361BFB7518332216CD`
- Train/validation: 800/200 images.
- Train person-present/person-free: 400/400; validation: 100/100.
- The manifest resolves `person` to class ID 14 and records per-class instances and person co-occurrences.

Variant D is selected for the formal Stage 2 run. An initial selection of C based on mean non-target AP was rejected after the required per-class check: C's worst non-target drop is 0.36402 versus 0.36009 for A, whereas D improves it to 0.25587, improves co-occurrence AP, and limits target rebound to 0.01101. The completed C formal seed-0 run is retained as failed-selection evidence under `stage2_formal_C_seed0_*`; it is not overwritten. TAL's raw gradient remains small and is treated as a mechanism risk, but D differs materially from C in the short ablation and therefore must be tested rather than dismissed from the raw gradient alone. Next: run a fresh D formal victim for seed 0 and, if the gate permits, seed 1; then write the Stage 2 report and update the summary.
