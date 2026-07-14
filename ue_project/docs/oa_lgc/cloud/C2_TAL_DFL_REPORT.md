# C2 TAL / Box / DFL Report

## Result

Status: `pass`.

Authoritative artifact: `artifacts/oa_lgc/cloud/20260714_144529_C2_0/`.

| Gate | Result |
| --- | --- |
| Target coverage median >= 0.50 | pass: 1.0 |
| Low-coverage episode ratio <= 0.50 | pass: 0.0 |
| Target box loss available | pass; median 1.544335 |
| Target DFL loss available | pass; median 1.228376 |
| Valid non-target class | pass: bicycle, dog, horse |
| Assignment/box/DFL schema complete | pass |
| Base state unchanged | pass |
| Proxy fallback absent | pass |

Three episodes were evaluated. Their target reference/clean/poison positive counts were respectively 10/10/10, 10/10/10, and 80/80/80. Target coverage minimum and median were both 1.0. Median target assignment Jaccard overlap was 1.0.

These results show that this C2 engineering smoke did not manufacture learning-gain changes by eliminating person positives. They do not show that a trained OA-LGC perturbation is effective; the delta is a small deterministic engineering input and the virtual horizon is J=1.

The full regression command passed: `117 passed in 10.10s`.
