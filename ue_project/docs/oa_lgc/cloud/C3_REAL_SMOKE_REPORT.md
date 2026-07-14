# C3 Real-YOLO End-to-End Smoke Report

## Result

Status: `real-detector engineering chain pass`.

Authoritative artifact: `artifacts/oa_lgc/cloud/20260714_145816_C3_0/`.

All five required runs passed:

| Run | J / mode | Delta change norm | Coverage median | Valid non-target gains | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 1 / classification | 0.109541 | 1.0 | 3 | 5.36 |
| B | 1 / detection | 0.109541 | 1.0 | 3 | 4.31 |
| C | 3 / classification | 0.087337 | 1.0 | 2 | 4.18 |
| D | 3 / detection | 0.087342 | 1.0 | 2 | 4.27 |
| E | 5 / classification | 0.055418 | 1.0 | 1 | 2.59 |

The base hash remained `25c0ad56...eec28c` before and after every run. Support/query overlap was zero. Native target box and DFL losses were available throughout, and every run had at least one valid non-target gain.

Across 11 main outer steps, target clean gain ranged from -0.00001904 to 0.00689316 and target poison gain ranged from -0.00002119 to 0.00691438. These values are reported as engineering observables, not evidence of attack effectiveness. Target coverage was 1.0 for every step, so gain values were not caused by assignment disappearance.

Nine of eleven active-hinge protect measurements were non-zero, and every run contained at least one non-zero protect-only mixed gradient. The smallest non-zero protect gradient was 2.993689; maximum total gradient was 17.705873. Final maximum absolute delta across the matrix was 0.0039324, below the `16/255` budget.

Run A reproduction was exact in this execution: identical IDs and reference counts, gain max difference 0, and final-delta max difference 0. The optional J=5 detection-head run was not required and was not run.

This stage closes the real-detector engineering chain only. It does not establish target AP suppression, non-target retention, or method superiority.
