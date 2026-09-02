# TAUSB-SDH-DGCAIP component-route v3 G1 analysis

## Outcome

The guarded GPU G1 completed normally at execution commit
`a316d748ff764c773208780aaa12dcfc3f2a69a6`.

- Controller status: `completed`
- Mechanism exit code: `0`
- GPU process observed: `true`
- Controller elapsed: `50.16 s`
- Mechanism elapsed: `44.26 s`
- Automatic shutdown: observed by the caller before no-card restart
- Scientific gate: `failed`

This is a valid negative mechanism result, not a crash or incomplete run. G2,
M1 and victim training remain closed.

## Integrity

The output used schema `tausb.dgcaip-dataset-strict-mechanism.v3`, route mode
`component_aligned_target_progress_v3` and component-row schema
`snapshot_class_family_v3`. All frozen input, replay, finiteness, adapter-change
and frozen-module checks passed.

Minimal evidence hashes:

- `mechanism_metrics.json`: `41aa37b73a2e11aa4877f0587d749b14c4cce5a0d93eb3a04b6822079a375082`
- `backtracking_trace.json`: `8fcf4df8cee8906e7dfd0bfe140a61191508d60d693776f650b9f8eb7d1fcbd7`
- `p5_dataset_strict_state.pt`: `ca701c98a47655c3a1dbf52390caa541eaeca6eed474f92152e9e44421698ee6`
- `controller_status.json`: `fe0d7ccc1c392569233a15c63d1885f6c176ca5f469936d3e82f0170b48f6818`

## Gate results

Passed:

- risk bank and replay binding;
- at least one update;
- final safe-row orthogonality;
- violated-row non-worsening direction;
- nonzero null dimension;
- final target progress;
- finiteness, adapter change and frozen-module integrity.

Failed:

- median `attack_retention >= 0.60`;
- `backtrack_skip_ratio < 0.70`.

Observed summary:

- accepted updates: `1/8`;
- backtrack-or-skip ratio: `1.0`;
- median attack retention: `0.504758`;
- median final target progress: `0.600000024`;
- safe component rows per step: `20-54`;
- violated component rows per step: `21-57`;
- component constraint rank: `9-21`;
- full perturbation Linf: `0.062744796`, within `16/255`;
- support-outside Linf: `0`.

| step | accepted | retention | target progress | attempts | safe rows | violated rows |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | no | 0.605476 | 0.600000024 | 6 | 30 | 30 |
| 1 | no | 0.322853 | 0.599999964 | 6 | 48 | 57 |
| 2 | no | 0.628705 | 0.600000024 | 6 | 22 | 38 |
| 3 | no | 0.474769 | 0.600000024 | 6 | 54 | 36 |
| 4 | yes | 0.564162 | 0.600000024 | 2 | 39 | 21 |
| 5 | no | 0.435414 | 0.600000024 | 6 | 22 | 38 |
| 6 | no | 0.511374 | 0.599999905 | 6 | 20 | 25 |
| 7 | no | 0.498143 | 0.600000024 | 6 | 42 | 33 |

## What v3 established

All eight component-aligned linear routes were feasible and satisfied the
post-cast safe, violated and target-progress audits. The previous v2 defect—a
combined gradient row being judged against separate nonlinear components—was
therefore removed successfully.

The remaining rejection is at the nonlinear acceptance layer. At the final
backtrack, the maximum positive component margins of the seven rejected steps
were between `1.69e-9` and `5.36e-7`, all below `1e-6`. The current mixed
backtracking tolerance is `1e-9`. The final traces also contained genuine
improvements between `1.98e-5` and `7.15e-4`, so rejection was caused by tiny
simultaneous positive margins, not absence of any protected-component progress.

Final-step residual violations were concentrated in NLA and alignment, with
occasional probability and IoU rows; no final rejection was caused by JS.

## Gate-definition conflict

Two gate definitions are no longer aligned with the v3 route contract:

1. `attack_retention` measures the norm retained after the safe-equality
   projection, not the final routed direction. The final routed direction met
   the explicit target-progress constraint at `0.60` on every step, while the
   median projection-only retention was `0.504758`.
2. `backtrack_skip_ratio` increments when `attempts > 1` even if the update is
   ultimately accepted. Thus step 4 was accepted but still counted as a
   backtrack-or-skip failure. This metric cannot distinguish successful line
   search from a skipped update.

## Evidence-based v4 recommendation

The smallest defensible next revision is limited to acceptance/gate semantics:

- keep the v3 carrier, dataset ranking, canonical component rows, route solver,
  learning rate, five backtracks and target-progress constraint unchanged;
- align nonlinear component comparison with the already used `1e-6` post-cast
  numerical audit, while preserving raw margins in the trace;
- record `backtrack_rate` and `actual_skip_rate` separately, and gate the actual
  accepted/skip outcome rather than penalizing every successful line search;
- make final target progress the authoritative attack-direction gate and retain
  projection-only `attack_retention` as a diagnostic.

Retrospective evaluation of the saved traces shows that all eight final
candidates are within `1e-6` of every component bound and contain a component
improvement larger than `1e-6`. This supports a new registered experiment; it
does not rewrite the v3 result or prove that v4 will improve victim AP50.
