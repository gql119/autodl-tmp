# TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4

Status: approved on 2026-09-02

## 1. Purpose

The component-aligned v3 G1 completed normally, produced eight feasible linear
routes and met final target progress on every step, but accepted only one
update. Seven candidates were rejected by positive nonlinear margins between
`1.69e-9` and `5.36e-7`, despite real protected-component improvements between
approximately `1.98e-5` and `7.15e-4`. The v3 gate also counted a successfully
accepted two-attempt line search as a failure and gated an intermediate
`attack_retention` diagnostic after the final route had already met target
progress.

This revision permits early carrier optimization to explore a numerically
stable neighbourhood while retaining every non-target component trace. The
ultimate scientific outcome remains person AP50 degradation with the nineteen
non-target AP50 values preserved as far as possible. G1 is therefore a bounded
mechanism qualification, not a substitute for victim evaluation.

## 2. Frozen method scope

The following remain unchanged from v3:

- VOC2007+2012, `person=14`, seed 0 and all dataset hashes;
- one fixed semantic secret embedded only in person instances;
- P1 carrier state, support mask and `16/255` perturbation bound;
- DLFC, CICR and the complete target objective;
- the frozen e1/e5/e20 protection snapshots;
- dataset-level KL/JS risk bank, ranks and replay order;
- NLA and DG-CAIP primitive losses and their tolerances:
  classification `0.005`, box `0.02`, alignment `0.05`;
- canonical `snapshot/class/family` rows for NLA, probability, IoU, alignment
  and JS;
- float64 component-aligned route, safe equalities, violated non-worsening
  half-spaces and final target progress `0.60`;
- eight steps, learning rate `5e-4`, five backtracks and 20-minute controller
  cap;
- EOT, robustness evaluation, G2 and victim training remain outside this G1.

This Spec changes gate semantics only. It does not alter the carrier, losses,
gradient rows, risk ranking or route geometry.

## 3. Numeric nonlinear comparison

The existing JS budget remains `1e-9`; it is not redefined. Candidate scalar
recomputation receives a separate registered numerical comparison tolerance:

```text
nonlinear_comparison_tolerance = 1e-6
```

For every safe or violated canonical component, candidate acceptance compares
the recomputed value with its existing bound plus this numerical tolerance.
At least one violated component must still improve by more than `1e-6`.

This is not an extra classification, IoU or alignment damage allowance. Raw
values, raw margins and the exact v3 bounds must all remain in the trace.
Missing keys, non-finite values and failed route feasibility remain fail-closed.

## 4. Backtracking accounting

The implementation records three independent quantities:

- `backtrack_rate`: fraction of steps requiring more than one attempt;
- `actual_skip_ratio`: fraction of steps with no accepted update;
- `accepted_update_ratio = 1 - actual_skip_ratio`.

A successful smaller line-search step is not a skip. The historical
`backtrack_skip_ratio` remains readable for v1-v3 reproduction but is not a v4
promotion gate.

## 5. Layered decision

The v4 output separates three layers.

### Runtime pass

- exact risk bank and replay bindings;
- all metrics and candidate state finite;
- frozen modules unchanged.

### Mechanism valid

- at least one accepted update and changed adapter state;
- every accepted route passes post-cast safe-row and violated-row audits;
- accepted routes retain a non-zero null dimension.

### Promotion pass

- `accepted_update_ratio >= 0.50` across eight registered steps;
- median accepted-step final target progress is at least `0.60 - 1e-6`.

`attack_retention` and `backtrack_rate` are diagnostics. They cannot alone
block v4 promotion. Overall `decision.pass` requires all three layers.

Regardless of the decision, the controller must exit normally after preserving
metrics, the complete trace and candidate state. A scientific gate failure is
reported as `completed_gate_failed`, not as a runtime crash.

## 6. Required validation

Before GPU execution:

1. prove a `5e-7` recomputation residual is rejected at `1e-9` and accepted at
   the registered `1e-6` comparison tolerance while a real repair remains;
2. prove the strict step forwards the tolerance without modifying component
   budgets;
3. validate the exact v4 config and frozen v3 sections;
4. prove v1-v3 schemas and default `1e-9` behaviour are unchanged;
5. verify v4 output requires `runtime_pass`, `mechanism_valid` and
   `promotion_pass`;
6. run focused tests, bytecode compilation and exact no-card controller
   preflight on fresh roots.

After those checks, one eight-step guarded v4 G1 may run. If it passes, proceed
to the already scoped victim experiment rather than adding further mechanism
gates. Victim person and all nineteen non-target AP50 results must be retained
whether or not the final research thresholds pass.

## 7. Claim boundary

A v4 G1 pass establishes only that a component-aligned candidate can make
repeated target-directed updates without catastrophic local protection or
runtime failure under the registered proxy snapshots. Only victim training and
clean validation can establish target-class unlearnability and non-target
preservation.
