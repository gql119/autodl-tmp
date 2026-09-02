# TAUSB-SDH-DGCAIP-COMPONENT-ALIGNED-ROUTE-v3

Status: approved on 2026-09-02

## 1. Purpose

Strict-route v2 completed normally and made all eight linear routes feasible,
but nonlinear backtracking accepted only one update. Evidence showed that the
linear router used one combined NLA + DG-CAIP row per snapshot/class while the
nonlinear gate checked probability, IoU, alignment and JS separately. A
non-worsening sum does not guarantee non-worsening components.

This revision makes every routed protection row identical to the scalar audited
by backtracking. It also makes the final target-progress gate use the same
post-cast tolerance as the router. It is a mechanism-only correction and does
not claim victim-model efficacy.

## 2. Frozen scope

The following remain unchanged from v2:

- VOC2007+2012, `person=14`, split and dataset hashes;
- SDH carrier, P1 source state, support mask and `16/255` epsilon;
- G0 dataset-level KL/JS bank, ranks and 32-slot replay;
- target surrogate and clean e1/e5/e20 protection snapshots;
- DLFC, CICR, target objective, NLA and DG-CAIP primitive losses;
- target-gradient construction and sequential snapshot extraction;
- safe equality / violated non-worsening routing geometry in float64;
- learning rate `5e-4`, five nonlinear backtracks and their step sizes;
- eight optimization steps, seed 0 and 20-minute controller hard cap;
- G2, M1, victim training, EOT and robustness evaluation remain closed.

No general tolerance relaxation, additional backtracking, carrier change,
ranking change, AP50 tuning or reuse of the failed v2 candidate is allowed.

## 3. Canonical component rows

Each constraint is registered by the exact key
`snapshot/class/family`. Families are:

```text
nla, probability, iou, alignment, js
```

For snapshot `s` and non-target class `c`:

```text
L[s,c,nla] = NLA.per_class_loss[c]
L[s,c,f]   = mean_i(weight_i * DGCAIP.instance_loss[i,f])
```

where `i` ranges over class-`c` instances and `weight_i` is the already frozen
dataset-risk-derived DG-CAIP weight. The same tensor must be used to compute the
gradient row and its detached pre-update scalar. Candidate backtracking must
recompute the same scalar definition from the candidate observation.

Missing families are absent, not represented by fabricated zero tensors. Row
keys must be identical across the gradient maps, current metrics and candidate
metrics. A mismatch fails closed before routing.

## 4. Partition and routing

Partition each canonical scalar independently:

- `js` remains a safe bounded row with limit `baseline + 1e-9`;
- any other component at or below `1e-12` is safe with fixed limit zero;
- any positive component is violated with its pre-update value as baseline.

The actual update remains `omega_new = omega - eta*d`:

```text
G_safe d = 0
G_violated d >= 0
g_t^T d / ||g_t||^2 >= 0.60
```

The v2 float64 SVD/half-space solver, SVD relative tolerance `1e-6`, maximum
128 passes and post-cast audits are unchanged. v1 and v2 modes and schemas must
remain readable and behaviorally unchanged.

## 5. Nonlinear acceptance

Backtracking evaluates exactly the canonical component registry used by the
router. The existing rules remain:

- every safe component stays within its fixed limit;
- every violated component is no larger than its pre-update baseline;
- at least one violated component improves by more than `1e-9`;
- failure restores the exact pre-step parameters.

No combined-loss surrogate may substitute for a component row in v3.

## 6. Target-progress consistency

Both route feasibility and the final scientific gate use:

```text
target_progress >= 0.60 - 1e-6
```

The threshold is not lowered; this only harmonizes the float32 post-cast audit
with the final gate. The recorded raw value remains unchanged.

## 7. Evidence and gates

Each step records the v2 routing diagnostics plus:

- `constraint_row_schema = snapshot_class_family_v3`;
- safe and violated component-row counts;
- the ordered row-name digest;
- component-aligned nonlinear trace.

The single v3 G1 passes only if all frozen bindings and integrity checks pass,
at least one update is accepted, backtrack-plus-skip ratio is below `0.70`, all
accepted updates satisfy the post-cast safe/violated audits, and accepted-step
target progress satisfies the harmonized tolerance.

Any failure is retained as evidence and stops G2/M1.

## 8. Required no-card validation

Before GPU:

1. reproduce the v2 mismatch where a combined row is non-worsening while one
   protected component worsens;
2. prove v3 emits independent NLA/probability/IoU/alignment/JS rows and that
   gradient, baseline and candidate keys are identical;
3. test risk-weighted component scalars and missing-family behavior;
4. test v3 feasible/infeasible routing, target tolerance and zero mutation;
5. prove v1/v2 configs, schemas and route behavior remain unchanged;
6. run focused tests, syntax compilation, config validation and exact remote
   preflight on a fresh v3 root.

Only a passing no-card review may authorize one new eight-step GPU G1.

## 9. Claim boundary

A passing v3 G1 would establish only that component-aligned dataset-ranked
updates can pass the registered local linear and nonlinear protection audit. It
would not establish person AP50 degradation, 19-class preservation, shortcut
learning, robustness, transfer or multi-seed performance.
