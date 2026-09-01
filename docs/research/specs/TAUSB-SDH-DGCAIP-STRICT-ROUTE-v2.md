# TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2

Status: approved on 2026-09-01

## 1. Purpose

G1 R3 proved that the dataset-ranked multi-snapshot implementation runs end to
end, but all eight updates were rejected by the frozen strict-route geometry.
The observed minimum repair norm was 0.3166--0.7575 of the target-gradient norm,
while v1 allowed at most 0.25. This v2 revision replaces that indirect norm
budget with constraints on the quantities the method actually needs:

1. safe non-target directions remain unchanged to first order;
2. already damaged non-target directions do not worsen to first order;
3. the complete update retains a minimum amount of target attack progress;
4. the existing nonlinear audit still requires real non-target protection.

This is a mechanism-only correction. It does not claim fresh-victim efficacy.

## 2. Frozen scope

The following remain byte- or protocol-identical to v1:

- VOC2007+2012, `person=14`, train split and all dataset hashes;
- the single SDH secret carrier, P1 source state, support mask and epsilon;
- the G0 dataset-level KL/JS risk bank, class-wise ranks and 32-slot replay;
- the main frozen target surrogate and clean e1/e5/e20 protection snapshots;
- DLFC, CICR, target objective, NLA and DG-CAIP loss definitions;
- sequential snapshot-gradient extraction introduced after G1 R1;
- five-step nonlinear backtracking and its safe/violated acceptance rules;
- eight G1 optimization steps, seed 0 and 20-minute controller hard cap;
- G2 and G3 protocols, which remain closed until v2 G1 passes.

No EOT, carrier change, ranking change, new loss, victim training, AP50 tuning,
or post-hoc use of validation AP is allowed in this revision.

## 3. Complete-update constraints

Let `g_t` be the flattened target gradient and let the optimizer update be
`omega_new = omega - eta*d`. Constraint gradients are normalized before
routing.

### 3.1 Safe rows

For active non-target constraints that are still within tolerance:

```text
G_safe d = 0
```

The complete update, including every correction term, must stay in this null
space.

### 3.2 Violated rows

For constraints already above tolerance:

```text
G_violated d >= 0
```

Because the actual update is `-eta*d`, this is the exact local non-worsening
condition. v2 removes the arbitrary positive repair floor. Active repair is
still required by the unchanged nonlinear gate: all violated metrics must be
non-increasing and at least one must strictly improve.

### 3.3 Target progress

The final complete update must retain target progress directly:

```text
g_t^T d / ||g_t||^2 >= 0.60
```

This replaces the v1 `repair_norm <= 0.25*||g_t||` rejection rule. Record both
the progress ratio above and `cos(g_t,d)`; the progress ratio is the gate.

## 4. Numerical solver

The v2 solver operates only on detached gradients and must not change loss or
autograd construction.

1. Cast the target and normalized constraint rows to float64.
2. Compute the safe row space with SVD relative tolerance `1e-6`.
3. Start from the target gradient projected into `null(G_safe)`.
4. In that null space, use deterministic cyclic half-space projection over all
   violated rows and the target-progress row, for at most 128 passes.
5. Audit the float64 candidate, cast it to the real parameter dtype, and audit
   the exact float32 update again.
6. If either audit fails, the null dimension is zero, the direction is finite
   but zero, or the half-spaces remain infeasible, return an explicit zero
   update and skip without mutating parameters.

Required post-cast tolerances:

```text
max(abs(G_safe d)) <= 1e-5
min(G_violated d) >= -1e-6
g_t^T d / ||g_t||^2 >= 0.60 - 1e-6
```

The legacy v1 route remains available and numerically unchanged for old
configs. v2 is selected only by an explicit route mode.

## 5. Nonlinear acceptance

For each feasible v2 direction, retain the existing maximum five backtracks:

- every safe metric stays within its fixed limit;
- every violated metric is no larger than its pre-update baseline;
- if violated metrics exist, at least one improves by more than epsilon;
- failure restores the exact pre-step parameters.

Linear feasibility alone is not an accepted update.

## 6. Evidence and gates

Each step must record:

- safe rank and null dimension;
- pre- and post-cast safe maximum absolute dot;
- pre- and post-cast violated minimum dot;
- final target progress ratio and target cosine;
- correction norm ratio as a diagnostic only;
- solver iterations, route mode, feasibility, backtracking trace and acceptance;
- deterministic batch and routed-gradient hashes.

The eight-step v2 G1 passes only if:

1. all bindings, support, frozen-module and finite checks pass;
2. every nonzero final update satisfies all post-cast linear constraints;
3. at least one update is accepted;
4. backtrack-plus-skip ratio is below 0.70;
5. median final target progress across feasible updates is at least 0.60;
6. no candidate from the failed v1 state is treated as a passing state.

Any failed gate remains a scientific result and stops G2.

## 7. Validation and execution

Before GPU:

- reproduce the v1 budget failure with a focused synthetic test;
- test feasible and infeasible v2 intersections, float64/post-cast audits,
  target-progress enforcement, determinism and zero-mutation skip;
- prove old v1 calls and tests are unchanged;
- run focused DG-CAIP tests, compile, config validation and remote no-card
  preflight on a fresh v2 root.

GPU execution is one eight-step G1 mechanism run with a 20-minute hard cap and
automatic shutdown on every terminal outcome. Do not launch G2 or M1 in the
same boot.

## 8. Claim boundary

A passing v2 G1 establishes only that a dataset-ranked complete update can
satisfy the registered local linear constraints and nonlinear replay audit. It
does not establish person AP50 degradation, non-target AP50 preservation,
shortcut learning, multi-seed performance, cross-model transfer or robustness.
