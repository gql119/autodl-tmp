# TAUSB-SDH-DGCAIP-R3-DIAG-v1

## 1. Status

- SpecID: `TAUSB-SDH-DGCAIP-R3-DIAG-v1`
- Status: `approved`
- Approved: `2026-08-26` (explicit user approval in the active task)
- Scope: diagnostic mechanism run only
- Parent evidence: `TAUSB-SDH-DGCAIP-S0-R2-MECHANISM`
- Target class: `person` (`class_id=14`)
- Dataset/model protocol: inherit the frozen R2 mechanism protocol

## 2. Problem and Current Evidence

R2 completed without NaN, Inf, CUDA OOM, or Traceback, but failed the mechanism
gate. P2-CAIP, P3, and P4-DGCAIP each accepted only 1 of 8 updates, giving a
backtrack-plus-skip ratio of 0.875. P2 does not include the JS constraint, so the
high rejection rate cannot currently be attributed to JS ranking alone. The
current backtracking implementation records only the final candidate values and
does not preserve per-attempt violations, so relaxing tolerances or changing
weights now would be post-hoc tuning without a demonstrated cause.

R2 also reproduced the P1 structure but failed the historical P1 numerical
replay tolerance. It is therefore necessary to separate within-process
nondeterminism from environment-dependent numerical drift.

## 3. Research Question

Which constraint family actually rejects the routed P2/P4 update after nonlinear
backtracking, and is the P1 replay mismatch reproducible within one process?

## 4. Frozen Hypotheses

### H1: common CAIP feasibility bottleneck

The dominant R2 rejection source is a non-JS CAIP constraint family shared by P2
and P4 (probability, IoU, or alignment), rather than the P4-only JS/ranking
mechanism.

H1 is supported only if all of the following hold:

1. all rejected P2 updates have complete, finite traces through the smallest
   attempted step;
2. in at least 5 of the 7 rejected P2 updates, one or more non-JS constraints
   remain violated at the smallest attempted step;
3. one non-JS metric family contributes at least 60% of P2 terminal violated
   constraints, using violation counts fixed before inspecting R3 results;
4. that same family is present among P4 terminal violations in at least 5
   corresponding rejected steps.

### H2: historical replay drift rather than within-process nondeterminism

The R2 historical P1 numerical mismatch is caused by cross-run/environment
numerical drift, not nondeterministic execution inside one process.

H2 is supported only if P1-A and P1-B start from identical state and batch hashes,
match structurally exactly, and pass the existing numerical tolerances
`abs_tol=1e-6` and `rel_tol=1e-4` for every replayed scalar and per-class field.

H1 and H2 are independent. One may pass while the other fails.

## 5. Compared Approaches

### A. Logging-only nonlinear rejection attribution (selected)

Record every candidate examined by the existing nonlinear backtracking function,
without changing routing, candidate construction, limits, tolerances, or the
accept/reject decision. Repeat P1 twice within the same process.

This is the smallest experiment that directly observes the current blocker and
preserves the meaning of R2.

### B. Linearized gradient/Jacobian audit only (not selected)

Inspect gradient cosine, row-space projection, and local first-order feasibility.
This is cheaper, but R2 already shows that projection checks can pass while the
nonlinear candidate is rejected. It cannot identify which evaluated constraint
blocks the actual update.

### C. Immediately relax tolerances or retune weights (rejected)

Changing limits, loss weights, step size, or routing before attribution would mix
diagnosis with method improvement and make a positive result scientifically
ambiguous.

## 6. Frozen Method and Data Protocol

R3-DIAG must inherit the exact R2 values for:

- model/checkpoint and frozen surrogate;
- VOC split and target class;
- seed `0`;
- calibration size `16`, held-out size `24`, batch size `4`;
- mechanism steps `8`;
- perturbation budget `eps=16/255`;
- carrier, embedding, P1/P2/P4 objectives, class balancing, routing, constraint
  limits, tolerance, learning rate, and backtracking schedule;
- no EOT, JPEG, blur, gray, JND, victim training, poisoned-dataset
  materialization, or AP50 evaluation.

P3 is omitted because R2 already showed the same 0.875 rejection ratio and the
diagnostic question is fully discriminated by P2 (no JS) versus P4 (with JS and
ranking). No P4 state is frozen or promoted by this run.

## 7. Diagnostic Implementation Contract

Diagnostics must be behind a default-off switch. With diagnostics disabled,
existing outputs and update decisions must remain unchanged.

For every P2 and P4 mechanism step and every backtracking attempt, record:

- arm, mechanism step, attempt index, and candidate step size;
- finite/non-finite state;
- every evaluated per-class constraint key;
- constraint value, frozen limit, raw margin `value - limit`, and violated flag;
- grouped maximum margin and violation count for `probability`, `iou`,
  `alignment`, and `js`;
- accepted flag and exact rejection reason;
- initial/final parameter hashes and batch/sample hashes;
- routed-gradient norm, final-gradient norm, route mode, null dimension,
  protection-budget ratio, and maximum projected row dot already available in
  R2.

The active accept/reject result must continue to come from the existing decision
path. The trace may observe and independently recompute the result, but it must
not replace it.

P1-A and P1-B must execute from cloned identical initial state on identical
batches inside one process. Record structural and numerical comparisons using the
existing replay fields and tolerances.

## 8. Pre-registered Root-Cause Labels

Exactly one diagnostic label is emitted for H1:

- `caip_common_infeasibility`: H1 conditions 1-4 pass;
- `js_incremental_blocker`: P2 terminal non-JS violations fail the H1
  concentration threshold, while P4 has JS as at least 60% of terminal
  violations in at least 5 rejected steps;
- `ranking_route_shift`: P2 and P4 routed directions differ materially before
  backtracking, defined as median cosine below 0.90, without either preceding
  concentration rule passing;
- `inconclusive_mixed`: none of the above rules passes.

H2 separately emits one of:

- `within_process_replay_pass`;
- `within_process_nondeterminism`;
- `replay_invalid_input_mismatch` when initial-state or batch hashes differ.

Labels are descriptive diagnostics, not success claims for DG-CAIP.

## 9. Metrics and Split

Primary metrics:

- accepted/skipped updates per arm;
- exact agreement between active decisions and trace-recomputed decisions;
- terminal violation count and maximum/median positive margin by metric family;
- number of rejected steps in which each family remains violated at the smallest
  attempted step;
- P2/P4 routed-gradient cosine by corresponding step.

Secondary metrics:

- P1-A/P1-B structural equality and maximum absolute/relative replay error;
- finite-value coverage and trace completeness;
- existing routing diagnostics, reported without changing their thresholds.

Calibration data may drive mechanism updates. Held-out data is used only for the
same mechanism consistency observations inherited from R2. No clean validation
AP50 is generated or interpreted in R3.

## 10. Success Signal

R3-DIAG is mechanically successful only if:

1. trace completeness is 100% and every required value is finite;
2. initial-state and batch hashes match across comparable arms/repeats;
3. trace-recomputed accept/reject decisions match the active decisions exactly;
4. diagnostic mode does not change final parameters, decisions, or existing R2
   summary fields in a fixed miniature regression test;
5. H1 receives one pre-registered label and H2 receives one replay label.

A label of `inconclusive_mixed` is a valid completed diagnostic result, but it is
not evidence authorizing constraint relaxation.

## 11. Independent Failure Signals

Any one of the following makes the run invalid or failed:

- missing attempt, constraint key, value, limit, margin, or decision record;
- NaN, Inf, CUDA OOM, or Traceback;
- mismatched initial-state or batch hashes for a claimed comparison;
- trace observation changes candidate values or active accept/reject decisions;
- active and recomputed decisions disagree;
- feature-off regression changes existing behavior;
- controller terminal state is missing;
- wall-clock hard cap is reached.

Within-process P1 numerical mismatch is an H2 failure, not permission to widen the
replay tolerance.

## 12. Stop Condition and Cost Guard

- One mechanism-only GPU run.
- Expected runtime: less than 3 minutes after initialization.
- Hard cap: 10 minutes, including initialization.
- Stop immediately on any independent failure signal.
- Auto-shutdown must run after success, failure, or hard-cap termination, and
  shutdown evidence must be checked.
- Do not start E20/E200, victim training, or a second GPU run under this Spec.

## 13. Required Artifacts

- frozen resolved config and source commit;
- controller and wrapper terminal status;
- outer log;
- P1-A/P1-B replay report with hashes;
- P2 and P4 per-attempt trace in machine-readable form;
- aggregated rejection-attribution report;
- active-versus-recomputed decision audit;
- H1/H2 labels and evidence fields;
- file hashes for the minimal pulled evidence set.

## 14. Claim Boundary

This experiment may claim only which currently evaluated constraint family is
associated with nonlinear update rejection and whether P1 repeats numerically
within one process. It cannot claim:

- improved target-class unlearnability;
- improved non-target preservation;
- AP50 improvement;
- carrier robustness or visual quality;
- DG-CAIP effectiveness;
- authorization to relax a constraint or train a victim.

Any subsequent repair must receive a new approved Spec whose change is chosen
from the R3 evidence, followed by implementation review before GPU execution.

## 15. Execution Gate

No CSV generation, implementation, remote checkout, or GPU run is authorized
until the user explicitly approves this Spec. After approval, execution must use
the repository mission workflow, pass local tests and pre-run implementation
review, bind the command to an exact branch and commit, and use a new artifact
root that cannot overwrite R2.
