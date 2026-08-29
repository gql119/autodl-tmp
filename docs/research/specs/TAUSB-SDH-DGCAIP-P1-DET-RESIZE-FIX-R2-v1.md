# TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-R2-v1

## 1. Scope

This is an evidence-driven, narrow repair amendment to
`TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1`. All method semantics, data bindings,
losses, thresholds, strict determinism settings, storage gates, and automatic
shutdown requirements from the parent Spec remain frozen.

## 2. First-boot evidence

The reviewed first GPU gate at commit
`a9e5448e2e5d8faa50d160a7e53b5bdd274b46a1` terminated after G0 in seven
seconds with label `resize_forward_or_gradient_mismatch`. The complete probe
evidence records:

- `bitwise_exact=true`;
- identical forward hashes in all three repeats;
- identical input-gradient hashes in all three repeats;
- finite execution with no deterministic-operator error, NaN, Inf, OOM, or
  traceback;
- `max_forward_abs_error=9.298324584960938e-05` for an unbounded
  standard-normal probe source.

G1 and G2 did not start. The five-file evidence set is preserved under the
experiment's `remote_artifacts` directory with local/remote SHA256 equality.

## 3. Root cause

The production resize input is `output.delta`, which is bounded by
`epsilon * tanh(...)` with frozen `epsilon=16/255`. The G0 source instead used
`N(0,1)`, outside the reachable production domain. In addition, interpolation
coordinates were constructed in float64 and then cast to the float32 input,
while the reference float32 bilinear implementation uses float32 coordinate
arithmetic. A no-card comparison on the four frozen real person-box sizes
showed:

- old production-range maximum error: up to `2.4922192096710205e-06`;
- input-dtype coordinate maximum error: at most
  `9.238719940185547e-07`;
- the approved absolute threshold remains `2e-6` and is not relaxed.

## 4. Authorized repair

Only the following changes are permitted:

1. construct fixed interpolation coordinates and weights in the input
   dtype/device instead of float64 followed by a cast;
2. construct the deterministic G0 source in the frozen reachable interval
   `[-epsilon, +epsilon]` and record both the configured bound and measured
   maximum absolute value;
3. add CPU regression coverage for the four frozen real box sizes in the
   production epsilon range.

The separable two-matmul implementation, `2e-6` parity threshold, 32-iteration
benchmark, G1/G2 logic, 480-second hard cap, and all scientific modules remain
unchanged. No change to SDH, D-LFC, CICR, target objective, NLA, DG-CAIP, CGR,
backtracking, model weights, split, or optimization hyperparameters is allowed.

## 5. Gates

Before another GPU boot:

- focused and adjacent no-card tests must pass with one CPU thread;
- real-box production-range forward parity must pass at `max_abs <= 2e-6`;
- gradient parity, reachability, epsilon/support, config, controller, Bash,
  binding, and clean-checkout gates must remain green;
- the exact commit and config SHA256 must be pinned and pushed.

After those checks, exactly one additional GPU boot is allowed with a unique
commit-derived artifact/control root. It reuses G0, G1, and G2 in order, keeps
the 480-second total hard cap, and shuts down on every terminal outcome. There
is no automatic retry. Only `repair_pass` permits preparation of paired E20.
