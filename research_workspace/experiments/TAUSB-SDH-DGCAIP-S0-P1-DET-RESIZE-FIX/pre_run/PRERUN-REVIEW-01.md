# TAUSB P1 Deterministic Resize Repair — Pre-run Review

## Review status

- SpecID: `TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-P1-DET-RESIZE-FIX`
- Pre-run review: `PASS`
- Remote no-card gate: `PASS`
- GPU gate: `READY, AWAITING USER GPU MODE`
- Reviewed runtime implementation commit:
  `f758355ea50b44f1576c6efdee47ea721342de75`
- Frozen config SHA256:
  `0294f29190b60b168afc54ac25e41eb5509a6103ceddf095bc713281a9480900`

This PASS authorizes only one approved G0→G1→G2 GPU gate. It does not authorize
P2/P4, dataset materialization, victim training, AP50, E20/E200, method tuning,
or an automatic retry.

## Reviewed repair boundary

1. `crop → canonical host` retains bilinear forward under an explicit no-grad
   boundary; host pixels still condition the carrier.
2. Only the gradient-reachable `output.delta → person box` resize changes to
   fixed align-corners-false interpolation matrices and two matmuls.
3. Normal, strict, and production paths share the same resize implementation.
4. SDH carrier, D-LFC, CICR, target objective, NLA, DG-CAIP, CGR, and nonlinear
   backtracking definitions and weights remain frozen.
5. No CPU round trip, detach of adapter output, nearest/grid-sample fallback,
   warn-only determinism, framework upgrade, or old TAUSB/ALCE path was added.

## Critical issues corrected before PASS

- The writeback frozen-module check now hashes the complete snapshot manifest
  instead of reading a nonexistent `sha256` field.
- The saved two-step smoke state is reloaded into a fresh carrier and its
  adapter hash must match before `state_loadable=true`.
- G0 timeout is classified as `performance_gate_failed`; writeback invariant
  failure receives a registered repair label.
- The launcher creates the control-log parent before redirection and requests
  shutdown if launch preflight fails.
- G1 executes only normal-reset diagnostics and strict-fresh scientific replay;
  strict bitwise equality remains the pass requirement.

## No-card evidence

- exact detached checkout and clean status: PASS;
- focused plus adjacent pytest: `98 passed in 5.71s`;
- Bash syntax: PASS for controller and launcher;
- production artifact and input hashes: PASS;
- dataset counts and deterministic split: PASS;
- artifact root absent and no GPU process launched: PASS;
- existing dirty R4 evidence and the original remote worktree were not touched.

## GPU execution contract

- one boot only, total hard cap 480 seconds;
- G0 deterministic resize microprobe, at most 60 seconds;
- G1 normal-reset diagnostic plus strict-fresh A/B replay, zero updates;
- G2 at most two real P1 writeback steps;
- artifact cap 100 MiB, all growing paths on `/root/autodl-tmp`;
- every controller and launcher terminal path requests AutoDL shutdown;
- exactly one registered primary label is retained and reported;
- only `repair_pass` permits preparation of the paired E20 effectiveness Spec.
