# TAUSB-SDH-DGCAIP-S0-R3-DIAG H→E→N

## Hypotheses

- H1 asked which constraint family rejects nonlinear P2/P4 updates.
- H2 asked whether the historical P1 replay mismatch occurs within one process.

Neither hypothesis was evaluated because the run stopped before P1-A, P1-B,
P2, or P4 was constructed.

## Evidence

- The reviewed commit and config were bound exactly to
  `8a54f74e094c7a15fe4bc487b29ea8712bfe426e` and config SHA256
  `3d213af011234cdf08ef0b54d78c377db18ddaf83903d771057b84f418839613`.
- The controller terminated with exit code 1 after 2.81 seconds and requested
  shutdown. No prelaunch failure, NaN, Inf, CUDA OOM, or hard-cap event occurred.
- The runtime raised `ValueError: DG-CAIP D0 report SpecID mismatch.` before
  mechanism optimization.
- The frozen D0 report itself is valid and passed its gate. Its SHA256, split
  hash, and P1-state hash match the R3 config.
- The D0 report records its producer SpecID as
  `TAUSB-SDH-DGCAIP-CGR-E20-v2`, while the downstream diagnostic config records
  `TAUSB-SDH-DGCAIP-R3-DIAG-v1`.
- Runtime line 539 compares those producer and consumer identifiers directly.
  That comparison is the confirmed blocker.
- No backtracking trace, P1 replay report, rejection attribution, mechanism
  metrics, H1 label, or H2 label was produced.

## Interpretation

Outcome: **invalid pre-mechanism run caused by provenance-binding logic**.

This is not evidence for or against CAIP feasibility, the JS constraint, route
shift, or within-process nondeterminism. It also provides no evidence about
person unlearnability, non-target preservation, AP50, robustness, or transfer.

The upstream D0 artifact should be authenticated by its frozen producer SpecID,
file hash, split hash, decision, and P1 hash. Requiring its producer SpecID to
equal every downstream consumer SpecID incorrectly rejects valid reuse across
R2/R3 diagnostic stages.

## Next

Do not rerun R3 under the current Spec. A separately approved R4 repair should:

1. add a frozen `expected_d0_spec_id` provenance field;
2. compare the D0 report against that field, not the current consumer SpecID;
3. retain all existing D0 hash, decision, split, and P1 checks;
4. prove legacy feature-off behavior and mismatched-producer fail-closed behavior;
5. use a new artifact root and one newly approved 600-second diagnostic run.

The draft repair contract is
`docs/research/specs/TAUSB-SDH-DGCAIP-R4-D0-BINDING-FIX-v1.md`.
