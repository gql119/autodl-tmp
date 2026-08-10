# TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1 local validation

Date: 2026-08-10

## Passed checks

- Formal CSV: 18 rows, 28 canonical columns, one active dependency, pre-run row before every remote row, final row `REVIEW-01`.
- Focused tests: 27 passed for prototype geometry, component/cross-batch gradients, rank-0/full-rank/selective CGR projection, decision order, matched microtrajectory, frozen config, old v2 config, calibration and graph-free nonlinear margin evaluation.
- Full local regression: 158 passed using a fresh workspace-local pytest base temp. The first run produced 18 fixture setup errors because the Windows default pytest temp root was unreadable; rerouting only the temp root removed all 18 without any code change.
- CLI: `probe_tausb_malc_geometry.py --help` passes and exposes only config/device/source override arguments.
- Compile: all three new active Python modules pass `py_compile`.
- Python 3.8 grammar audit: all three new active Python modules pass `ast.parse(..., feature_version=(3, 8))`.
- Diff whitespace audit: no `git diff --check` findings in active implementation/config/test files.

## Graph and state audit

- Every persisted component gradient and projector record is a JSON tree with no tensor object.
- The optional in-memory raw component gradients are detached CPU float64 tensors with `requires_grad=false`.
- The workflow deletes observation/loss/route objects after every batch and stores only scalar/list records across batches.
- Prototype collection and held-out collection run under `torch.no_grad()`.
- Held-out carrier coefficients and the frozen prototype bank are hash-checked before/after the audit.
- G0 component projection calls the existing CGR router and never calls nonlinear candidate evaluation or updates the carrier.
- The matched trajectory alone calls the existing nonlinear backtracking route; A0/A1 share the same warm state and batch object at every step.

## Validation gap

A full local real-VOC geometry run was not executed: the formal workflow intentionally refuses reduced batch counts, and a complete 64/96-image CPU run would duplicate the bounded remote probe rather than be a cheap smoke. Local tests therefore establish mechanical correctness and regression safety only. Real detector graph, fixed input hashes, CUDA memory behavior, and exact artifact completeness remain mandatory pre-run/remote evidence; no scientific geometry conclusion is claimed locally.
