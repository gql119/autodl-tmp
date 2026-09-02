# TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2 review handoff

## Decision

No-card review **passed**. The only authorized next execution is one guarded
eight-step GPU G1 at exact commit
`7978182b620707525d84870c73b49ef5bd923dba`. G2 and M1 remain closed.

## Frozen boundary

The revision changes only the strict complete-update router. Dataset-level
KL/JS ranking, the frozen G0 bank and replay, e1/e5/e20 protection snapshots,
P1 carrier, DLFC/CICR/target/NLA/DG-CAIP losses, sequential gradient extraction,
nonlinear backtracking and downstream victim protocols are unchanged.

The v2 route solves safe equalities and violated non-worsening half-spaces in
float64, enforces final target progress of at least `0.60`, and audits the exact
parameter-dtype update. The v1 route remains the default and retains its legacy
step schema.

## Validation evidence

- Remote environment: Python 3.8.10, PyTorch 2.0.0+cu118, no visible GPU.
- Exact isolated checkout: `7978182b620707525d84870c73b49ef5bd923dba`, clean.
- Syntax compilation passed for the router, strict step, experiment, evaluation,
  config validation and G1 controller modules.
- Focused router, strict-step, G1 binding, configuration, dataset-risk,
  proxy-agreement and P4 evaluation tests: **71 passed in 7.17 seconds**.
- A broader suite was hard-stopped after **367 passed** because unrelated CPU
  integration tests exceeded 30 minutes. Before interruption it exposed two
  pre-existing P4 evaluation `NameError`s. Their root cause was one historical
  drop block placed in the wrong comparison function; the block was moved back
  without changing any threshold and the affected tests are included in the
  71-pass final run.
- `git diff --check` passed for the implementation revisions.

## Final preflight binding

- Status/schema: `passed` / `tausb.dgcaip-g1-strict-preflight.v2`.
- Spec: `TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2`.
- Config SHA-256:
  `71f631e2feb6343afeb494f001fc7300915154ecaed31b5b0d870795d86bba1b`.
- Preflight evidence:
  `/root/autodl-tmp/tausb-dgcaip-preflight/TAUSB-SDH-DGCAIP-S0-DSR-SCGR-V2-G1-R1-7978182.json`.
- Evidence SHA-256:
  `389cecc1190e14327c48a2661febcf5462e3237fdb17ca04b909a1cb1f928825`.
- G0/P1 coverage: `1.0`; replay slots: `32`; all risk-bank, manifest,
  P1-state and e1/e5/e20 snapshot hashes matched.
- Storage free at final preflight: `7,708,569,600` bytes.
- Artifact, control, log, cache and tmp roots were rechecked and remained fresh.

## Runtime boundary

The GPU run must use the existing guarded controller, a 20-minute wall cap and
shutdown on every terminal path. It must retain pass or fail evidence and must
not tune on-GPU, reuse the failed v1 state, or continue into G2/M1 in the same
boot. Scientific effectiveness is still unproven until this real G1 completes.

## GPU G1 terminal result — 2026-09-02

The exact approved commit completed normally on an RTX 4090D. The controller
reported exit code `0`, observed a GPU process, finished in `35.17` seconds and
automatically shut the instance down. The fatal scan was clean and the complete
minimal evidence set was pulled back with matching controller hashes.

The scientific gate failed. All eight v2 linear routes were feasible and passed
safe/violated dot audits, but nonlinear backtracking accepted only one update;
the skip ratio was `0.875`. The other failed gate is an exact-comparison defect:
the accepted target progress was `0.599999964`, within the router's frozen
`1e-6` tolerance but below a separate exact `>=0.60` final comparison.

Root-cause inspection found a more important constraint mismatch. Routing uses
one combined NLA + DG-CAIP gradient row per snapshot/class, whereas nonlinear
backtracking independently checks probability, IoU, alignment and JS. A
non-worsening combined row cannot guarantee that every checked component is
non-worsening. Therefore increasing backtracks or broadly relaxing tolerances
is not approved as the next repair.

The full evidence analysis is at
`research_workspace/experiments/TAUSB-SDH-DGCAIP-S0-DSR-SCGR-V2-G1-R1/analysis/HEN.md`.
G2 and victim training remain closed. The next revision must align gradient-row
and nonlinear-check granularity and must harmonize the post-cast progress
tolerance before another single G1 can be proposed.
