# TAUSB-SDH-DGCAIP-R4-D0-BINDING-FIX-v1 review handoff

## Approval and scope

- The user explicitly approved this Spec on 2026-08-27.
- The change is restricted to distinguishing the frozen D0 producer SpecID from
  the downstream diagnostic consumer SpecID.
- No objective, gradient route, tolerance, learning rate, candidate,
  backtracking rule, H1/H2 threshold, or feature-off behavior was changed.
- No remote checkout, GPU run, P3, victim training, AP50, E20, or E200 has run.

## R3 evidence preservation

- R3 terminal evidence was preserved in local commit `dbedbf3` before creating
  the R4 branch.
- R3 remains closed as `failed_invalid_pre_mechanism`; H1/H2 were not evaluated.

## Local implementation

- Branch: `codex/tausb-sdh-dgcaip-r4-d0-binding-fix-v1`.
- Runtime now compares `d0_report.spec_id` with
  `dgcaip.expected_d0_spec_id`.
- If the new field is absent, legacy configurations still compare against the
  current consumer SpecID.
- R4 validation requires the exact frozen producer
  `TAUSB-SDH-DGCAIP-CGR-E20-v2`.
- The R4 YAML differs from R3 only in SpecID, ExpID, the explicit producer
  SpecID, and the unique artifact root.
- Frozen config SHA256:
  `7143daef47321275661081a6118f22ff30b0e5d922ba968aaf061d1c0a3b2004`.
- R4 controller and tmux launcher preserve the 600-second cap and automatic
  shutdown on success, failure, or hard cap.

## Local evidence

```text
compileall changed Python files: pass
isolated binding AST regression: pass, 6 cases
controller Bash syntax: pass
launcher Bash syntax: pass
config SHA256 binding: pass
stray plus-token scan: pass
git diff --check: pass (line-ending notices only)
execution CSV: 28 columns, 13 tasks, final row REVIEW-01, error scan clean
```

The isolated regression covers authentic producer acceptance, wrong producer
rejection, split rejection, source-P1 rejection, failed-D0 rejection, and the
legacy fallback.

## Validation gap and next gate

The bundled local Python has no `pytest`, `PyYAML`, or `torch`, so the full
focused DGCAIP test suite cannot run locally without changing the environment.
This is recorded as a validation gap, not as a test pass.

Next steps are:

1. create an exact local R4 commit;
2. obtain explicit authorization for a normal non-force push;
3. run the full focused suite and frozen-input audit in AutoDL no-card mode;
4. perform independent pre-run review;
5. request explicit GPU enablement only if every prior gate passes.

The GPU gate is currently closed.
