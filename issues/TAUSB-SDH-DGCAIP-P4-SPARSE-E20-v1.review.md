# TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1 review handoff

## Approval and scope

- The user explicitly approved this Spec on 2026-08-30.
- The only authorized GPU outcome is one strict P4 mechanism followed by one
  paired sparse C0/M1 E20 run if and only if the P4 state-integrity gate passes.
- The mechanism scientific gate remains reportable and binding, but it does not
  masquerade as the state-integrity gate and does not suppress candidate AP50.
- This is single-seed diagnostic evidence, not E200, multi-seed, cross-model,
  EOT, robustness, or formal-method evidence.

## Frozen method

1. One low-frequency/high-level-semantic secret is embedded only in person GT
   boxes by the fixed deep-hiding carrier.
2. Detector-LFC concentrates person-instance hidden features.
3. CICR aligns P3/P4/P5 person feature-residual directions with an energy floor.
4. The target objective suppresses person learnability.
5. DG-CAIP ranks co-occurring non-target instances using clean-to-poison
   divergence and geometry/alignment risk.
6. CGR projects target gradients away from non-target constraint rows.
7. Explicit clean/poison non-target response alignment consumes a frozen 25%
   protection budget, followed by at most five nonlinear backtracking attempts.

## Deterministic repair binding

- Runtime repair commit:
  `83f43070f04e2a98401ad17ec098c01d83d96665`.
- Task-level attestation:
  `docs/research/evidence/TAUSB-P1-DET-RESIZE-REPAIR-PASS-83f4307.md`.
- Attestation SHA256:
  `f05f5f9ca255083d3697af69ad47127c28f8349219e1cf50530edd632bc91b3b`.
- The attestation closes only the deterministic resize/writeback execution
  blocker; the P4 and AP50 questions remain unevaluated.

## Local implementation status

- Candidate states are explicitly `P4-DGCAIP` with
  `diagnostic_candidate_ap50_evaluation` scope and cannot load as legacy P1.
- Raw state, candidate state, metrics, scientific decision, integrity decision,
  config, historical P1 inputs, D0, hiding inputs, and repair attestation are
  hash-bound.
- P4 state saving is governed by state integrity in production; the scientific
  decision is preserved independently.
- Sparse E20 remains 16,551 images: C0 points to original JPEGs; M1 writes only
  6,095 person-containing PNGs and points the remaining 10,456 rows to original
  JPEGs.
- Evaluation reports all VOC20 classes and compares each non-target drop with
  the frozen historical P1 E20 comparison
  `fb1041032fc4b3a349bdb1a62e22b92f81fa7f79b44ffc0eb643437ff685340f`.
- The nested sparse controller now emits stage heartbeats so the outer oneboot
  guard cannot kill a healthy silent controller at its first-progress gate.
- The Bash wrapper changes into the repository's `ue_project` directory before
  module execution and retains one unconditional shutdown trap.

## Local evidence so far

```text
Python 3.14 compileall of framework and tests: pass
Python 3.8-compatible compileall: pass
P4 YAML parse and repair-attestation hash binding: pass
shell wrapper bytes: LF
git diff --check: pass (line-ending notices only)
```

The local Windows environment lacks the complete AutoDL dependency set for the
focused pytest suite. That suite, CLI help, Bash syntax, exact input hashes,
data-disk paths, fresh output roots, and real config loading remain required in
the remote no-card gate.

## Git isolation requirement

The current working branch descends from local evidence commit `3dc2aae`, which
must not be pushed as part of this code snapshot. After local review, the task
changes must be committed explicitly and cherry-picked onto a clean branch based
on `83f43070f04e2a98401ad17ec098c01d83d96665`. No existing worktree or evidence
is to be deleted or rewritten.

## Next gate

1. Complete local static and regression review.
2. Create and push the clean task snapshot.
3. Run the remote no-card audit on the exact commit.
4. Write `pass_allow_run` pre-run evidence only if every no-card gate passes.
5. Ask the user to enable GPU only after step 4.

No GPU execution is currently authorized by this handoff.
