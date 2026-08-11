# TAUSB-SDH-HIDING-SB-v1 review handoff

## Current workflow state

- Approved Spec: `docs/research/specs/TAUSB-SDH-HIDING-SB-v1.md`.
- Durable execution state: `issues/TAUSB-SDH-HIDING-SB-v1.csv`.
- Approved ExpID: `TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25`.
- Frozen baseline: `HIDING-S0-R2` at reviewed method commit `20c35b6`.
- Active branch: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`.
- SB25 implementation and local no-card validation are complete; no GPU run or remote
  SB25 artifact exists yet.

## Objective

Test one parameter only: preserve the Haar LL residual subband and scale LH/HL/HH by `0.25`
before `tanh`. The goal is to reduce the held-out high-frequency energy ratio to `<=0.40`
without losing secret recovery, finite/support/Linf, non-fixed pixel texture or non-target
leakage gates.

RMS CV is descriptive only. The approved work adds no RMS or host-diversity loss and does not
require RMS CV to increase. The old r2 result remains a failure under its original v3 contract;
this revision does not relabel it.

## Claim boundary

This is seed-0 hiding-only mechanical evidence. Even a PASS does not establish detector
mechanism efficacy, target-class unlearnability, non-target AP preservation, perceptual quality,
robustness, transferability or SOTA performance. Mechanism and victim work require a later
independent gate and user decision.

## Execution log

- 2026-08-11: User explicitly approved the revised logic: RMS CV is descriptive, no forced
  sample-wise energy diversity is added, and the spectral bottleneck targets high-frequency
  dependence while preserving the shared carrier identity. The approved Spec and a validated
  10-row CSV were created.
- 2026-08-11: Implemented the single parameter `hf_subband_scale`. `0.25` preserves Haar LL
  and scales LH/HL/HH before `tanh`; `1.0` takes the exact legacy path without a DWT round trip.
  The value is fail-closed in the approved config, recorded in new checkpoints/frozen states,
  propagated through checkpoint reload and clone paths, and omitted from the legacy architecture
  descriptor so existing scale-1 hashes remain compatible.
- 2026-08-11: Revised gate schema v2 continues to report per-channel RMS CV and its diagnostic
  check, but excludes `rms_diversity` from `required_checks`. The legacy v1 gate retains its RMS
  hard check. No RMS/host-diversity loss was added.
- 2026-08-11: No-card local validation passed: 81 focused SDH, non-target-protection and
  evaluation regressions; Python compile; config binding; exact scale-1 output/input/parameter
  gradient rollback; quarter-scale Haar attenuation; finite gradient; and `git diff --check`.
  The first pytest attempt had 24 passes plus 8 setup errors caused only by an inaccessible
  Windows default temp directory; rerunning unchanged tests with a workspace basetemp passed.
- 2026-08-11: Created implementation commit
  `7f59ab84483f02207594fe6bf89ff870035685cd` from exactly 14 scoped task files and pushed it
  normally (non-force) to `origin/codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`. The staged
  credential scan passed; datasets, weights, checkpoints, test basetemps and unrelated dirty
  files were excluded. Independent pre-run review is the active gate; no GPU work is authorized
  by this snapshot alone.
