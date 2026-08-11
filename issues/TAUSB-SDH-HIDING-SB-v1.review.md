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
- 2026-08-11: `PRERUN-REVIEW-01` passed for exact clean run commit
  `d244c3270eb24d7a6515e79ff643cb015ebb0bb9`. AutoDL no-card probes verified config/secret-bank
  loading, the active quarter-scale Haar sink, finite backward, r2 scale-1 checkpoint
  compatibility, VOC/checkpoint counts and hashes, and fresh SB25 worktree/artifact/control/tmux
  paths. The cost wrapper, input audit and launch gate were uploaded and matched SHA-256
  `1ab2898...c6a6`, `83aa3db...aad9`, and `80020b15...b5e`; both scripts passed local and remote
  `bash -n`. GPU mode was unavailable by design, the launch gate was not executed, and no
  experiment process started. The remote row now waits for the user to enable GPU mode.
- 2026-08-11: After the user enabled GPU mode, the exact launch gate passed every check at
  commit `d244c3270eb24d7a6515e79ff643cb015ebb0bb9`: clean checkout, wrapper/input hashes,
  fresh artifact/control/session paths, 21.49 GB free disk, idle RTX 4090D, Python/Torch/
  Ultralytics/CUDA inputs, secret bank and r2 compatibility. It launched tmux
  `tausb_sdh_hiding_sb25_r1` with no mechanism or victim stage. The first follow-up SSH attempt
  was refused, indicating the instance had already shut down as required. This is not yet proof
  of successful completion: `status_hiding.json`, `hiding_metrics.json`, the guard state and log
  must be read after no-card restart. `REMOTE-HIDING-SB25-01` remains open with
  `shutdown_observed_artifact_pull_pending`; no GPU retry is authorized.
- 2026-08-11: After no-card restart, the canonical status showed `completed`, exit code 0,
  elapsed time `22.2687 s`, `gate_pass=false`, and the cost guard recorded the required shutdown
  request. Nine small required files were pulled by exact SCP paths after the stock manifest
  inventory rejected unrelated `/root` filenames. Local verify-only reported no missing required
  files; canonical status, metrics and split hashes exactly match the ready snapshots. No
  checkpoint, dataset, weights, images or credentials were transferred.
- 2026-08-11: The revised gate failed only `retrieval_top1` (`0.424479 < 0.90`) and
  `primary_l1_margin` (`0.067159 < 0.20`). High-frequency energy passed decisively at `0.034223`,
  recovery SSIM was `0.641143`, pixel cosine `0.682427`, co-occurrence balanced accuracy
  `0.458333`, and non-target macro-AUROC `0.477039`. RMS CV was reported descriptively and did
  not determine the decision. Per the frozen gate, no mechanism, victim or AP50 work was started.

## Objective scientific result

`hf_subband_scale=0.25` is rejected. It corrected the r2 high-frequency failure but removed too
much secret-discriminative capacity: broad reconstruction similarity remained, while bank-level
secret identity retrieval and relative L1 separation failed. This is a spectral-capacity versus
identity-recovery trade-off rather than a Pareto improvement.

## Final evidence

- Transfer report:
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/remote_artifacts/transfer-report.json`.
- Hiding-only summary:
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/hiding-metrics-summary.json`.
- H→E→N analysis:
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/analysis/result-TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25.md`.
- STATE proposal:
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/analysis/state_candidate.md`.

## Final claim/evidence review

Decision: `vision_met` for the approved hiding-only execution contract. The scientific
hypothesis itself is rejected. The implementation was bound to the reviewed commit, the capped
run completed and shut down, all required small evidence was verified, every revised hard gate
was judged without post-hoc changes, and the failure correctly blocked downstream work.

No Current Best or `STATE.md` truth was changed automatically. No detector efficacy, AP50,
unlearnability, non-target preservation, perceptual quality, robustness, transferability or SOTA
claim is made. Missing visualization, PSNR and LPIPS are explicit validation gaps. The proposed
single next experiment (`hf_subband_scale=0.50`) requires a new user-approved Spec.
- 2026-08-11: The user approved the proposed failed-path disposition. `STATE.md` now records
  the r2/SB25 spectral-capacity versus secret-identity trade-off, keeps Current Best unchanged,
  blocks mechanism/victim/AP50, and lists a separately approved `hf_subband_scale=0.50`
  hiding-only Spec as the only candidate continuation. This approval does not itself authorize a
  new GPU run.
