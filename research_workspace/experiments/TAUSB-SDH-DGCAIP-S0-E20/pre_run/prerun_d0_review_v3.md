# DG-CAIP D0 Pre-run Review v3

## PRERUN-REVIEW-D0-03

- Result: pass
- Decision: allow_run
- Gated run: `REMOTE-D0-01`
- Code snapshot: `81bf37e5b19b318ffbfee18edbbf2071e69702dc`
- Branch: `codex/tausb-sdh-dgcaip-cgr-e20-v2`
- Intent: run the read-only D0 locator on frozen person-cooccurring held-out
  instances; do not update the hiding state, adapter, carrier, or model.
- Code location: `ue_framework.tools.run_tausb_sdh` dispatches `stage=mechanism`
  to `run_dgcaip_pilot`; `dgcaip.run_mode=d0` selects the no-update D0 path.
- Parameter data flow: D0 YAML -> config validator -> frozen hiding/P1 input
  validation -> clean real-TAL instance aggregation -> Bernoulli-JS/KL -> raw
  positive probability/IoU/relative-alignment damage -> Spearman and Q1-Q4
  locator report.
- Runtime state: the surrogate and historical checkpoints are frozen; D0 runs
  under `torch.no_grad()` and has no optimizer step.
- Sink effect: D0 writes `d0/d0_locator.json` and status/log evidence only. Raw
  positive damage feeds the locator, while tolerance-aware hinge losses remain
  confined to optimization and nonlinear backtracking paths.
- Baseline/disable path: D0 is diagnostic-only. The later mechanism template
  remains fail-closed until a real passed D0 report and SHA256 are bound; P1-R
  remains bound to historical P1 state/metrics hashes and replay tolerances.
- Local validation: 41 focused tests and 170 broad SDH/DG-CAIP/NLA/CGR tests
  passed; six changed Python files passed AST/in-memory compile; D0 config, two
  module CLIs, and both Bash scripts passed.
- Remote exact-snapshot probe: detached Python 3.8 checkout at commit
  `81bf37e5b19b318ffbfee18edbbf2071e69702dc` passed config validation,
  module CLI import, a raw-damage/zero-hinge locator probe, finite poison-only
  backward, Bash syntax, clean-worktree, input-hash, split-hash, and
  non-overwrite checks. The remote environment has no `pytest`, so the local
  170-test suite was not duplicated remotely.
- Run command binding:

  ```bash
  EXECUTION_COMMIT=81bf37e5b19b318ffbfee18edbbf2071e69702dc \
    /bin/bash \
    /root/autodl-tmp/tausb-dgcaip/preflight-checkouts/81bf37e5-d0-review/research_workspace/experiments/TAUSB-SDH-DGCAIP-S0-E20/pre_run/dgcaip_d0_tmux_launch_v1.sh
  ```

- Experiment validity: VOC20, target id 14, 16,551 train images/labels, 6,095
  person images, seed 0, clean real-TAL assignments, no EOT/JPEG/blur/gray, and
  frozen D0 split hash are all bound and verified.
- Output non-overwrite: formal checkout, artifact, control, cache, temporary,
  outer log, prelaunch-failure log, and tmux session were all absent during the
  review. The launcher refuses any collision instead of deleting or reusing it.
- Recoverability/secrecy: detached checkout, branch/commit, session, outer log,
  controller status, terminal record, 1,200-second outer timeout, and all-terminal
  shutdown are fixed. All growing paths and caches are on `/root/autodl-tmp`;
  no credential is embedded in the command or repository.
- Blockers: none for `REMOTE-D0-01` once a GPU instance is explicitly enabled.
- Validation gaps: no real D0, mechanism, victim training, or AP50 evidence exists
  yet. A D0 pass would establish only locator quality on the frozen surrogate.

## Verified hashes and inputs

- D0 config SHA256:
  `3f570e76c081a2fa08901c198fb1a0cc2a7e62544885ceed5ddae6cca7855a77`
- D0 controller SHA256:
  `1c861d881257fddfc6dfaae7ace30c1ea91f1d1dfe285311192a8c045de5606c`
- D0 launcher SHA256:
  `41acec28d9fbdd5e13a4208122ea7cb3056d3b7947b5b77143fa170da1a613a1`
- Canonical secret manifest SHA256:
  `a25277499e07310e68a39277461f176dd0d8666e69a4b890328d7b913601ac3e`
- Split SHA256:
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`
- P1 state SHA256:
  `2e102026a9356116de38acb1f5056bf5728afcd453e3447b516d4222f4d70b81`
