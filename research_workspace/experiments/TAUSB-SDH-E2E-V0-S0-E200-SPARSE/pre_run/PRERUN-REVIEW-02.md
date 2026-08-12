## PRERUN-REVIEW-02

- Result: pass
- Decision: allow_run
- Gated run: only a fresh retry of `SPARSE-E200-S0-R1`; runtime execution commit remains `36f74cab2222f41cb1f206b42db3118237f18a52`.
- Trigger: the first launch attempt stopped before checkout/tmux/controller because the launcher incorrectly treated `/root/autodl-tmp/ue_project` as the Git worktree root. Its prelaunch shutdown trap powered off the instance; no E200 training process or experiment output was created.
- Root cause: the remote repository root is `/root/autodl-tmp`, while `ue_project` is its project subdirectory. The failed assertion was therefore an orchestration-path error, not a dataset, P1, GPU, model or training-code failure.
- Fix snapshot: launcher/review commit `96264f3edb5132234e505cb6e4afce9eb5c196af`; launcher SHA256 `dafe7b93823eb3ba18fde623a53aec5ce67149ca80ce5d563e01aec0753cab7f`.
- Fix: `SOURCE_REPOSITORY=/root/autodl-tmp`; repository detection uses `git -C ... rev-parse --is-inside-work-tree`; the exact detached runtime checkout remains on the data disk. A prelaunch failure record now includes exit code, failed line, failed command and execution commit before shutdown.
- Local validation: Git Bash `bash -n` passed; exact repository-root and failure-log strings were inspected; run-contract JSON parsed and preserved the execution commit; scoped diff check passed.
- Preserved controls: exact runtime commit and hashes, frozen P1/surrogate, unique fresh paths, visible-idle-GPU gate, data-disk checks, 9-hour outer cap, per-arm caps, fatal-log/idle checks, terminal evidence retention and automatic shutdown are unchanged.
- Output non-overwrite: the retry still refuses checkout, binding, C0/M1, control, log, comparison, cache, temp, outer-log and prelaunch-failure-log paths if any already exist. The failed attempt created none of these contract paths.
- Blockers: the AutoDL instance is powered off after the safe prelaunch failure. Retry is allowed only after the user enables GPU mode again and the full live preflight passes again.
- Validation gaps: the corrected remote Git-root assertion and fresh paths must be rechecked on the next live instance; actual E200 progress and metrics remain unobserved.
