# Baseline Namespace

Legacy-best and ALCE-era code is intentionally kept in its original modules for
compatibility with the existing best configuration. The new P1/P2 trajectory
methods live under `ue_framework.methods.learning_trajectory` and do not import
ALCE, RLCP, or context-prototype modules.

No files were moved or deleted in this pass because the working tree already
contains unrelated modifications and large untracked experiment artifacts.
