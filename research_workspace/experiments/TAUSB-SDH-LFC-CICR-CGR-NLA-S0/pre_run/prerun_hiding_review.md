## PRERUN-REVIEW-HIDING-1

- Result: pass
- Decision: allow_run, conditional on the launch-time GPU gate in
  `verify_and_launch_sdh_hiding.sh`; no direct Python or tmux command is allowed.
- Gated run: single-secret SDH hiding pilot only. It must not run mechanism arms,
  materialization, victim training, evaluation, or robustness transforms.
- Code snapshot: branch `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`, commit
  `e3f674497569087a79dd3782fcdfbabd4e7c8d04`.
- Intent: validate whether the fixed unseen building/sky secret can be recovered on
  held-out person hosts without collapsing to one host-independent pixel pattern, under the
  approved 20-minute cost cap. This is mechanism evidence only and cannot support an
  unlearnability or AP50 claim.
- Code location:
  - `ue_project/ue_framework/tools/run_tausb_sdh.py`
  - `ue_project/ue_framework/methods/sdh_experiment.py`
  - `ue_project/ue_framework/methods/semantic_hiding_carrier.py`
  - `ue_project/ue_framework/methods/semantic_hiding_validation.py`
  - `ue_project/ue_framework/configs/tausb_sdh_mechanism_v3.yaml`
- Parameter data flow: `python -m ue_framework.tools.run_tausb_sdh --stage hiding`
  reads the frozen mechanism YAML, validates seed `0`, VOC20 target id `14`, no EOT/JND,
  exact dataset/model/secret hashes and the 20-minute cap, then calls
  `run_hiding_pilot`. That path builds a deterministic calibration/held-out person split,
  loads three pretrain secrets plus unseen `bg-building-sky-09`, optimizes the DWT/coupling
  carrier with `hiding_pretrain_step`, evaluates recovery/diversity/spectrum/leakage, and
  applies `evaluate_hiding_gate`.
- Runtime state: hiding uses `120` steps, batch `8`, learning rate `2e-4`, cover weight
  `0.01`, device `cuda:0`, one frozen audited VOC20 surrogate, and artifact root
  `/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0`. It does not instantiate or train a
  victim. The final primary secret is index `3`; pretrain secrets are decoder anti-collapse
  inputs only.
- Sink effect: the only formal sinks are
  `status_hiding.json`, `hiding/hiding_checkpoint.pt`, `hiding/hiding_metrics.json`, and
  `hiding/split_manifest.json`. The next mechanism row remains gated on the hiding result;
  no poison dataset or AP50 metric is produced here.
- Baseline/disable path: formal `tausb_sdh` config requires deep hiding, D-LFC, CICR, CGR and
  NLA switches and fails closed when disabled; the method factory never maps it to
  `tausb_mask`. The hiding-only CLI has exactly two explicit stages and the reviewed wrapper
  passes only `--stage hiding`.
- Local validation: the focused SDH suite passed `56` tests after the portable-hash fix; the
  current carrier/validation/config subset passed `18` tests; the two remote-input path
  assertions passed `7` tests; compile/config parsing passed. Previously recorded touched
  legacy regression paths passed `26` tests. Direct script-file CLI invocation was rejected
  during review and replaced by the pinned module entrypoint.
- Minimal probe: local and AutoDL no-card module `--help` both passed from the reviewed
  project root. AutoDL directly validated the config, canonical secret-manifest hash, four
  `(3,256,256)` secret tensors, primary index `3`, exact VOC/checkpoint paths, and surrogate
  SHA-256. Earlier local hiding validation performed a finite 60-step CPU optimization smoke;
  this is not a substitute for the held-out GPU gate.
- Run command binding:
  - wrapper target: `/root/run_sdh_hiding_cost_guard_e3f6744.sh`
  - wrapper SHA-256:
    `99fe8cbcac2a82d8af20b7df5f165688a1b090c586261233e3ec231e3c3f6419`
  - launch-gate target: `/root/verify_and_launch_sdh_hiding_e3f6744.sh`
  - launch-gate SHA-256:
    `39a07c572554e1b0e9b4199a20405118ded9369fec9a51e2299d6c5329ca8baa`
  - only allowed launch command:
    `/bin/bash /root/verify_and_launch_sdh_hiding_e3f6744.sh`
  - tmux session: `tausb-sdh-hiding-s0-e3f6744-r1`
  - log:
    `/root/tausb-sdh-control/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/hiding-e3f6744-r1/hiding.log`
- Experiment validity: remote and local manifests match for 16,551 train images/labels,
  4,952 val images/labels, person-image counts `6095/2007`, all 20 class counts, label
  contents and image path/sizes. All 21,503 remote VOC images have zero SHA-256 overlap with
  the four secret sources. The secret manifest canonical hash is
  `a25277499e07310e68a39277461f176dd0d8666e69a4b890328d7b913601ac3e`;
  the surrogate hash is
  `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`.
- Output non-overwrite: the formal artifact root, run-specific control root and tmux session
  were all absent during review. The launch gate rechecks all three and fails closed. The
  hiding implementation creates its sink with `exist_ok=False`.
- Recoverability/secrecy: wrapper and launch gate pass remote `bash -n`; the launch gate
  requires an exact clean commit, matching wrapper hash, at least 10 GiB free disk, a visible
  unused GPU, passing CUDA/config/input checks, and then starts one tmux session. The wrapper
  has an external `1200s` timeout, a 10-minute no-progress plus idle GPU/CPU watchdog, a final
  evidence snapshot, and calls `/usr/bin/shutdown` after success, failure, timeout, signal or
  post-preflight exception. Launch-gate failure also calls shutdown. No SSH endpoint,
  credential, private key, token, dataset, checkpoint, or source secret bytes are embedded in
  the packet or scripts.
- Blockers: none after `PRERUN-FIX-PATHS-01` and `PRERUN-FIX-ENTRYPOINT-01`.
- Validation gaps: CUDA is intentionally unavailable in the present no-card mode, so the
  launch-time GPU gate has not been executed. Hiding metrics and gate outcome remain unrun.
  AutoDL lacks pytest; active Linux runtime/config/secret loading was validated directly.

The pass decision authorizes only a future execution of the frozen launch-gate script after
the user enables GPU mode. If the gate fails, roots are no longer fresh, hashes differ, a GPU
process already exists, the wrapper cannot request shutdown, or any command differs from the
binding above, the decision reverts to `do_not_run`.
