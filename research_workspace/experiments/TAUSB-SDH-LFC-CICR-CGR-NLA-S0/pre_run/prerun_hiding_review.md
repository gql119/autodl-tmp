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

## PRERUN-REVIEW-HIDING-2

- Result: pass
- Decision: allow_run, conditional on the launch-time GPU gate in the r2
  `verify_and_launch_sdh_hiding.sh`; invoking its inner Python or tmux command directly is
  prohibited.
- Gated run: `REMOTE-HIDING-02`, single-secret hiding retry only. Mechanism arms,
  materialization, victim training, evaluation, robustness transforms, EOT and JND remain
  out of scope.
- Code snapshot: branch `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`, commit
  `20c35b6b69bca8a04a69608ddaba315e0ab88325`, clean detached AutoDL checkout
  `/root/tausb-sdh-checkouts/20c35b6-r2-worktree`.
- Intent: rerun the approved 120-step building-secret hiding pilot after correcting the
  width-height order used to crop held-out person hosts. The retry must preserve the r1
  failure evidence and must not change any scientific parameter or claim boundary.
- Code location:
  - `ue_project/ue_framework/methods/sdh_experiment.py`
  - `ue_project/ue_framework/configs/tausb_sdh_mechanism_v3_r2.yaml`
  - `ue_project/tests/test_sdh_experiment_hosts.py`
  - `ue_project/tests/test_sdh_pipeline_config.py`
- Parameter data flow: the only authorized command enters the launch gate, verifies the
  exact clean commit and wrapper hash, then starts one tmux wrapper. The wrapper runs
  `python -m ue_framework.tools.run_tausb_sdh --config
  ue_framework/configs/tausb_sdh_mechanism_v3_r2.yaml --stage hiding`. The validated config
  reaches `run_hiding_pilot`; `_first_person_host` now passes actual image width followed by
  height to `_bbox_to_pixels`, and no host is skipped or filtered.
- Runtime state: the retry retains seed `0`, target id `14`, 120 hiding steps, batch `8`,
  learning rate `2e-4`, cover weight `0.01`, device `cuda:0`, the same four secret tensors,
  primary index `3`, and the same VOC/checkpoint hashes. Removing `runtime.artifact_root`
  from the original and r2 YAMLs makes the parsed documents exactly equal.
- Sink effect: only the new formal root
  `/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-r2` may be created. The r1 root and
  control evidence remain read-only; no AP50, poison dataset or victim checkpoint can be
  produced by this command.
- Baseline/disable path: unchanged from review 1. `tausb_sdh` remains fail-closed and cannot
  map to the legacy Fourier/JND carrier. Both wrapper `--stage` occurrences are explicitly
  `hiding`; no mechanism or victim stage is reachable.
- Local validation: three crop regressions pass; the focused SDH suite passes 67 tests; the
  retry-config subset passes 8 tests; the shared support regression passes. The exact three
  r1 failing VOC images produce finite `3x256x256` hosts on the fixed AutoDL checkout. The
  deterministic 64/96 split hash is
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`, with zero empty
  slices on the hash-matched local VOC copy.
- Minimal probe: AutoDL no-card validation on the exact r2 checkout passed root-only config
  equality, config validation, corrected active source, audited VOC/checkpoint paths,
  surrogate SHA-256, portable secret-manifest SHA-256, `(4,3,256,256)` secret-bank loading,
  and primary index `3`. The checkout remained clean because bytecode writes were disabled.
- Run command binding:
  - reviewed code commit: `20c35b6b69bca8a04a69608ddaba315e0ab88325`
  - wrapper: `/root/run_sdh_hiding_cost_guard_20c35b6_r2.sh`
  - wrapper SHA-256:
    `228954570877815c868f314672509c2547657a796c238c6dc1376dc73aa5d37e`
  - launch gate: `/root/verify_and_launch_sdh_hiding_20c35b6_r2.sh`
  - launch-gate SHA-256:
    `f82bac8fd119cfd5527155b6a91d194b4f8227ec24117e775fdef1b1ba907389`
  - only allowed launch command:
    `/bin/bash /root/verify_and_launch_sdh_hiding_20c35b6_r2.sh`
  - tmux session: `tausb-sdh-hiding-s0-20c35b6-r2`
  - log:
    `/root/tausb-sdh-control/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/hiding-20c35b6-r2/hiding.log`
- Experiment validity: dataset/model/secret/split/seed/target settings are unchanged. The r2
  config differs only in the formal output root required to avoid overwriting r1 failure
  evidence. No robustness or transfer claim is enabled.
- Output non-overwrite: the r2 formal root, r2 control root, r2 session, both remote script
  targets and the clean r2 checkout path were verified absent before creation. After script
  upload, the formal root, control root and session remain absent. The r1 formal status file
  remains present with SHA-256
  `a0df443edb1b5e68ce875a2ef96c6ea77680215842851f17d52d69710f3adc0a`.
- Recoverability/secrecy: both uploaded scripts match the frozen hashes and pass `bash -n`.
  The launch gate requires at least 10 GiB free disk, a visible unused GPU, exact CUDA/config
  inputs, clean code and fresh r2 paths. The wrapper retains the external 1200-second timeout,
  the 10-minute no-progress plus idle-compute watchdog, evidence snapshot and shutdown after
  success, failure, timeout or signal. No connection details or credentials are persisted.
- Blockers: none in no-card review. The launch-time GPU gate remains mandatory.
- Validation gaps: CUDA is unavailable in no-card mode, so GPU visibility, free GPU and first
  finite runtime progress are intentionally unverified until the user enables GPU. No hiding
  success metric or UE/AP50 conclusion exists yet.

This pass authorizes only the exact r2 launch-gate command above after GPU mode is enabled.
Any hash, commit, root, session, input, GPU occupancy or command mismatch changes the decision
to `blocked / do_not_run` and the gate requests shutdown.
