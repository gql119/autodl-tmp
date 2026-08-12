## PRERUN-REVIEW-02

- Result: pass
- Decision: allow_run
- Gated run: `SPARSE-E20-S0-R1`; paired sparse E20 only, with no mechanism/smoke rerun and automatic shutdown on every terminal state.
- Code snapshot: detached clean checkout `/root/tausb-sdh-checkouts/e2e-v0-sparse-b70fc87-worktree` at execution commit `b70fc87ecfcda8c2adb5f40b86a1147dbe738633`; wrapper SHA-256 `a18a96de35e130d2af28a3781aa959094b04b8fbf20e2fcb2f9432d4d0c46ce2`; controller SHA-256 `12fe62c8f0c06f5a735d061d4c9d09d3eb1cc120835ebfb3020429a1c4b29215`.
- Intent: obtain the first matched C0/M1 E20 effectiveness evidence while storing only 6,095 target-image PNGs and referencing all unmodified images directly.
- Code location: the active E20-only binder, sparse materializer/list audit, victim TXT consumer, clean VOC evaluator, paired comparison and shutdown wrapper in the reviewed execution commit.
- Parameter data flow: reviewed controller -> hash-verified P1 -> E20-only C0/M1 configs -> C0 original-JPEG list / M1 6,095 PNG plus 10,456 original-JPEG list -> real Ultralytics loader -> fresh victims -> clean VOC20 AP50 -> comparison.
- Runtime state: Python 3.8.10, PyTorch 2.0.0+cu118, Ultralytics 8.4.33, CUDA available, one idle RTX 4090 D with 24,081 MiB free; no compute process was present.
- Sink effect: remote full-VOC probe resolved 16,551 images, 6,095 person images, 0 backgrounds, 0 corrupt labels, and yielded batch shape `[1,3,96,640]`; P1 feasibility state hash is `c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168`.
- Baseline/disable path: C0 remains an independently initialized clean victim with zero poison generator; legacy full-PNG behavior is untouched and not selected by this run.
- Local validation: 92 focused tests plus compile/AST/CLI/Bash/diff/credential gates passed before the execution commit.
- Minimal probe: remote ordered-stem hash `2e0f30546c8848b7f2b9c4239b49ff417dba3d51e2bd54b354fb0f299ea00011`; label-manifest hash `022fbdace84899bf5d340cd07f2eb1d51834c3d0bd35b446f4fef11eb2a53216`; remote path-list hash differs from the Windows local hash only because the approved lists contain absolute platform-specific paths.
- Run command binding: tmux `tausb-sdh-e2e-v0-sparse-e20-s0-r1`; exact environment and roots are recorded in `sparse_e20_run_contract.json`; execution must invoke only the reviewed `sparse_e20_controller.sh` wrapper.
- Experiment validity: target `person` id 14; VOC train 16,551; clean VOC val; seed0; fresh YOLOv8n-style C0/M1; E20; imgsz640; batch36; SGD; 20 named class AP50 values; single-seed feasibility claim only.
- Output non-overwrite: checkout, binding, control, log, comparison, C0 and M1 roots and tmux session were all absent immediately before launch.
- Recoverability/secrecy: one tmux controller, atomic status, per-stage logs, two-hour overall cap, 40-minute materialization/arm caps, ten-minute idle guard, C0 all-zero stop, and shutdown trap. No credential is persisted.
- Blockers: none.
- Validation gaps: real M1 materialization, saved-reload quality, wall time, victim AP50, paired decision and shutdown confirmation remain outputs of the gated run, not pre-run evidence.
