# TAUSB-SIRC-MALC-CGR-MAP50-v2 Draft Handoff

## Current state

- Research scope: authorized academic machine-learning research on public YOLOv8/Pascal VOC
  and user-controlled code/AutoDL. It does not involve unauthorized access, network scanning,
  exploitation, credential collection, malware, persistence, access-control bypass, or service
  disruption.
- Current Git branch: `codex/tausb-sirc-lfc-cgr-map50-v1`.
- Base HEAD: `6262d918f6d2355757c1b8e98e1d6728cf005b69`.
- New draft Spec:
  `docs/research/specs/TAUSB-SIRC-MALC-CGR-MAP50-v2.md`.
- Approval status: `approved` by explicit user message on 2026-08-09.
- The v2 execution CSV is generated after this approval and becomes the only active durable
  execution source; the old v1 CSV remains paused for audit.
- No v2 method code, local validation, AutoDL training, evaluation, artifact generation,
  deletion, or overwrite has been performed.
- The previous approved v1 Spec remains on disk for audit but has
  `execution_state: paused_for_method_revision` and points to this v2 candidate.
- The previous v1 CSV row `CARRIER-01` remains `进行中` only as historical durable state;
  its notes contain `do_not_execute:pending user approval of TAUSB-SIRC-MALC-CGR-MAP50-v2`.

## Why v1 was revised

The original Semantic Deep Hiding LFC is a classification-oriented module. It computes
pairwise cosine distance between flattened final-convolution ResNet-18 features of perturbation
maps `x_pm=x_ue-x_c`, with paper weight `omega_3=1e-4`. It does not model object instances,
clean TAL assignment, P3/P4/P5 scale, co-occurring classes, or detector head structure.

Directly adding an external ResNet-18 LFC would create task mismatch, an extra checkpoint,
and an objective partly duplicated by the existing detector residual CICR. The user requested
a detector-native redesign rather than a backbone substitution.

## Frozen v2 proposal: MALC

MALC means **Multi-scale Assignment-aware Latent Concentration**.

1. Keep one shared SIRC carrier family: 16 bases / 48 RGB coefficients, radial range `[2,24]`,
   `eps=16/255`, instance-canonical rendering, deterministic JND, and forced pseudo fallback
   support.
2. Freeze the existing YOLOv8 surrogate. Only carrier coefficients `theta` are updated.
3. Capture P3/P4/P5 `cv3` classification-tower features immediately before their final class
   convolutions. The `cv2` box tower is monitoring-only.
4. Clean real TAL supplies foreground, assigned target scores, and `target_gt_idx`. PAG selects
   person-relevant detector locations.
5. For each person GT and scale, pool the clean-to-poison `cv3` residual with normalized clean
   assigned-score weights.
6. Fit per-scale direction prototypes, median residual energy, and `0.5 * Q25` energy floors
   only on the fixed calibration split; freeze all of them before optimization/held-out use.
7. MALC uses scale-balanced, instance-balanced terms:
   - residual direction concentration;
   - log residual-magnitude concentration;
   - non-zero energy floor.
8. MALC replaces both external ResNet-18 LFC and standalone CICR. Existing
   `instance_cicr.py` is an implementation starting point, not a second simultaneous loss.
9. Loss weights are not copied from the paper or all set blindly to 1.0. A fixed warm-up
   calibration matches median coefficient-gradient norms to the easy-classification route,
   clips ratios to `[0.1,10]`, then freezes them without using victim mAP.

## Non-target protection

CGR remains the only non-target protection mechanism:

- clean TAL real non-target foreground positives only;
- one class-balanced assigned-class probability-drop constraint per active class;
- tolerance and near-boundary both `0.005`;
- row-normalized constraint gradients and SVD relative threshold `1e-4`;
- null-space projected target step near the boundary;
- repair-only when violated;
- at most five nonlinear backtracks, then skip;
- no scalar non-target distillation, non-target feature loss, ALCE, or late repair;
- non-target box/CIoU is monitoring-only.

## Minimum experiment

Mechanism gate, not UE evidence:

- A0: shared SIRC + easy-classification route + CGR, MALC off.
- A1: exactly matched A0 with MALC on.
- Fixed calibration/held-out split; held-out never updates prototypes or weights.
- A1 must pass the frozen direction, magnitude, energy, scale-coverage, leakage and CGR
  retention gates before M1 is allowed.

Fresh-victim experiment after the mechanism gate:

- C0: clean VOC train, fresh YOLOv8n victim.
- M1: A1 carrier materialized on all 6,095 person-containing train images, followed by an
  independent fresh victim.
- Matched protocol: seed 0, 200 epochs, image size 640, batch 36, SGD, original clean VOC val.
- No EOT or JPEG/blur/gray robustness evaluation in this first run.

## Main success criteria

- Mechanism A1 vs A0: held-out level-balanced residual cosine median gain at least `0.10`;
  log-energy MAD at most `0.90x`; coverage at least `0.80`; zero-norm at most `0.20`;
  floor-pass at least `0.80`; at least two scales remain valid; CGR attack-retention median
  at least `0.20`; repair-plus-skip below `0.50`.
- Fresh victim: person AP50 drop at least `0.30`; non-target macro AP50 drop at most `0.05`;
  at least 16/19 non-target classes drop no more than `0.10`; poisoned count exactly 6,095.
- Single-seed evidence remains tentative. A0/A1 mechanism evidence cannot prove UE efficacy.

## Dirty-worktree boundary

The worktree contains pre-existing user changes and many untracked research files/artifacts.
Do not reset, stash, clean, bulk-delete, or stage unrelated paths. Tracked changes already
exist in `launch_one.py`, `paths.py`, `runtime.py`, `stages/aggregate.py`,
`stages/evaluate.py`, and `stages/train_victim.py`; audit and preserve them.

## Instructions for the next window

1. Read `AGENTS.md`, `ue_project/AGENTS.md`, `research_workspace/STATE.md`, this handoff,
   the approved v2 Spec, the v1 review, and the old v1 CSV.
2. Do not resume the v1 CSV. The approved v2 CSV is the only active execution source.
3. Continue with `mission -> mission-csv-execute` on the v2 CSV; do not mutate the old v1
   CSV into v2.
4. Continue on `codex/tausb-sirc-malc-cgr-map50-v2` while preserving the dirty worktree.
5. Implement in order: carrier/render -> MALC -> gradient calibration -> CGR integration ->
   A0/A1 mechanism gate -> pipeline/config -> named VOC20 AP50 -> local/no-card validation ->
   scoped Git snapshot -> remote-input audit -> pre-run review -> GPU execution.
6. Never launch formal training before `pre_run_result=pass`, exact branch/commit and commands
   are frozen, all input hashes exist, roots are fresh, and a GPU is available.

## Pre-run decision

`not_applicable_yet` -- implementation and local validation have not reached the pre-run row.

## Final claim/evidence review

`pending` -- no v2 implementation or experiment evidence exists.

## Append-only log

- 2026-08-09: user stopped/cancelled the parallel ResNet-18 work and requested a detector-native
  improvement rather than a direct LFC transplant.
- 2026-08-09: original paper equations and the active YOLO tower/TAL/CICR/router code paths were
  audited.
- 2026-08-09: draft `TAUSB-SIRC-MALC-CGR-MAP50-v2` created and self-reviewed; old v1 execution
  explicitly paused and gated against accidental recovery.
- 2026-08-09: cross-window handoff created; no method code or remote run had started.
- 2026-08-09: user explicitly approved `TAUSB-SIRC-MALC-CGR-MAP50-v2`; Spec status changed
  to `approved`, the dedicated branch was created, and CSV-driven execution was authorized.
- 2026-08-09: generated and structurally validated the dedicated 21-row v2 execution CSV;
  `CARRIER-01` was selected first. Carrier audit found the existing shared 48-coefficient,
  four-variant implementation consistent with the frozen SIRC design. Added fail-closed
  validation for non-finite/non-positive carrier parameters and deterministic-JND regression;
  the focused carrier/render/support suite passed `17 passed`. `MALC-01` is now active.
- 2026-08-09: implemented detector-native MALC primitives in `malc.py`: clean-score-weighted
  per-instance residual pooling, frozen normalized scale prototypes, equal-scale direction,
  log-magnitude and non-zero-floor losses, and explicit coverage/energy/scale diagnostics.
  No external ResNet or duplicate CICR loss was introduced. The combined MALC, legacy CICR,
  tower-hook and carrier regression suite passed `30 passed`; `CALIBRATION-01` is now active.
- 2026-08-09: implemented deterministic MALC prototype and gradient-norm calibration. Prototype
  updates accept only the calibration split and freeze per-scale direction/median/Q25-floor
  values with a reproducibility hash. Loss weights are immutable after one-time calibration;
  disconnected, zero/non-finite and excessive-clipping paths fail closed. The focused suite
  passed `18 passed`; `CGR-INTEGRATION-01` is now active.
- 2026-08-09: added the MALC-CGR integration boundary. Only per-class classification-probability
  margins become normalized CGR rows; box/CIoU margins remain diagnostics. Composite target
  gradients use SVD projection or repair-only and every class-constrained candidate is checked
  by the actual nonlinear evaluator with five backtracks before skip. The combined routing,
  selective-route and MALC suite passed `29 passed`; `MECHANISM-HARNESS-01` is now active.
- 2026-08-09: implemented the matched A0/A1 held-out mechanism harness. It enforces a single
  MALC-switch difference, aggregates level-balanced cosine/energy/coverage/scale/CGR/leakage
  metrics, retains size and person-only/co-occurrence groups, applies every frozen Success and
  Failure Signal independently, and writes a no-overwrite JSON explicitly scoped as mechanism
  evidence only. The focused suite passed `24 passed`; `PIPELINE-01` is now active.
- 2026-08-09: completed `PIPELINE-01`. The canonical entrypoints now bind the approved chain
  `config -> frozen SIRC bank -> YOLO cv3 -> clean TAL/PAG -> score-weighted MALC -> per-class
  CGR -> 48 coefficient update -> gate-passed A1 state -> deterministic materializer`. The
  mechanism runner starts A0/A1 from one route-warm state and one frozen prototype/gradient
  calibration; the arm configs may differ only at `method.enable_malc`. It never calls the EOT
  path. Formal M1 rejects a failed gate, A0 state, malformed/tampered content, mismatched
  semantic-bank/source/split hashes, partial feature switches, EOT, support leakage and budget
  overflow. All v2 switches off dispatch the exact legacy `tausb_mask` method config. Fresh
  entrypoints now fail closed on existing roots instead of deleting them. Formal victim settings
  are seed 0, 200 epochs, imgsz 640, batch 36 and SGD. The focused pipeline/config/MALC/CGR/SIRC
  suite passed `55 passed`; direct frozen-state/materializer tests passed `15 passed`. No
  mechanism or fresh-victim experiment has been run. `EVAL-01` is now active.
- 2026-08-09: completed `EVAL-01`. Train/eval dataset YAMLs now use the canonical VOC20 class
  names and order. Full-val AP50 is mapped through explicit Ultralytics `ap_class_index`; missing,
  duplicate, out-of-range, non-finite, unit-invalid or map50-inconsistent mappings fail closed.
  Metrics contain the 20-name AP50 mapping, person AP50, all 19 non-target AP50 values, the
  non-target macro, all-class mean, poisoned count, actual Linf and explicit PSNR/LPIPS gaps.
  Aggregation writes an exact C0/M1 per-class table with both delta directions and retention,
  checks C0 count 0/M1 count 6095, and evaluates the frozen single-seed success rules while
  labeling the result `tentative_single_seed`. M1 provenance requires one carrier-state hash,
  matching bank/split hashes, valid deterministic variants and only forced-pseudo fallback
  support. The focused implementation/metrics suite passed `61 passed`; no validation metric or
  fresh-victim result has been generated. `LOCAL-VALIDATION-01` is now active.
- 2026-08-09: tightened the formal M1 evidence chain before closing local validation. Every
  poisoned manifest row now carries the frozen source-manifest hash in addition to the state,
  semantic-bank and split hashes, and evaluation rejects any mismatch. M1 `metrics.json` now
  embeds the immutable A0/A1 held-out diagnostics only after validating the report schema,
  evidence scope, split hash, passing gate and `allow_fresh_victim=true`; C0 and legacy arms
  record the mechanism evidence as not applicable.
- 2026-08-09: completed `LOCAL-VALIDATION-01`. The entire local suite passed `134 passed`;
  the 12 active v2 entry files passed compile and Python-3.8 grammar parsing, and module import
  plus formal config parsing resolved person id 14, no-EOT and 200 victim epochs. Read-only VOC
  checks found exactly 16,551 train and 4,952 val images, and the surrogate checkpoint hash
  matched the frozen contract. A real-VOC CPU smoke exercised frozen YOLO/TAL/PAG, MALC
  calibration, A0/A1 coefficient gradients, CGR projection and held-out aggregation. All ten
  JSON artifacts were finite; projected-row dots were about `1e-8` and both updates were
  accepted. As expected for the one-step non-evidentiary smoke, the scientific mechanism gate
  did not pass (`allow_fresh_victim=false`) and no frozen carrier was written. This is not a UE
  result or a formal mechanism failure. Actual Python-3.8/AutoDL runtime availability remains a
  read-only `REMOTE-INPUTS-01` check. `GIT-SNAPSHOT-01` is now active.
- 2026-08-09: the Git-boundary audit recovered prior AutoDL evidence that Python 3.8 evaluated
  the ordinary type-alias assignment `BandRange = tuple[float, float]` at import time despite
  postponed annotations. This was the only top-level PEP-585 alias in `ue_framework`; it was
  changed to `typing.Tuple[float, float]`. The complete `134`-test suite passed again. Actual
  AutoDL import remains a mandatory read-only recheck, but the previously observed blocker is
  now corrected rather than hidden behind a grammar-only claim.
- 2026-08-09: created the scoped implementation snapshot
  `d85967ed070340718b8e805fb59021f560f1eb71` on
  `codex/tausb-sirc-malc-cgr-map50-v2`. The commit contains exactly 35 audited v2 Spec/CSV/review,
  source, formal-config and test files. Secret and binary-artifact scans passed. It excludes
  `.tmp`, the local smoke config, datasets, weights, credentials, old v1 task state and all
  unrelated dirty-worktree files. Push remains pending exact remote payload authorization.
- 2026-08-09: normal non-force push succeeded to
  `origin/codex/tausb-sirc-malc-cgr-map50-v2`; the remote branch contains implementation commit
  `d85967ed070340718b8e805fb59021f560f1eb71` and its workflow-metadata successor `f0995e5`.
- 2026-08-09: `REMOTE-INPUTS-01` found and repaired two pre-run environment defects without
  touching the legacy dirty worktree. A real AutoDL Python 3.8 import exposed an eager
  `tuple[...]` annotation in `shadow_tal.py`; the minimal future-annotations fix is commit
  `25c85435d8c234fc84d6cc49bcef63c21a26de03`. The mechanism config was then bound to the
  isolated input root `/root/.local/share/tausb/TAUSB-SIRC-MALC-CGR-MAP50-v2` in commit
  `c233f38ccbc7a226ffb8b99a03ae2551e2cbb588`. Both commits passed all `134` local tests and
  were normally pushed.
- 2026-08-09: AutoDL now uses clean worktree `/root/tausb-malc-wt-039c7fc` at exact HEAD
  `c233f38ccbc7a226ffb8b99a03ae2551e2cbb588`, dirty count `0`; the old clean checkout remains
  clean and `/root/autodl-tmp` remains at dirty count `2162`. A slow incomplete shallow-clone
  attempt was stopped by its exact process group and retained unused at `/root/tausb-malc-039c7fc`;
  no directory was deleted. Committed-code-only bundles were used to update the worktree after
  AutoDL-to-GitHub TLS failures.
- 2026-08-09: remote input evidence passed for Python `3.8.10`, torch `2.0.0+cu118`,
  ultralytics `8.4.33`, all active imports, no-EOT formal config, VOC train/val
  `16551/4952`, `16551` train labels, exactly `6095` person images, VOC20/person=14, and
  surrogate SHA256 `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`.
  All eight authorized background files passed per-file SHA256 and decode checks; canonical
  source hash is `3a13b0f38b06006fd7f68ae03c7206b4b047d4b6129ee7357b05b966641d47af`.
  The 64-calibration/96-held-out split resolved against VOC with hash
  `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1`.
  Semantic-bank and C2-LM remote reconstruction invoke large CPU matrix-rank/SVD work and
  exceeded the no-card audit budget; the exact hashes passed local real-VOC smoke and remain
  fail-closed in persistent formal initialization. Formal root is absent. The instance exposes
  no GPU, so no mechanism or victim command was started; this is the remaining external blocker.

## PRERUN-REVIEW-01

- Result: `blocked`（实现与输入审计通过，但正式运行环境门禁未满足）
- Decision: `do_not_run`
- Gated run: `REMOTE-MECHANISM-01`、`REMOTE-C0-01`、`REMOTE-M1-01` 全部保持禁止；
  C0/M1 还继续受 A0/A1 mechanism gate 约束。
- Code snapshot: branch `codex/tausb-sirc-malc-cgr-map50-v2`，
  `pre_run_code_commit=93f49beeeb608c4ed5d78fd762a4f8d080b4590a`；GitHub 已普通非 force
  push；AutoDL clean worktree `/root/tausb-malc-wt-039c7fc` 同一 HEAD、dirty count 0。
- Intent: 在 VOC2007+2012 / YOLOv8n 上，以 `person`（id 14）为唯一目标类，先验证
  detector-native MALC 是否让共享 SIRC carrier 在 held-out person 实例产生更一致、非零且
  多尺度有效的分类塔残差，再验证该 gate-passed carrier 是否令 fresh victim 的 person AP50
  显著下降，同时用逐类 CGR 尽量保持另外 19 类。
- Code location: 机制入口
  `ue_framework/tools/probe_tausb_sirc_malc.py`；A0/A1 主循环
  `ue_framework/methods/sirc_malc_mechanism.py`；MALC/CGR
  `ue_framework/methods/malc.py`、`malc_calibration.py`、`malc_cgr.py`；冻结状态与
  materializer `ue_framework/methods/sirc_malc_cgr.py`；正式流水线
  `ue_framework/launch_one.py`、`stages/generate.py`、`stages/train_victim.py`、
  `stages/evaluate.py`、`stages/aggregate.py`。
- Parameter data flow: mechanism YAML 冻结 source/split/surrogate/basis/bank hashes、seed、
  no-EOT、40 steps 与 CGR 阈值；运行时依次进入共享 48 系数 carrier、YOLO cv3 P3/P4/P5、
  clean TAL/PAG、score-weighted MALC、逐类 classification-probability CGR、非线性回溯和
  held-out gate。只有通过 gate 的 A1 写出冻结 carrier；正式 YAML 再核验其内容 hash、
  bank/source/split hash、`eps=16/255`、forced-pseudo support 与 no-EOT，随后 materialize
  6,095 张 person 图像。`run_tag=C0/M1` 隔离 victim artifacts，poisoned dataset 仅供 M1。
- Runtime state: AutoDL Python `3.8.10`、torch `2.0.0+cu118`、ultralytics `8.4.33`；
  当前 `torch.cuda.is_available()=False`、device count `0`。正式 run root 仍不存在，未创建
  tmux session，未启动机制、训练或评估。
- Sink effect: `evaluate.py` 通过显式 `ap_class_index` 映射落盘完整 VOC20 命名 AP50、
  person AP50、19 类宏平均、person-free/co-occurrence 指标、poison count、实际 Linf 和质量
  gap；`aggregate.py` 只在恰好一条 C0 和一条 M1 指标存在时生成逐类 delta/retention 与冻结
  success rules，结论固定为 `tentative_single_seed`。
- Baseline/disable path: mechanism A0 与 A1 除 `enable_malc` 外完全匹配；MALC-off 保留
  easy classification route、RMS 和 CGR；所有 v2 开关全关时精确回退已有 `tausb_mask`
  method config。正式 M1 拒绝 partial switches、MALC-off、CGR-off、EOT 或未通过 gate 的状态。
- Local validation: 审查中发现并修复 victim 随机性漏洞：此前 seed 未覆盖
  `YOLO(config)` 初始化。提交 `93f49be` 现在在构造模型前执行 `set_global_seed(ctx.seed)`，
  并向 Ultralytics 显式传入同一 seed；新增行为测试验证调用顺序与参数。完整本地测试为
  `135 passed in 9.54s`，v2 聚焦测试 `32 passed`，seed 聚焦测试 `5 passed`；diff check 通过。
- Minimal probe: 先前 real-VOC CPU smoke 的 10 个 JSON 均为有限值，A0/A1 前后向和 CGR
  投影/非线性检查机械通过；它只是一阶非证据 smoke，科学 gate 按预期未通过且没有冻结
  carrier。AutoDL Python 3.8 已成功导入修订后的 `run_train_victim`；远端未安装 pytest，
  因此远端行为测试由本地完整套件覆盖，保留为环境性 validation gap。
- Run command binding: 以下命令仅绑定，当前不得执行。工作目录固定为
  `/root/tausb-malc-wt-039c7fc/ue_project`，解释器固定为 `/root/miniconda3/bin/python`。

  ```bash
  # mechanism（应放入 tmux: tausb-malc-mech-s0）
  /root/miniconda3/bin/python -u ue_framework/tools/probe_tausb_sirc_malc.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_mechanism_v2.yaml \
    --device 0

  # C0（仅在 mechanism gate PASS 后；tmux: tausb-malc-c0-s0）
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 --stage train_victim \
    --gpu_id 0 --run_tag C0 \
    --poisoned_root_override /root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 --stage evaluate \
    --gpu_id 0 --run_tag C0 \
    --poisoned_root_override /root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready

  # M1（仅在 mechanism gate PASS 且 C0 完成后；tmux: tausb-malc-m1-s0）
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 \
    --stage generate_poisoned_dataset --gpu_id 0 --run_tag M1
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 --stage train_victim \
    --gpu_id 0 --run_tag M1
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 --stage evaluate \
    --gpu_id 0 --run_tag M1
  /root/miniconda3/bin/python -u ue_framework/launch_one.py \
    --config ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml \
    --method sirc_malc_cgr --steps 40 --seed 0 --stage aggregate \
    --gpu_id 0 --run_tag M1
  ```

- Experiment validity: C0/M1 都从同一配置独立构造 fresh YOLOv8n，并在模型初始化之前固定
  seed 0；epochs 200、imgsz 640、batch 36、SGD 和 clean VOC val 一致。C0 数据根不存在
  `manifest.csv/status.json`，不会被旧 poisoning metadata 污染。M1 评估必须看到 6,095 条
  poisoned rows、完整 provenance 和 passing mechanism report，否则 fail closed。
- Output non-overwrite: mechanism root 已存在即拒绝；fresh generate 拒绝既有 poisoned/M1
  artifact roots；fresh victim 拒绝既有 `train_runs/victim`。C0/M1 artifacts 的真实路径分别为
  `.../artifacts/sirc_malc_cgr/steps40/seed0_C0` 与 `seed0_M1`；M1 poisoned dataset 为
  `.../poisoned_datasets/sirc_malc_cgr/steps40/seed0`。正式命令不使用 `--force_resume`。
- Recoverability/secrecy: GPU 运行时必须使用独立 tmux 和独立日志路径，记录 hostname、
  branch、commit、session、GPU、环境、artifact root 与首进度；每 10 epoch 快照和打包。
  凭据、SSH key、数据集、权重和 smoke artifacts 均不进入提交、CSV 或日志。历史 dirty
  worktree 未被 reset/stash/clean；未删除任何目录。
- Blockers: 当前 AutoDL 实例没有 GPU。因此 `pre_run_result=blocked`，不得生成 carrier、
  不得训练 C0/M1，也不得把无卡 smoke 当作正式证据。
- Validation gaps: semantic-bank 与 C2-LM basis 的远端 CPU 重建因大矩阵 SVD 超过无卡审计
  时间预算；精确 hash 已在本地 real-VOC 初始化中通过，正式持久 mechanism 初始化仍会在任何
  优化前重新构建并 fail closed。GPU 可用后必须先重查 HEAD/dirty、CUDA、所有 hashes 和
  fresh roots，再重复本节审查；只有 `pass / allow_run` 才能调用远程运行技能。

- 2026-08-09: pre-run review detected unmatched victim initialization randomness, fixed it in
  `93f49be`, reran the full 135-test suite, pushed normally, fast-forwarded the clean AutoDL
  worktree and verified the Python-3.8 import. The review remains blocked solely at the formal
  run gate because CUDA is unavailable; no formal artifact was created.

## PRERUN-REVIEW-02

- Result: `pass`
- Decision: `allow_run`
- Gated run: 允许启动 `REMOTE-MECHANISM-01`；`REMOTE-C0-01` 和 `REMOTE-M1-01`
  仍以 mechanism report 的 `pass=true`、`allow_fresh_victim=true` 为运行时硬门禁。
- Code snapshot: `codex/tausb-sirc-malc-cgr-map50-v2`；
  `pre_run_code_commit=93f49beeeb608c4ed5d78fd762a4f8d080b4590a`；远端 clean worktree
  `/root/tausb-malc-wt-039c7fc` 同一 HEAD，`git status --porcelain` 为空。
- Intent: 与 `PRERUN-REVIEW-01` 相同；不加入 EOT、robustness transforms、外部 ResNet
  或第二套非目标保护损失。
- Code location: 与 `PRERUN-REVIEW-01` 相同；当前审查没有新增方法代码。
- Parameter data flow: CLI/config/runtime/MALC/CGR/materializer/metrics 链与上一审查一致；
  victim 的 seed 现在同时覆盖模型初始化和 Ultralytics trainer。
- Runtime state: 远端 Python `3.8.10`、torch `2.0.0+cu118`、ultralytics `8.4.33`；
  CUDA 可用，device count 1，GPU 为 `NVIDIA GeForce RTX 4090 D`，总显存 24,564 MiB、
  复审时空闲 24,081 MiB；无其他 compute process、无已有 tmux session。
- Sink effect: 与上一审查一致；机制、C0、M1、aggregate 均有独立 status/log/metrics sink。
- Baseline/disable path: A0/MALC-off、全关回退和正式 M1 partial-switch 拒绝行为均保持不变。
- Local validation: `135 passed in 9.54s`；v2 聚焦 `32 passed`；seed 聚焦 `5 passed`；
  AutoDL Python 3.8 导入通过。
- Minimal probe: 输入文件、surrogate、source manifest/map、split 均存在；VOC/labels/person
  counts 与已冻结审计一致。正式根 `/root/tausb-sirc-runs/TAUSB-SIRC-MALC-CGR-MAP50-v2`
  不存在；overlay 空闲约 25 GiB，`/root/autodl-tmp` 空闲约 14 GiB。
- Run command binding: 使用 `PRERUN-REVIEW-01` 冻结的机制/C0/M1/aggregate 内层命令；
  由 `autodl-remote-run-snippet` 生成 tmux 绑定，并置于成本保护 wrapper 中。解释器固定
  `/root/miniconda3/bin/python`，工作目录固定 `/root/tausb-malc-wt-039c7fc/ue_project`。
- Experiment validity: C0/M1 均 fresh、seed 0、200 epochs、640、batch 36、SGD、同一 clean
  VOC val；机制失败时 shell 编排和 Python 两层均禁止进入 victim。
- Output non-overwrite: 所有正式子 root 当前不存在；不使用 `--force_resume`；任何现有
  mechanism、poisoned、C0/M1 victim root 都会触发 fail closed。
- Recoverability/secrecy: 唯一正式 tmux 将记录 session、日志、commit、GPU 和 artifacts。
  根据用户成本要求，wrapper 在完整流水线成功或任一命令失败后都执行官方建议的
  `/usr/bin/shutdown`；另设 watchdog：日志连续 20 分钟无增长且 GPU 空闲时判作 hang，
  终止 runner 并关机。正常长计算但 GPU 忙时不误杀。日志和产物在关机前落盘。
- Blockers: 无。
- Validation gaps: semantic-bank/C2-LM 的远端精确重建将在正式 mechanism 初始化时执行并
  fail closed；这不是允许 victim 绕过的缺口。远端没有 pytest，但代码行为已由本地完整套件
  和远端 Python 3.8 import 覆盖。

- 2026-08-09: user enabled the GPU and added a strict cost policy. The repeat pre-run audit
  confirmed one idle RTX 4090 D, clean code commit `93f49be`, fresh outputs and sufficient disk.
  `PRERUN-REVIEW-02` therefore changed the mechanism gate to `pass / allow_run`; automatic
  shutdown and a 20-minute idle-hang watchdog are mandatory for the launched workflow.

## REMOTE-MECHANISM-01 launch and recovery state

- Launch UTC: `2026-08-09T15:06:34Z`.
- Code/session: `93f49beeeb608c4ed5d78fd762a4f8d080b4590a` in clean worktree
  `/root/tausb-malc-wt-039c7fc`; tmux `tausb-malc-v2-s0`.
- Wrapper: `/root/autodl-tmp/tausb-malc-v2-formal-cost-guard.sh`, SHA256
  `3a0ac8845b4463ae8db8c5406993e46013cf47ed0d4400531eaf80c1472d0961`.
- Persistent diagnostics: `/root/autodl-tmp/tausb-malc-v2-control/formal-seed0.log` and
  `/root/autodl-tmp/tausb-malc-v2-control/cost-guard-status.json`.
- First health evidence: tmux pane PID `1414`, pane alive, and the first durable marker was
  `stage=mechanism detail=matched_A0_A1_gate`.
- Observed failure boundary: the next SSH health check was refused and a second attempt timed
  out. The cost wrapper invokes `/usr/bin/shutdown` on every command failure, so this is strong
  evidence that the instance shut down during early mechanism initialization. C0/M1 were not
  started; no mechanism PASS or UE result is claimed.
- Cost outcome: the requested automatic shutdown operated immediately; the GPU is not being
  left online for debugging.
- Debug decision: `systematic-debugging / evidence_pending`. Do not infer a cause or patch code
  until the persisted full traceback, mechanism status and cost-guard state are read. The
  cheapest faithful next step is AutoDL no-card mode, followed by read-only retrieval of the two
  persistent diagnostics and any partial mechanism status. GPU must remain off during diagnosis.

- 2026-08-09: the formal mechanism workflow launched after a passing pre-run review, emitted
  its first mechanism marker, then triggered the cost guard and shut the instance down before
  C0/M1. Root cause remains unknown until persistent logs are retrieved in no-card mode.

## Systematic debugging: portable provenance hash correction

- Persisted cost-guard evidence recorded `state=failed`, `stage=mechanism`, and
  `formal_pipeline_exit_1_shutdown_requested`; the traceback failed in
  `SIRCProbeWorkflow.__init__` with `Frozen semantic carrier bank hash mismatch`. No mechanism
  artifact root was created and C0/M1 never started.
- All eight authorized source files matched the frozen per-file SHA256 values on both systems.
  Local OpenCV 5.0 and AutoDL OpenCV 4.9 also produced identical RGB uint8 shapes, sums and
  decoded-tensor SHA256 values for every source. Source bytes, ordering and image decoding are
  therefore ruled out.
- The failed gate hashed exact float32 bytes produced by large FFT/SVD pipelines. Local
  Windows/Torch 2.11 and AutoDL Linux/Torch 2.0 are not required to produce bit-identical last
  bits, so that checksum was an environment fingerprint rather than a portable protocol hash.
- Corrective commit `a7df8684b0c8ccb767df9712c7bc88a4fde29321` adds opt-in
  `recipe-v1` hashes over the already verified source manifest, ordered source IDs and SHA256
  values, and frozen carrier parameters. Historical configs continue to default to
  `tensor-v1`. Actual tensors still pass finite/rank/zero-mean/unit-norm checks, and the frozen
  A1 carrier state retains an exact content hash.
- Re-derived formal hashes are semantic bank
  `0b8a94efc55155bea20a1ec799bfac14c8a6f11fd6530538f3e0437b37c0dd4b` and C2-LM
  `8350c0a608150839c98a8dad8db862d0c9dfaeca4714f05d1714afac0f30cfa5`.
- Local validation: four modified modules compile; Python 3.8 AST parsing passes; all formal
  YAML files parse; focused carrier tests pass `18/18`; pipeline/MALC/CGR tests pass `43/43`;
  complete suite passes `139/139`; formal hashes are independently re-derived from the frozen
  manifest and match both mechanism and M1 configs.
- The fix was normally pushed to
  `origin/codex/tausb-sirc-malc-cgr-map50-v2`. The new local cost wrapper is bound to the full
  corrective commit and has SHA256
  `430e80e02288bc8adc7b31e15c2ef5913784f57b8d711335803a0a32d68c4415`.
- AutoDL was shut down after the no-card diagnostic exceeded the user's cost budget. The
  remote clean worktree is still known to be at `93f49be`; no new remote command was launched.

## PRERUN-REVIEW-03

- Result: `blocked`
- Decision: `do_not_run`
- Gated run: `REMOTE-MECHANISM-01`; C0/M1 remain additionally gated by mechanism PASS.
- Code snapshot: local/GitHub branch `codex/tausb-sirc-malc-cgr-map50-v2`, commit
  `a7df8684b0c8ccb767df9712c7bc88a4fde29321`; remote clean checkout deployment pending.
- Intent: unchanged approved SIRC-MALC-CGR VOC20/person experiment. This correction changes
  provenance checking only; carrier tensors, MALC, CGR, epsilon, no-EOT, victim protocol and
  metrics are unchanged.
- Code location: `provenance_hash.py`, `semantic_residual_carrier.py`,
  `background_spectral_basis.py`, `sirc_probe.py`, the two formal v2 configs, and tests.
- Parameter data flow: verified source bytes and canonical manifest -> ordered provenance plus
  frozen resolution/bands/seeds -> `recipe-v1` hash -> mechanism frozen A1 state -> formal M1
  config/state equality gate. The exact frozen-state content hash remains downstream.
- Runtime state: AutoDL is off. The last known clean worktree is commit `93f49be`, dirty 0;
  formal run root was absent and the failure happened before mechanism artifact creation.
- Sink effect: input audit now records semantic/C2-LM hash values and modes; existing
  mechanism, C0/M1 status, named VOC20 AP50 and aggregate sinks are unchanged.
- Baseline/disable path: configs without explicit hash modes still execute the original
  `tensor-v1`; v2 feature-off and historical TAUSB paths remain covered by the 139-test suite.
- Local validation: compile, YAML parse, Python 3.8 AST, independent formal recipe check and
  `139 passed`.
- Minimal probe: exact source/decode parity across local and AutoDL plus pure-standard-library
  formal recipe recomputation. Full 640x640 numerical reconstruction is intentionally not
  repeated on the half-CPU no-card mode because the hash no longer depends on float bytes.
- Run command binding: local wrapper binds commit
  `a7df8684b0c8ccb767df9712c7bc88a4fde29321`, method `sirc_malc_cgr`, steps 40, seed 0,
  device 0, formal/mechanism v2 configs and the unique approved run root. It retains automatic
  shutdown on completion/failure and the 20-minute idle-hang watchdog.
- Experiment validity: unchanged VOC train/val, target id 14, fresh C0/M1 YOLOv8n victims,
  200 epochs, imgsz 640, batch 36, SGD and clean validation; no robustness transforms.
- Output non-overwrite: wrapper requires the formal run root to be absent and does not use
  force-resume. This must be rechecked remotely before launch.
- Recoverability/secrecy: persistent external control log/status paths and tmux are retained;
  no credentials, data, weights or local paths enter the commit or CSV.
- Blockers: start AutoDL in no-card mode; update the clean worktree to exact `a7df868`; verify
  dirty count 0, Python 3.8 import, formal recipe hashes and absent formal root; upload and hash
  the new wrapper; then repeat this review. Do not start GPU work before `pass / allow_run`.
- Validation gaps: no remote Python 3.8 runtime import or deployed-commit verification for
  `a7df868` yet. No fresh-victim or mechanism effectiveness claim exists.
