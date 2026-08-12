# TAUSB-SDH-E2E-V0-ONEBOOT-v2 Review Handoff

## R4 current status (authoritative)

- R4 terminal class: `cost_gate_stop`; this is an approved operational stop,
  not a code failure and not a scientific failure.
- Execution commit: `83cfb21c11195e1b1e034db3422716a34b18e166`.
- `PRECHECK`, P1/binding reuse, `SMOKE_C0`, and `SMOKE_M1` completed. Both
  smoke arms exited 0 and completed generate, fresh-victim train, and clean
  evaluation.
- The paired-smoke data-flow review passed. E20 did not start because the
  paired estimate was 59.29 GPU hours versus the approved 8-hour cap, and
  projected disk need was 29.79 GB versus 13.16 GB free.
- The controller automatically shut the instance down. Nineteen minimal files
  were pulled and SHA-256 verified with no missing required file and no
  dataset, image, weight, checkpoint, or credential.
- Scientific status: `inconclusive / not evaluated`; one-epoch all-zero AP50
  is not interpretable, so `STATE.md` Current Best remains unchanged.

## Current workflow state

- Spec：`docs/research/specs/TAUSB-SDH-E2E-V0-ONEBOOT-v2.md`
- Status：approved execution closed；R4 按成本门禁终止，所有 CSV 行已收口。
- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`
- Method implementation base：`3a7a1aaff912d0904794a91a4d3512d18b5c69fa`
- Execution commit：`83cfb21c11195e1b1e034db3422716a34b18e166`（R4 provenance 修复后 review pass，已 push）。
- Active CSV row：无；`REVIEW-01` 已完成。
- Remote/GPU：R4 已自动关机；E20 未启动，后续实验必须由新批准 Spec 定义。

## Objective scientific result

本修订不改变父 Spec 的方法和科学判据。目标是在一次 GPU 开机中，以 fail-closed
状态机串行运行 mechanism、P1 binding、paired smoke、CPU-only 数据流/费用审查、
条件 E20 C0/M1 和 VOC20 comparison；任何错误或最终完成均自动关机。

## Implemented scope

- `ue_project/ue_framework/tools/run_tausb_sdh_e2e_v0_oneboot.py`
  - fresh-root、commit、input-audit、GPU 与磁盘 precheck；
  - 真实 P1/binder 复用；
  - 带首进度、GPU 进程、无进度和 wall timeout 的串行命令监控；
  - smoke C0/M1 数据流核验；
  - 8 GPU-hours 与 1.5 倍磁盘安全余量门禁；
  - E20 C0 完成后才允许 M1；最终显式 comparison。
- `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_controller.sh`
  - 29 行幂等 EXIT/INT/TERM shutdown wrapper；
  - execution commit 与 clean checkout 由环境显式绑定。
- `ue_project/tests/test_sdh_e2e_v0_oneboot.py`
  - success/failure shutdown、wall timeout、smoke pass、time/disk stop、metrics/config
    drift、E20 epoch/count 与 fresh CLI 路径。

没有修改 r2 carrier、D-LFC、CICR、CGR、NLA、binder、materializer、victim trainer、
evaluation、comparison、配置语义或 Success/Failure 阈值。

## Local evidence

- `53 passed`：
  `test_sdh_e2e_v0_config.py`、`test_sdh_e2e_v0_oneboot.py`、
  `test_sdh_evaluation.py`、`test_sdh_materializer.py`、
  `test_sdh_pipeline_config.py`、`test_sdh_experiment_hosts.py`。
- 新 one-boot 定向测试：`8 passed`。
- Python 3.8 AST：2 files pass。
- `py_compile`：pass。
- controller CLI `--help`：pass。
- Git Bash `bash -n`：pass。
- CSV：11 rows、28 fields、dependencies、pre-run-before-remote 与 final review 均 pass。
- task-scope `git diff --check`：pass。
- Pytest 只有工作区 `.pytest_cache` 无写权限 warning，不影响测试结果。

## Active risks and blockers

1. 旧 `mechanism_run_contract.json` 成功后会立即关机，只能保留作历史证据，不能执行。
2. smoke→E20 时间公式是保守外推；若 paired 预测超过 8 GPU-hours，会形成
   `cost_gate_stop` 并关机，不记为科学 Failure。
3. 本地测试不能证明远程 CUDA、真实 P1、victim 训练或 AP50 有效；这些仍是 remote evidence gap。

## PRERUN-REVIEW-01

- Result：`blocked`
- Decision：`do_not_run`
- Gated run：`REMOTE-ONEBOOT-01`
- Code snapshot：`ad3a3e5ba59934dae11f32bf20143f1936aa2dd5`
- Intent：单次开机、逐阶段 hard gate、任何失败或最终完成自动关机；匹配 Spec。
- Code location：controller、29 行 wrapper 与 tests 都在 active commit；方法代码未修改。
- Parameter data flow：controller → binder → 四份 config → `launch_one --stage all` →
  `SDHMaterializer` → fresh victim → clean evaluate → explicit comparison。
- Runtime state：本地 53 tests、CLI、AST、shell 与 CSV 通过；GPU 未启动。
- Sink effect：metrics 与 comparison sink 可达；但 smoke C0 单臂 sink 校验发生在 M1 后。
- Baseline/disable path：formal SDH 配置、E2E loader 与现有 tests 未回归；没有 TAUSB/SIRC fallback。
- Run command binding：branch 已普通 push 到 `ad3a3e5`；clean checkout 与 payload 尚未冻结。
- Experiment validity：VOC20、person id14、seed0、steps40、clean val、no EOT/JND 已绑定。
- Output non-overwrite：fresh roots 与无 `force_resume`/override 已验证。
- Recoverability/secrecy：tmux/control/log/shutdown 路径已设计；没有凭据落盘。
- Blocker：smoke C0 若命令以 0 返回但 metrics/checkpoint/count 不完整，当前代码仍会启动
  smoke M1，然后才在 paired review 失败。这违反逐臂 fail-closed，并可能浪费 M1 GPU 成本。
- Minimum fix：在 smoke 循环内、每臂命令返回后立即调用 `validate_arm`，通过后才进入
  下一臂；保留两臂后的 paired identity/cost review。
- Validation gaps：真实 CUDA/P1/smoke/E20 仍未运行。

## PRERUN-REVIEW-02

- Result：`pass`
- Decision：`allow_run`
- Gated run：`REMOTE-ONEBOOT-01 / ONEBOOT-S0-R1`
- Code snapshot：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3@52d57fd005a318c912f5e43a5cf91dfe1357cddf`；远端同名 branch 已解析到相同 SHA。
- Intent：只改变 orchestration；mechanism→P1/bind→smoke C0/M1→CPU-only review→条件 E20 C0/M1→comparison。
- Code location：active controller module、29 行 shutdown wrapper；与 method base 比较时 methods/stages/config 无 diff。
- Parameter data flow：
  `oneboot_controller.sh` → `run_tausb_sdh_e2e_v0_oneboot.py` →
  `run_tausb_sdh mechanism` → `bind_tausb_sdh_e2e_v0` → 四份 bound config →
  `launch_one(all)` → `SDHMaterializer` → fresh YOLO victim → clean VOC evaluate →
  `compare_tausb_sdh_e2e_v0`。
- Runtime state：54 focused SDH tests pass；单臂 validation 位于 smoke loop 内且早于
  `state.complete`/下一 iteration；GPU process、首进度、idle 与 wall cap 均记录。
- Sink effect：每臂必须有 completed generate/train/evaluate、best checkpoint、期望 epoch/count、
  20 类 finite AP50 和 P1 provenance；paired review 后才生成 E20 caps。
- Baseline/disable path：formal 200-epoch SDH gate 与既有 E2E config/materializer/evaluation tests
  未回归；无 `tausb_mask`、SIRC、carrier-only、pseudo mask 或 robustness fallback。
- Local validation：54 tests、Python 3.8 AST、py_compile、CLI help、bash syntax、CSV dependencies、
  task-scope diff check 均 pass；pytest cache warning 为本地缓存写权限，不影响结果。
- Minimal probe：synthetic C0/M1 metrics/status/config 贯通 pass、time stop、disk stop、missing
  class、config drift、wrong epoch/count；wrapper success/failure 均调用 fake shutdown。
- Run command binding：
  - checkout：`/root/tausb-sdh-checkouts/e2e-v0-52d57fd-worktree`
  - tmux：`tausb-sdh-e2e-v0-oneboot-s0-r1`
  - wrapper SHA-256：`4a18558f97bd4e5c6ab71b006069fbdd6ac8be922a0aa9486e467e714a14e345`
  - controller SHA-256：`3925d377c566c221bd752765a02fbfd67e32b338f0e3cd9e7b2b0c9613f0de6f`
  - launch payload SHA-256：`6bebac2825a38fa0d42b4f177b281bafb62d71931c96840901fd1a081ebb47ac`
  - contract：`research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_run_contract.json`
- Experiment validity：VOC20、person id14、16,551/6,095、seed0、steps40、eps16/255、
  bbox support、no EOT/JND、full clean val、C0/M1 fresh victims 与原 Success/Failure 不变。
- Output non-overwrite：9 个 exact roots precheck fresh；run roots分离；无 force resume、override、
  delete 或历史覆盖。
- Recoverability/secrecy：唯一 tmux、controller status、阶段 logs、health checks 和 recovery
  路径已冻结；credential scan pass；无主机、密码、私钥或 token 落盘。
- Blockers：none。
- Validation gaps：真实 CUDA/P1/smoke/E20/AP50 尚未运行；remote 首进度通过前不能声称启动成功。

## R1 prelaunch failure and R2 correction

- R1 terminal state: `failed_prelaunch`; the launcher created the detached
  `52d57fd` checkout, but the controller failed in `PRECHECK` before any
  mechanism, training, run root, or scientific metric was created.
- Cost guard: the controller shutdown trap ran; a follow-up SSH probe returned
  connection refused. No retry was attempted in GPU mode.
- Pulled evidence: `remote_artifacts/controller_status.json`, 441 bytes,
  SHA-256 `a295ce5564205150a4efc97105241838ebf53199d5591c1551f34098d51afe7f`;
  the transfer report has `missing_required=[]` and contains no dataset,
  checkpoint, poisoned image, model weight, or credential.
- First bad boundary: the frozen value `46f757...` is the raw YAML file SHA-256,
  while the R1 precheck compared it with canonical JSON SHA-256 `b75bb7...`.
  The YAML contents and PyYAML 6.0 parser were valid; two different hash
  conventions were mixed.
- Minimum correction: precheck now compares the frozen value with the raw file
  SHA-256. Runtime P1/binder provenance remains canonical JSON SHA-256 and is
  reported separately. No method, stage, config, data, metric, seed, epoch, or
  scientific gate changed.
- Non-overwrite correction: every retry output uses fresh R2 names. R1 control
  and checkout evidence remain untouched.
- Local evidence: 11/11 focused one-boot tests and 82/82 expanded SDH tests
  passed; py_compile, CLI help, Bash syntax, CSV integrity, and diff check
  passed. Relative to `52d57fd`, `methods/`, `stages/`, and `configs/` have no
  diff.
- Claim boundary: R1 is an operational prelaunch failure, not a mechanism or
  scientific failure. CUDA/P1/smoke/E20/AP50 evidence is still absent.

## PRERUN-REVIEW-03

- Result: `pass`
- Decision: `allow_run`
- Gated run: `REMOTE-ONEBOOT-02 / ONEBOOT-S0-R2`
- Code snapshot: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3@7535a3c9d9167648eaafaa1afbb8c895673404d0`; the pushed branch and AutoDL Git object resolve to the same SHA.
- Intent: correct only the R1 raw/canonical config-hash mismatch and allocate fresh retry paths; preserve the approved one-boot state machine and every scientific setting.
- Code location: the active controller precheck uses `_file_sha256` for the frozen YAML file; mechanism generation and the binder still use canonical JSON SHA-256 for runtime provenance.
- Parameter data flow: launch gate → shutdown wrapper → controller PRECHECK → mechanism T0/T1/P0/P1 → binder → four bound configs → `launch_one(all)` → fresh victims → clean VOC evaluation → explicit comparison.
- Runtime state: local 11/11 focused and 82/82 expanded SDH tests pass; remote Python 3.8.10 raw/canonical hash assertions, import, `py_compile`, and nine-root uniqueness assertions pass. Remote pytest is unavailable and was not installed.
- Sink effect: no method/loss/stage/config code differs from `52d57fd`; the correction affects only the precheck comparison and provenance field naming.
- Baseline/disable path: C0 remains an independent clean-data fresh victim; M1 remains the P1-materialized arm. No TAUSB/SIRC fallback, EOT, robustness transform, 200-epoch run, resume, or override was introduced.
- Local validation: py_compile, CLI help, Bash syntax, CSV integrity, credential scan, payload round-trip, and diff check pass.
- Minimal probe: AutoDL Python 3.8 reports raw SHA `46f757...`, canonical SHA `b75bb7...`, nine unique R2 controller roots, and no formal R2 root creation.
- Run command binding:
  - execution checkout: `/root/tausb-sdh-checkouts/e2e-v0-7535a3c-r2-worktree`
  - tmux: `tausb-sdh-e2e-v0-oneboot-s0-r2`
  - wrapper SHA-256: `4a18558f97bd4e5c6ab71b006069fbdd6ac8be922a0aa9486e467e714a14e345`
  - controller SHA-256: `635037bc1f0443641e4c80fdbcd0b840282308d1117be3e4d34967b99284c0f2`
  - launch payload SHA-256: `99f86ee6451611d4f8cb08498141b6a0a093342202bccddadc9d3325f8046890`
  - contract: `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_r2_run_contract.json`
- Experiment validity: VOC20, person id 14, seed 0, steps 40, eps 16/255, bbox support, no EOT/JND, full clean validation, fresh C0/M1 victims, and the approved success/failure thresholds are unchanged.
- Output non-overwrite: the R2 session, checkout, control, log, mechanism, binding, smoke, E20, and comparison paths were all remotely confirmed absent. All R1 evidence remains untouched.
- Recoverability/secrecy: one tmux, atomic controller status, per-stage logs, failure/final shutdown, and recovery paths are frozen; credential scan has zero hits.
- Blockers: none.
- Validation gaps: CUDA/P1/smoke/E20/AP50 remain unexecuted. A successful GPU launch may only be claimed after first progress, finite quantities, GPU-process observation, and controller status creation.

## Final claim/evidence review

- Status：pending。
- 当前只可声称 orchestration 实现和本地机械验证通过；不能声称 mechanism、smoke、E20
  或目标类不可学习效果已运行或有效。

## Append-only execution log

## R2 evidence correction and R3 continuation

- R2 controller reported `MECHANISM failed` only because it read a `-MECH-R2`
  path while the frozen YAML wrote to the approved root ending in `-MECH`.
- The actual mechanism process completed in about 40.2 seconds. Its real P1 is
  finite, has Linf `0.0627451017`, and all required state files exist. The
  scientific mechanism gate is false and remains diagnostic under the approved
  feasibility-only claim boundary.
- CPU-only binding succeeded against the actual mechanism root. It froze a
  deterministic 200-image smoke selection (40 person, 160 person-free) and
  generated four load-valid configs with distinct fresh roots.
- Four reviewed pull manifests transferred 13 files totaling about 3.16 MB;
  every `missing_required` list is empty. No dataset, poisoned image tree,
  victim weight, credential, or unrelated checkpoint was transferred.
- Execution commit `34e28f1622f2b3f053de70e1cb0d013f62d42f15` adds only
  a `--resume-from-binding` controller path and auto-shutdown R3 wrapper.
  Method losses, configs, materializer, victim trainer, evaluation, metrics,
  seed, epochs, and scientific thresholds are unchanged.
- Validation: 12/12 focused one-boot tests, 83/83 expanded SDH tests,
  py_compile, CLI help, Bash syntax, and diff check pass.

## PRERUN-REVIEW-04

- Result: `pass`
- Decision: `allow_run`
- Gated run: `REMOTE-ONEBOOT-03 / ONEBOOT-S0-R3`
- Code snapshot: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3@34e28f1622f2b3f053de70e1cb0d013f62d42f15`
- Intent: reuse the hash-verified real P1/binding, start paired smoke immediately,
  and continue to E20 only after the existing dataflow/cost/disk gate passes.
- Code location: active one-boot controller, R3 resume wrapper, and focused
  tests; no method or scientific config changes.
- Parameter data flow: frozen P1/binding report -> four bound configs ->
  `launch_one(all)` C0 then M1 -> per-arm status/20-class AP50 validation ->
  paired smoke review -> conditional E20 -> explicit comparison.
- Runtime state: exact clean AutoDL worktree exists at commit `34e28f1`;
  Python 3.8 compile/CLI and wrapper Bash syntax pass; binding and frozen state
  hashes match the locally verified evidence.
- Sink effect: each arm must complete generate/train/evaluate, emit a best
  checkpoint and finite VOC20 AP50, and match exact poison counts before the
  next arm or E20 can start.
- Baseline/disable path: C0 remains an independent clean victim. R3 does not
  rerun mechanism and introduces no TAUSB/SIRC fallback.
- Local validation: `83 passed`; no test-tool installation or GPU job was used.
- Minimal probe: real P1 binding load-valid; all R3 control/log/comparison and
  R2 smoke/E20 output roots are absent.
- Run command binding:
  - tmux: `tausb-sdh-e2e-v0-oneboot-s0-r3`
  - wrapper SHA-256: `92a7b53fc47e0f1f68dcb71ef016febe6f981ff9e0688535aab72c2d3c85889d`
  - controller SHA-256: `6d5461c124d3323509ad9c7d4256f4d900eb789940190f9ec60e3478a7b8e925`
  - launch gate SHA-256: `01656d879efeaaca27b13209f4a6f633bd01f9c0a7981be0be98c69b199e26b8`
  - contract: `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_r3_run_contract.json`
- Experiment validity: VOC20, person id 14, seed 0, steps 40, eps 16/255,
  bbox support, no EOT/robustness transform, clean validation, independent fresh
  victims, and the approved feasibility-only claim boundary are unchanged.
- Output non-overwrite: R1/R2 evidence and successful mechanism/binding are
  read-only; every new GPU/control/log/comparison root is fresh and the launch
  gate rejects collisions.
- Recoverability/secrecy: one tmux, atomic status, per-stage logs, 20-minute idle
  guard, 8 GPU-hour paired cap, and failure/final auto-shutdown remain active;
  no credential appears in the contract or commands.
- Blockers: none.
- Validation gaps: smoke and E20 AP50 have not yet run. The next GPU launch is
  the evidence-producing step; smoke is mechanical, while E20 remains tentative
  single-seed directional evidence.

- 2026-08-11：用户明确批准 `TAUSB-SDH-E2E-V0-ONEBOOT-v2`。
- 2026-08-12：Spec 状态写为 approved；生成并校验 11 行执行 CSV。
- 2026-08-12：完成 one-boot orchestrator、shutdown wrapper、P1/binding 与 smoke/cost gates。
- 2026-08-12：53 项 SDH focused regression、Python 3.8 AST、CLI、shell、CSV 与 diff 检查通过；未启动 GPU。
- 2026-08-12：创建并普通 push execution commit `ad3a3e5`。
- 2026-08-12：`PRERUN-REVIEW-01` blocked：smoke 单臂完整性校验晚于下一臂启动；插入最小修复与第二次 pre-run，GPU 未启动。
- 2026-08-12：修复 commit `52d57fd` 完成并 push；54 项 focused regression 通过。
- 2026-08-12：`PRERUN-REVIEW-02` pass / allow_run；冻结 one-boot payload `6bebac…47ac`，等待 GPU 开启。
## R3 operational failure and minimal R4 correction

- R3 completed the 200-image C0 smoke and the M1 generation, one-epoch victim
  training, and full clean evaluation without CUDA OOM, NaN, or Inf. It stopped
  at the M1 evaluation provenance sink before E20.
- The first bad boundary is proven: `sdh_materializer.py` emitted
  `secret_source_sha256`, but `generate.py` omitted that value from both the
  manifest schema and per-row provenance initializer. The evaluator therefore
  correctly rejected the incomplete manifest.
- Commit `83cfb21c11195e1b1e034db3422716a34b18e166` adds that single provenance
  field and R4 path overrides. It does not change losses, P1, dataset, target,
  seed, steps, epochs, metric sinks, or scientific gates.
- Evidence: the real 200-row R3 manifest replay passes after supplying the
  frozen expected hash; 55 focused and 85 full SDH tests pass; py_compile,
  Bash syntax, diff check, and credential scan pass.
- Claim boundary: R3 remains an operational failure with no E20 scientific
  result. The successful generation/training/evaluation is dataflow evidence,
  not evidence that the method is effective.

## PRERUN-REVIEW-05

- Result: `pass`
- Decision: `allow_run_when_gpu_is_enabled`
- Gated run: `REMOTE-ONEBOOT-04 / ONEBOOT-S0-R4`
- Code snapshot: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3@83cfb21c11195e1b1e034db3422716a34b18e166`.
- Frozen evidence metadata: commit `f87efebe196b422c07f0375c2d030c66859b856a`.
- Intent: reuse the exact verified P1, repeat paired smoke in fresh R4 roots,
  and proceed to E20 only if the unchanged automatic dataflow/cost/disk review
  returns `continue_e20`.
- Parameter data flow: frozen P1 -> CPU-only R4 binder -> four load-valid R4
  configs -> `launch_one(all)` C0 then M1 -> 20-class clean VOC AP50 validation
  -> paired smoke review -> conditional E20 -> explicit comparison.
- Root-cause closure: the manifest writer now records
  `secret_source_sha256`; the materializer and evaluator contracts already
  required the same value. A schema regression test prevents recurrence.
- Baseline/disable path: C0 is still a separately trained clean-data victim;
  M1 is still the P1-materialized arm. No mechanism rerun, resume, override,
  EOT, robustness transform, new seed, or 200-epoch run is permitted.
- Local validation: `55 passed` focused, `85 passed` complete SDH regression,
  py_compile, controller CLI, Bash syntax, diff check, and credential scan.
- Remote no-card validation: exact clean execution checkout exists; controller
  `_verify_binding()` loads all four configs; R2/R4 semantic comparison is
  paths-only; selection content is identical (200 images: 40 target and 160
  person-free); seven R4 execution roots and the tmux session are absent.
- Binding evidence: six files pulled and individually SHA-256 verified with
  `missing_required=[]`; no dataset, poisoned image, checkpoint, weight, or
  credential was transferred.
- Run binding:
  - checkout: `/root/tausb-sdh-checkouts/e2e-v0-83cfb21-r4-worktree`
  - tmux: `tausb-sdh-e2e-v0-oneboot-s0-r4`
  - wrapper SHA-256: `778093577f45ddd31bc705f22d6b8d4bc14a944b44e96ade90b8110da417b7e1`
  - controller SHA-256: `91bb9d1db1c86bd1a4498ad5a018224972c407b0db1677f95a41ba85feefce61`
  - manifest writer SHA-256: `5ba9b8af6cd3bb04057b89e330773a1d054b0fc8138e22d87179bdc9ee0898c1`
  - binding report SHA-256: `f06efd60bf2adad91c0cfb148d8713747d9d3250e5d828f930d5bc2bc472c6b5`
  - frozen state SHA-256: `c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168`
  - launch gate SHA-256: `94c9446da49dee68b629cdeb2259698a95fefeee22eddbc1fa71b0ea768a83f1`
  - contract: `research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_r4_run_contract.json`
- Cost/recovery: 7,200-second paired-smoke cap, 1,200-second idle guard,
  28,800-second paired-E20 cap, 1.5x disk projection gate, atomic status,
  per-stage logs, failure/final auto-shutdown. No-card audit observed
  13,420,335,104 free bytes; GPU precheck will re-evaluate this value.
- Blockers: GPU is not enabled, so the launch gate was intentionally not run.
- Validation gap: R4 smoke and E20 AP50 have not run. A scientific claim is
  allowed only after the resulting status, metrics, logs, and comparison are
  pulled and verified.

## R4 terminal execution and final review

- Final review result: `vision_met_for_approved_cost_guarded_workflow`.
- Scientific hypothesis: `inconclusive / not evaluated` because paired E20
  was never started. This result is neither Success nor scientific Failure.
- R4 controller completed `PRECHECK`, P1/binding reuse, C0 smoke, and M1 smoke.
  C0 and M1 exited 0 after generate, one-epoch fresh-victim train, and clean
  evaluation. Finite training losses and an active RTX 4090 D process were
  observed for both arms.
- M1 materialized 40 poisoned images in the frozen 200-image smoke selection.
  Actual Linf max was `0.0627445`; all 40 poisoned manifest rows recorded the
  same frozen secret hash. This closes the R3 provenance failure.
- The data-flow gate passed. The cost gate stopped E20 because the paired
  estimate was `213440.76` seconds (`59.29` hours) against the approved
  `28800`-second cap. Projected disk need was `29.79` GB with only `13.16` GB
  free, also below the frozen 1.5x safety margin.
- Both one-epoch smoke arms produced all-zero 20-class AP50, which is expected
  to be non-discriminative at this training budget. These values are excluded
  from scientific comparison rather than reported as a method result.
- Artifact closure: 19 selected files, 365148 bytes, three transfer reports,
  `missing_required=[]`, no failed transfer, individual SHA-256 verification,
  and no data, image, weight, checkpoint, or credential transfer.
- Automatic shutdown and the user's cost constraint were honored. No E20 or
  comparison root was created, so no additional GPU spending occurred.
- `STATE.md` was not modified. The evidence-backed candidate is to leave
  Current Best unchanged.
- Remaining validation gap: target-class unlearnability and 19-class
  preservation remain untested at an interpretable victim-training budget.
  The next experiment requires a new approved Spec for a cost-bounded paired
  pilot whose clean C0 AP50 must first leave the all-zero region.
