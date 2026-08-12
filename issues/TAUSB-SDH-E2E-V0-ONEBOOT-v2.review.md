# TAUSB-SDH-E2E-V0-ONEBOOT-v2 Review Handoff

## Current workflow state

- Spec：`docs/research/specs/TAUSB-SDH-E2E-V0-ONEBOOT-v2.md`
- Status：approved；第一次 pre-run blocked 的单臂顺序缺口已修复，第二次 pre-run `pass / allow_run`。
- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`
- Method implementation base：`3a7a1aaff912d0904794a91a4d3512d18b5c69fa`
- Execution commit：`52d57fd005a318c912f5e43a5cf91dfe1357cddf`（第二次 review，pass，已 push）。
- Active CSV row：`REMOTE-ONEBOOT-01`，等待用户开启 GPU 后启动唯一 reviewed payload。
- Remote/GPU：未启动；AutoDL 当前不应执行旧的 mechanism-only contract。

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

## Final claim/evidence review

- Status：pending。
- 当前只可声称 orchestration 实现和本地机械验证通过；不能声称 mechanism、smoke、E20
  或目标类不可学习效果已运行或有效。

## Append-only execution log

- 2026-08-11：用户明确批准 `TAUSB-SDH-E2E-V0-ONEBOOT-v2`。
- 2026-08-12：Spec 状态写为 approved；生成并校验 11 行执行 CSV。
- 2026-08-12：完成 one-boot orchestrator、shutdown wrapper、P1/binding 与 smoke/cost gates。
- 2026-08-12：53 项 SDH focused regression、Python 3.8 AST、CLI、shell、CSV 与 diff 检查通过；未启动 GPU。
- 2026-08-12：创建并普通 push execution commit `ad3a3e5`。
- 2026-08-12：`PRERUN-REVIEW-01` blocked：smoke 单臂完整性校验晚于下一臂启动；插入最小修复与第二次 pre-run，GPU 未启动。
- 2026-08-12：修复 commit `52d57fd` 完成并 push；54 项 focused regression 通过。
- 2026-08-12：`PRERUN-REVIEW-02` pass / allow_run；冻结 one-boot payload `6bebac…47ac`，等待 GPU 开启。
