# TAUSB-SDH-E2E-V0-ONEBOOT-v2 Review Handoff

## Current workflow state

- Spec：`docs/research/specs/TAUSB-SDH-E2E-V0-ONEBOOT-v2.md`
- Status：approved；implementation 与本地轻量验证完成，pre-run implementation review 尚未执行。
- Branch：`codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`
- Method implementation base：`3a7a1aaff912d0904794a91a4d3512d18b5c69fa`
- Execution commit：pending。
- Active CSV row：`LOCAL-VALIDATION-01`，完成后进入 `PRERUN-REVIEW-01`。
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

1. Execution commit、detached remote checkout、controller payload hash 和 tmux command 尚未冻结；因此 GPU hard gate 仍关闭。
2. 旧 `mechanism_run_contract.json` 成功后会立即关机，只能保留作历史证据，不能执行。
3. smoke→E20 时间公式是保守外推；若 paired 预测超过 8 GPU-hours，会形成
   `cost_gate_stop` 并关机，不记为科学 Failure。
4. 本地测试不能证明远程 CUDA、真实 P1、victim 训练或 AP50 有效；这些仍是 remote evidence gap。

## Pre-run decision

- Result：`pending`
- Required next evidence：scoped execution commit、independent review packet、exact clean checkout、
  wrapper/payload SHA-256、fresh roots、tmux launch command、success/failure shutdown paths。

## Final claim/evidence review

- Status：pending。
- 当前只可声称 orchestration 实现和本地机械验证通过；不能声称 mechanism、smoke、E20
  或目标类不可学习效果已运行或有效。

## Append-only execution log

- 2026-08-11：用户明确批准 `TAUSB-SDH-E2E-V0-ONEBOOT-v2`。
- 2026-08-12：Spec 状态写为 approved；生成并校验 11 行执行 CSV。
- 2026-08-12：完成 one-boot orchestrator、shutdown wrapper、P1/binding 与 smoke/cost gates。
- 2026-08-12：53 项 SDH focused regression、Python 3.8 AST、CLI、shell、CSV 与 diff 检查通过；未启动 GPU。
