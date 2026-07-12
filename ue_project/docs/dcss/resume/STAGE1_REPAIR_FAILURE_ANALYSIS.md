# Stage 1R 修复失败分析

状态：completed。最终结论基于预注册 Gate，不把工程成功写成方法成功。

## 最终方法失败

- 时间：2026-07-12；git commit：`e02a752`。
- 配置：D1–D6 均由历史 E4 配置做相对倍数覆盖；命令记录在各 `diagnostic_*/command.txt`。
- checkpoint：clean surrogate `checkpoints/voc20_surrogate.pt`；原 Q 来自 Stage 0 `subspaces.pt`；no-P_t Q 来自 `no_pt_20260712_v2/subspace.pt`。
- 数据：800/200 VOC mini split；筛选只做 1 epoch、400 张 person 图相关的 universal protected-data optimization，不训练 victim。
- 失败阶段：Stage 1R-C2 mechanism screening。
- 预期：至少一项同时满足 coverage、target energy、NT leakage、R_shift、finite 和 budget Gate。
- 实际：六项全部只在 NT leakage Gate 失败；最优 D5 为 0.2113 > 0.1803。
- 日志/指标：`artifacts/dcss/resume/diagnostic_summary.csv` 与 `diagnostic_gate.json`。
- 初步原因：固定 universal carrier 下 target shift 与 non-target units 仍强耦合；P_t 确实是部分泄漏来源，但移除 P_t 不足以达到选择性门槛；更高 leakage 权重没有产生单调修复。
- 已排查：NaN/Inf、coverage、assignment overlap、扰动预算、target energy 不足、R_shift 不足均不是失败原因。
- 修复内容：仅测试预注册的 margin/leakage 相对倍率与 no-P_t 因果消融。
- 修复后结果：D1–D6 无通过项；正式复验未触发。
- 是否影响 E0–E4：否；全部新 run 使用独立目录。
- 分类：mechanism failure + selectivity failure + method failure；旧 E0 另有 baseline underfitting，E4 另有 transfer failure。

## 2026-07-12 diagnosis/no-P_t v1 归档失败

- git commit：`e02a752`；数据为既有 E2–E4 CSV 与 Stage 0 `raw_statistics.pt`。
- 失败阶段：Stage 1R-B/C1 派生文件归档。
- 预期：写出统一比较 CSV 与 no-P_t config YAML。
- 实际：CSV 字段集合只取首行，E4 增量列触发 `ValueError`；命令处理函数进入 YAML，触发 `RepresenterError`。
- 分类：implementation failure；核心输入与求解数值未报错。
- 修复：CSV 使用所有行字段的有序并集；YAML 排除 `func` 字段。
- 历史影响：无。v1 目录保留，不覆盖 E0–E4；修复后使用独立 v2 run id。
