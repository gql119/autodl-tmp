# L5 Objective 失败分析

## 首轮 budget 断言精度失败

- 日期：2026-07-13；branch：`codex/oa-lgc-local-chain`；起始 commit：`22034a3`。
- 环境：Python 3.12.13、PyTorch 2.11.0+cu128；synthetic CPU test。
- 命令：`python -m pytest tests/test_oa_lgc_objective.py -q`。
- 预期：投影后 max abs delta <= 0.2。
- 实际：float32 表示为 0.20000000298023224；7 passed，1 failed。
- 原因：test 使用无容差的 Python double 比较，不是 projection 越界。
- 修复：断言加入 `1e-7` 表示容差；projection 算法不变。
- 分类：test failure / numerical failure。
- 历史代码/实验影响：无。

测试还主动覆盖 checkpoint overwrite、empty authorized 与 invalid target protect；均按预期显式处理，无 silent fallback。修复后无 blocking failure。
