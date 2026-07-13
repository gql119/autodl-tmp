# L1 Carrier 失败分析

## 首轮面积口径测试失败

- 日期时间：2026-07-13；branch：`codex/oa-lgc-local-chain`；起始 commit：`f1f3d0f`。
- Python/CUDA：同机 Python 3.12.13，PyTorch 2.11.0+cu128，RTX 2070。
- 配置：`configs/oa_lgc/local/carrier.yaml`。
- 命令：`python -m pytest tests/test_oa_lgc_carrier.py -q`。
- 数据/checkpoint：此单测为 synthetic 32×32 image，不使用 checkpoint。
- 失败阶段：L1 area metrics test。
- 预期行为：`perturbed_area <= actual_support_area`。
- 实际行为：0.125 > 0.102539；13 passed，1 failed。
- traceback：断言失败，无运行时异常。
- 初步原因：`actual_support_area` 使用 soft mask 权重均值，而 `perturbed_area` 使用二值非零面积，两种口径不可直接比较。
- 修复：`actual_support_area` 与 `valid_support_area` 改为二值几何覆盖；新增 `support_weight_mass` 与 `valid_support_weight_mass` 保留 soft 权重质量。
- 是否影响历史代码/实验：否；仅 OA-LGC 新模块。
- 分类：test failure / implementation failure。

修复后结果：L1 tests 14/14 通过；与历史 DCSS tests 合并为 53/53 通过。No blocking failure was triggered.

历史 `dcss/stage15.py::object_aligned_warp` 缺少本阶段要求的 soft edge、插值配置、跳过原因和完整面积指标，分类为 implementation gap，不修改历史实现；OA-LGC 在独立命名空间补齐。
