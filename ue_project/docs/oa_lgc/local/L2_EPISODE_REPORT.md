# L2 Disjoint Support–Query 报告

状态：**pass**。证据：`artifacts/oa_lgc/local/20260713_220613_368856_L2_seed0/`。

真实 mini VOC episode 使用 4 个 support 和 4 个 query 原始图像：

- support：003860、009577、000159、000048。
- query：001460、001414、000752、007048。
- overlap count：0。
- 四分支 clean/poison pair identity：pass。
- support/query 均有 person：pass。
- 本 episode 可计算的非目标类：class 6。

8 项 L2 测试覆盖 seed 可复现/变化、严格互斥、pair identity、数据不足显式失败、augmentation ID、不缺 target、class validity 和多 worker episode 内互斥。与 L1 和历史测试合并为 `61 passed in 3.27s`。

L2 Gate：ID overlap=0、pair 对齐、target 两侧有效、无 silent fallback、可复现，全部 pass。

