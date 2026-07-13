# L1 Object-Aligned Carrier 实现

- `oa_lgc/carrier.py`：`CarrierConfig`、`CarrierResult` 和 `apply_object_aligned_carrier`。
- `delta_obj` 是标准对象坐标中的唯一扰动张量；按目标框 resize 后应用 soft box mask。
- non-target 框先栅格化并可配置膨胀；valid mask 在施加扰动前严格清零 overlap。
- 低于 `min_valid_fraction` 的实例记录原因并跳过；不会 silent fallback。
- 多目标扰动累加，最终在图像坐标按 eps 投影，并再次强制 non-target 区域为零。
- `oa_lgc/artifacts.py` 提供带微秒的 run id、存在即报错的目录创建和 Unicode-safe PNG 写出。
- box jitter 字段存在但本地默认 0；非零值明确报 `NotImplementedError`，不静默启用未验证行为。

