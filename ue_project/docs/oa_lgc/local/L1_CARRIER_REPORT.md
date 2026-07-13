# L1 Object-Aligned Carrier 报告

状态：**pass**。这只表示 carrier 工程链通过，不表示 OA-LGC 方法有效。

## 证据

- artifact：`artifacts/oa_lgc/local/20260713_220255_751416_L1_seed0/`
- 真实 mini VOC 图像：`000009`，原始尺寸 500×375，3 个 person 实例。
- 测试：OA-LGC carrier + 历史 DCSS 共 `53 passed in 3.32s`。

## 关键结果

| 指标 | 值 |
| --- | ---: |
| applied / target instances | 3 / 3 |
| actual support area | 0.108427 |
| valid/perturbed area | 0.048373 |
| raw non-target overlap ratio | 0.553761 |
| direct non-target perturbation max | 0 |
| max abs perturbation | 0.049964 |
| delta gradient norm | 0.000739842 |
| model parameters with gradient | 0 |

raw overlap 较高说明该图适合验证排除逻辑；排除后的 non-target 区域直接扰动严格为 0。输出有限，最大幅度低于 16/255。soft mask、nearest/bilinear/bicubic、小目标、边界裁剪、空有效区跳过和 no-overwrite 均有独立测试。

## Gate

- warp 形状正确：pass。
- target support 与多实例累积：pass。
- non-target overlap 排除：pass。
- 梯度只回传到 `delta_obj`：pass。
- NaN/Inf：无。
- 所有 L1 单元测试：14/14 pass。

