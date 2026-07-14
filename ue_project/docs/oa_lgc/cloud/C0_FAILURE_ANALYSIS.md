# C0 Failure Analysis

时间：2026-07-14

分支：`codex/oa-lgc-real-yolo-pilot`

commit：`04448a338239863d71a12198ede2fb08980be3a0`（C0 提交前）

环境：Windows 11；Python 3.12.13；PyTorch 2.11.0+cu128；Ultralytics 8.4.90；RTX 2070 8 GiB

配置：mini VOC，VOC20 surrogate，imgsz 320，batch 1，target class 14

## Failure 1：含中文路径跨解释器编码损坏

- 命令：从外部虚拟环境加载当前副本中的 `checkpoints/voc20_surrogate.pt`。
- 预期：Ultralytics 识别 `.pt` 并加载模型。
- 实际：路径中的“副本”被损坏，Ultralytics 将错误片段当成 suffix。
- traceback：`AssertionError: ... acceptable suffix is {'.pt'}, not .f://autodl-tmp -`。
- 原因：PowerShell 到外部 Python 的路径编码边界，不是 checkpoint 损坏。
- 修复：使用纯 ASCII 原工作区中的同 SHA checkpoint 与数据；当前分支源码通过 `PYTHONPATH` 加载。
- 修复后：checkpoint 成功加载；两份 checkpoint SHA256 完全一致。
- 历史影响：无；没有改写或删除任何 checkpoint。
- 下一阶段：允许，但 C1-C3 命令必须记录双 workspace 边界。

## Failure 2：旧 checkpoint 参数配置与新 Ultralytics 不兼容

- 命令：直接调用 `DetectionModel.loss(batch)`。
- 预期：输出 box、cls、DFL loss。
- 实际：`AttributeError: 'dict' object has no attribute 'box'`。
- 原因：checkpoint 加载后的 `model.args` 只包含 task/data/imgsz/single_cls/model；Ultralytics 8.4.90 loss 读取属性式 `box/cls/dfl`。
- 修复：通过 `ultralytics.cfg.get_cfg` 合并完整配置，显式记录有效 gains 7.5/0.5/1.5。
- 修复后：三项 loss 均为有限值并能相加。
- 历史影响：无；只修改运行时对象。
- 下一阶段：允许；adapter 必须把兼容处理做成显式、可审计步骤。

## Failure 3：EMA checkpoint 参数默认冻结

- 命令：首个 C0 artifact run，计算原生 loss 后执行 backward。
- 预期：loss 对模型参数可反向。
- 实际：`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`。
- 原因：加载的 EMA 模型参数 `requires_grad=False`。
- 修复：运行时显式开启模型参数梯度，再由 adapter 严格选择 fast parameter subset。
- 修复后：总梯度范数 26.973413，有限且非零。
- 历史影响：无；失败 artifact `artifacts/oa_lgc/cloud/20260714_141531_C0_0/` 原样保留。
- 下一阶段：允许。

最终结果：No blocking failure was triggered. C0 Gate 全部通过，允许进入 C1。
