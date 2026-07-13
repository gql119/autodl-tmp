# L3 Multi-step Virtual Update 实现

- `oa_lgc/model.py`：轻量 `ObjectCropDetector`，从真实检测标注提取对象 crop，输出 class logits 与 normalized box。
- `oa_lgc/virtual_update.py`：复制参数/buffer 后以 `functional_call` 前向，使用 `autograd.grad` 更新选定 fast parameters。
- `first_order=true` 在每个后续 inner step 截断旧 fast-weight 历史，但保留当前 support gradient 对输入的 mixed derivative，使 outer loss 仍可回传到 `delta_obj`；这不是完整二阶 MAML。
- `first_order=false` 接口保留完整 fast-weight 链；本地不做大模型二阶实验。
- `head_only` 更新分类头；`detection_head` 更新分类与 box 头；`selected_modules` 由前缀选择；`full_model` 选择全部参数。

