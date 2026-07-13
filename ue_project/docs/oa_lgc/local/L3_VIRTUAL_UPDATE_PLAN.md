# L3 Multi-step Virtual Update 计划

以 `torch.func.functional_call` 和参数副本构造 clean/poison 两条独立轨迹，支持 J=1/3/5 以及 `head_only`、`detection_head`、`selected_modules`、`full_model` 参数选择。base model state_dict 必须逐 tensor 不变，不使用持久 optimizer state。

本地 detector proxy 对标注对象 crop 做 20 类分类与 box 回归，只用于验证数学、内存和梯度链。完整 YOLO functional virtual update 不在本地 Gate 内。

