# C0 Implementation Plan

## C1 实施边界

1. 新建独立 `oa_lgc/yolo_adapter.py`，不把 functional update 逻辑塞入 CLI，也不改写受保护的 victim 训练、评估或 TAL 文件。
2. 直接使用 `torch.func.functional_call`；原生 `DetectionModel.loss` 只负责 inner full detection loss。
3. 参数范围由 Detect head 的结构和精确模块路径决定，不使用模糊字符串匹配：
   - `classification_head_only`：`model.22.cv3` 的递归参数；
   - `detection_head`：`model.22` 的递归可训练参数；
   - `selected_neck_and_head`：精确层 `model.15`、`model.18`、`model.21`、`model.22`；
   - `full_model`：保留接口。
4. 每条 clean/poison 轨迹克隆独立 functional buffers，base model 保持固定；运行前后做参数和 buffer hash。
5. J=1 使用 SGD、momentum 0、weight decay 0、`create_graph=True`。先证明 classification-head protect-only mixed derivative，再验证 detection head。
6. query gain 默认使用从 clean query / base state 得到并 detach 的 fixed reference assignment；recomputed 只作为诊断。
7. classwise query loss只聚合 reference TAL positive units，按 target-score mass 归一化；缺失类标记 invalid，不填零参与平均。
8. 不启用 RCDS、QP、ALCE context 或 feature collision，不允许 proxy fallback。

## 已知实现风险

- `functional_call(model, ...)` 与 `model.loss(batch)` 的内部 self-forward 需要验证参数替换是否贯穿原生 loss；必要时先 functional forward，再把 raw predictions 交给明确构造的原生 criterion。
- BatchNorm 在 train mode 会更新传入 buffer；必须给每条轨迹克隆 buffer，且不允许写回 base。
- TAL 的 top-k/index 是离散 stop-gradient 边界；mixed derivative 只声称 assignment 以外的 loss 图可微。
- RTX 2070 只有 8 GiB；C1 从 320、batch 1、J=1 开始。C3 的 J=5 允许按协议记录 memory partial pass，但不得伪造通过。
- 当前仅 9.53 GiB 磁盘空闲，足够 C1-C3 小 artifact；进入 C4 前必须重新评估 clean baseline 与 checkpoints 的空间需求。

## C1 成功标准

真实 forward、原生 box/cls/DFL loss、Mode A/B J=1、base hash 不变、protect-only mixed gradient 非零且有限、无 proxy fallback、相同 seed 可复现。任一失败即停止进入 C2。
