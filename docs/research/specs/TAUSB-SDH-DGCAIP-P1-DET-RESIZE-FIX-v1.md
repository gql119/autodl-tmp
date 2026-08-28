# TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1

## 1. 状态与目标

- SpecID：`TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1`
- 状态：`approved`
- 批准时间：`2026-08-29`（当前任务中用户明确批准）
- 日期：`2026-08-29`
- 拟用 ExpID：`TAUSB-SDH-DGCAIP-S0-P1-DET-RESIZE-FIX`
- 父审计：`TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT`
- 冻结基线提交：`b2fa96f98ea88d6b347bbbf751768a06e983d47c`
- 目标类：VOC `person`（class id `14`）

本 Spec 只修复阻止当前方法稳定落地的可微 resize 非确定性，并完成一次
真实 P1 写回 smoke。它不调节方法权重，不改变载体、目标损失、非目标保护、
候选接受或回溯语义，也不训练 victim、不生成 AP50。

## 2. 已确认事实

上一轮 P1 determinism audit 已机械完成，唯一主标签为
`cuda_nondeterministic_operator`：

- 输入张量一致，reset/fresh 初始状态一致，所有重复均未更新参数；
- render、clean/poison forward、TAL assignment 和标量损失在梯度前一致；
- 首个 bitwise 差异位于 `grad.components/cicr`，首个超过冻结容差的差异位于
  `grad.components/dlfc`；
- strict lane 在 `torch.autograd.grad` 中明确报错：
  `upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation`；
- 当前 PyTorch/CUDA 环境为既有 AutoDL 环境，本 Spec 不升级依赖。

PyTorch 的确定性接口会在没有确定性实现时抛错；仅使用
`warn_only=True` 会保留非确定算子，不能作为修复。官方参考：

- <https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>
- <https://docs.pytorch.org/docs/stable/notes/randomness.html>

## 3. 当前方法契约：全部冻结

本轮必须保持当前已经敲定的方法：

1. **SDH person-box carrier**：同一低频/高层语义 secret 以每个 person 边界框
   为宿主，由隐藏网络产生有细微宿主差异的 sample-wise 扰动；
2. **D-LFC**：在检测器原生特征空间集中 canonical carrier 的隐特征；
3. **CICR**：约束 P3/P4/P5 上 person 实例的 clean→poison 特征残差方向一致，
   并保留残差能量下限；
4. **目标检测攻击目标**：保持现有 classification、box、alignment、distribution
   以及 reveal/RMS 组合定义；
5. **NLA**：对 clean/poison 的非目标类预测 logit/响应进行显式对齐；
6. **DG-CAIP**：以非目标预测分布差异识别 person 共现样本中保护不足的类别；
7. **CGR**：目标攻击梯度投影到逐类非目标梯度行空间的正交补；随后显式加入
   NLA 下降分量。正交性约束的是 target component，不要求带 NLA 的最终梯度
   与保护行空间正交，这是当前冻结语义；
8. **非线性回溯**：沿现有最终 routed gradient 评估逐类非目标约束并决定接受。

不得恢复旧 `tausb_mask`、Fourier universal perturbation、ALCE/PAG 或 late repair。

## 4. 代码级故障边界

当前 person-box renderer 中有两处 bilinear resize：

1. `crop → carrier.input_size`：输入图像不需要梯度；它只产生固定 host，
   不应进入 adapter 的反向链；
2. `output.delta → person box size`：`output.delta` 依赖 adapter 参数，当前
   `F.interpolate(..., mode="bilinear")` 的 CUDA backward 会进入 P1 目标梯度，
   是本轮必须替换的路径。

实现前必须用 autograd reachability 单测确认上述边界。若发现另一处
gradient-reachable bilinear resize，必须记录明确路径并停止扩大修改，不能顺手
重构其他 renderer。

## 5. 选定修复

### 5.1 确定性 separable bilinear resize

在 `semantic_hiding_carrier.py` 中新增一个小型、无参数 helper，仅用于
`output.delta → box patch`：

- 按 `align_corners=False` 坐标公式生成高度和宽度的固定线性插值矩阵；
- 每个目标坐标仅有两个非零邻域权重，权重不参与梯度；
- 通过两次二维矩阵乘法完成 separable bilinear resize；
- adapter 梯度只经过确定性 matmul backward，不再经过
  `upsample_bilinear2d_backward_out_cuda`；
- normal、strict 和正式 mechanism 共用同一实现，禁止按审计模式切换算法；
- 保持输入 dtype/device，不允许 CPU 往返、`.detach()` adapter 输出或
  `warn_only=True`。

生产与审计控制器继续固定 `cuda.matmul.allow_tf32=False`。strict/GPU 正式
机制进程必须在 CUDA 初始化前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并启用
现有严格确定性后端。

### 5.2 host 梯度边界显式化

`crop → canonical host` resize 保留现有 bilinear forward，但在代码中显式置于
no-grad 边界；host 数值不得改变。该变更只防止未来误把图像 resize backward
接入 adapter 优化，不改变 carrier 对 host 像素的条件依赖。

### 5.3 明确禁止的替代

- 不改成 nearest/grid-sample；
- 不把 resize 放到 CPU；
- 不通过放宽 allclose、关闭 strict mode 或 `warn_only=True` 掩盖问题；
- 不升级 PyTorch、CUDA、Ultralytics；
- 不改变 secret、person box、epsilon、loss、权重、SVD tolerance、学习率、
  backtracking 或数据 split；
- 不借本轮处理 calibration weight 达到上限、P2/P4 效果或 AP50 问题。

## 6. 无卡实现与测试门禁

批准后先在本地和远端无卡环境完成：

1. **forward parity**：覆盖上采样、下采样、混合长宽、`1×N`、`N×1` 和
   VOC 首批次真实框尺寸；与 CPU `F.interpolate(..., bilinear,
   align_corners=False)` 比较，float32 `max_abs <= 2e-6`、
   `allclose(atol=2e-6, rtol=1e-5)`；
2. **gradient parity**：在 CPU 上比较旧实现和新 helper 对输入/adapter 的梯度，
   `allclose(atol=2e-5, rtol=1e-4)`；
3. **reachability**：host resize 不产生 image gradient graph，patch resize 保留
   非零、有限 adapter gradient；
4. **carrier invariant**：`Linf <= 16/255`、person union support 外严格为零、
   overlap 仍取平均、canonical delta/reveal 数量不变；
5. **method invariant**：D-LFC、CICR、目标 objective、NLA、DG-CAIP、CGR 和
   backtracking 的配置与序列化 hash 不变；
6. focused tests、相邻回归、config validator、Python compile、Bash syntax、
   CSV schema 和 stray-token scan 全部 exit 0；
7. 不修改或提交现有 R4 dirty evidence、数据集、权重和历史 artifacts。

任何 parity 或 invariant 失败都在无卡阶段停止，不开启 GPU。

## 7. 唯一一次 GPU 门禁

GPU 只允许一个 boot，总 wall-clock **硬上限 480 秒**。所有 terminal（pass、
fail、timeout、diagnostic error）都必须自动请求关机。禁止在同一 boot 中修代码、
重启或尝试第二种 resize。

### G0：算子 microprobe（最多 60 秒）

- 使用实际 AutoDL PyTorch/CUDA、首批次真实 person 框尺寸；
- strict mode 下对新 resize 的 forward/backward 重放三次；
- 要求无 deterministic-op error、梯度有限，三次 forward 与 input gradient
  bitwise 相同；
- 记录旧 bilinear 路径的已有错误为父证据，不再次耗费 GPU 复现；
- 32 个代表性 resize 的总时间和峰值显存必须记录。若 render 估算使完整 P1
  超过本轮硬上限，停止并标记 `performance_gate_failed`。

### G1：冻结 P1 strict replay（G0 通过后）

复用父审计的同一 source commit bindings、config、首批次、16 个 distribution
calibration batch、4 个 weight warm-up batch 和 24 个 held-out batch：

- strict fresh-A/B 各执行一个首步，零参数更新；
- normal lane 仅保留一对 reset replay作为诊断，不作为 scientific pass 必要条件；
- strict pair 的输入、初始状态、render、forward、TAL、loss、component gradients、
  route matrix/projector/final gradient 和 candidate drop 必须 bitwise 相同；
- 不得出现任何新的 unsupported deterministic operator；
- 所有状态保持未更新。

若 normal reset 不一致但 strict fresh 精确一致，记录
`deterministic_backend_required`，正式 mechanism 必须使用 strict backend，
不再阻塞效果验证。

### G2：两步真实写回 smoke（G1 通过后）

- 从同一冻结 P1 source state 启动，执行最多两个真实 optimization step；
- 只允许 adapter 参数变化；YOLO、hiding trunk、reveal decoder、prototype banks
  和冻结 calibration state 不得变化；
- 每步 loss/gradient/candidate/constraint 全部有限；
- 至少一个 candidate 被接受且 adapter hash 改变；
- `Linf/support` 继续通过，输出的 P1 smoke state 可加载；
- 不运行 P2/P4、materialization、victim 或 AP50。

若两个候选均被现有非线性约束拒绝，标记 `algorithmic_no_acceptance`；这不是
继续调试代码的许可，而是后续方法诊断输入。

## 8. 唯一结论标签

GPU 结果按优先级输出恰好一个主标签：

1. `invalid_binding_or_input`
2. `resize_forward_or_gradient_mismatch`
3. `performance_gate_failed`
4. `new_cuda_nondeterministic_operator`
5. `strict_replay_mismatch`
6. `algorithmic_no_acceptance`
7. `repair_pass`

基础设施错误另行记录为 `infra_failure`，不得伪装成方法结论，也不得自动重跑。

## 9. 修复完成判据

只有以下条件全部满足才允许进入效果实验：

1. 无卡门禁全部通过；
2. G0 严格 backward 三次 bitwise 一致；
3. G1 strict fresh-A/B 全 trace bitwise 一致，且不再出现
   `upsample_bilinear2d_backward_out_cuda` 或其他 unsupported operator；
4. G2 至少一次真实写回被接受，adapter state 改变且其余冻结状态不变；
5. epsilon、person-box support、finite、存储和自动关机门禁全部通过；
6. 唯一主标签为 `repair_pass`。

## 10. 成本、存储与证据

- GPU：一次 boot，硬上限 480 秒；预计有效计算 2–5 分钟；
- 证据集硬上限 100 MiB；
- artifact、cache、TMP、controller、P1 smoke state 全部位于 AutoDL 数据盘
  `/root/autodl-tmp`；系统盘不得新增实验数据；
- 必需产物：resolved config/source hash、operator microprobe JSON、strict replay
  trace/comparator、writeback smoke metrics、state mutation report、controller
  terminal、shutdown evidence、manifest SHA256 和本地 H→E→N 分析；
- 所有失败结果同样保留，不删除历史运行。

## 11. 效果实验的直接衔接

若本 Spec 得到 `repair_pass`，下一步不再讨论载体或继续做微型调试，直接冻结
该 P1 state，建立一轮 seed-0 paired E20 pilot：

- VOC2007+2012、YOLOv8n、person id 14；
- C0 clean 与 M1 poison 使用相同训练协议；
- 稀疏数据集只保存含 person 的加噪 JPEG，其余训练图片直接引用数据盘原图；
- 报告 person AP50、19 个非目标类各自 AP50、非目标宏平均、下降量和保持率；
- 无论是否达到成功判据都保留并报告结果。

E20 的训练时长、关机和拉取清单另用一个短效果实验 Spec 冻结；本修复 Spec
不得借机启动 victim 训练。

## 12. 批准门禁

本 Spec 已获用户明确批准，可以：

1. 建立新分支并实施上述最小修复；
2. 完成本地/远端无卡门禁、独立 pre-run review 与普通非 force push；
3. 请求用户开启一次 GPU；
4. 执行 G0→G1→G2 的单 boot、480 秒硬上限流程；
5. 自动关机、拉回最小证据并给出唯一标签；
6. 仅在 `repair_pass` 后制定并进入 paired E20 pilot。
