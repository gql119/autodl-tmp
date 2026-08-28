# TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1

## 1. 状态与边界

- SpecID：`TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1`
- 状态：`approved`
- 批准时间：`2026-08-28`（当前任务中用户明确批准）
- 拟用 ExpID：`TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT`
- 父证据：`TAUSB-SDH-DGCAIP-S0-R4-DIAG`
- 冻结基线提交：`4eb064ade919fecec6d1466900442e9f9a9a2bf5`
- 目标类：`person`（VOC class id `14`）
- 范围：仅定位 P1 首个不确定性来源；不优化方法，不训练 victim，不生成 AP50。

本 Spec 已授权生成执行 CSV 与进行本地实现/验证。普通推送、远端 checkout
与 GPU 执行仍分别受后续门禁约束。

## 2. 为什么现在只审计 P1 determinism

R4 已完成真实 YOLOv8/TAL 机制路径，41.57 秒内正常退出，无 NaN、Inf、
CUDA OOM 或 Traceback。但是 P1-A/P1-B 在相同进程内仅 7/47 个数值比较
通过冻结容差，40/47 失败；全部 40 个结构检查仍相同。

更重要的是，第 0 步已经出现差异：

- 八个 batch 的磁盘文件 hash 全部相同；
- P1-A/P1-B 的适配器初始 hash 相同；
- 第 0 步 routed-gradient SHA256 已不同；
- 第 0 步 `max_projected_row_dot` 相差约 `2.861e-6`；
- 后续差异随更新累积，step 7 attack retention 相差约 `0.03880`。

因此，继续跑八步、P2/P4、victim 或放宽容差不能定位原因。本审计只复现
首个 batch 的首个 P1 梯度，不提交任何参数更新，并按计算图顺序找出第一个
发生差异的张量或算子。

## 3. 当前潜在缺陷清单

### 3.1 已确认的审计契约缺陷

1. **输入一致性证据不足。** 当前 `_batch_sha256` hash 的是图像/标签文件与
   路径，不是 `load_sdh_batch` 后真正送入模型的图像、标签、边界框及索引
   张量。磁盘 hash 相同不能证明 CUDA 输入张量逐字节相同。
2. **初始状态证据不足。** 当前只 hash `adapter_parameters(carrier)`，没有覆盖
   完整 carrier、YOLO 参数和 buffers、原型 bank、校准权重、模块 train/eval
   状态、已有 `.grad`、hook/capture 缓存和 engine counter。
3. **RNG 没有冻结和恢复。** `_clone_detector_carrier` 会先构造一个含卷积层的
   新 carrier，再加载源 `state_dict`。构造过程会推进全局 Torch RNG，现有
   P1-A/P1-B 重放没有保存和恢复 Python、NumPy、Torch CPU/CUDA RNG。
4. **首个分歧点没有记录。** R4 只保留最终标量与 routed-gradient hash；一旦
   梯度 hash 不同，无法判断差异来自载体渲染、YOLO 前向、TAL、组件梯度、
   SVD 投影还是显式 NLA 梯度。

### 3.2 高优先级但尚未证实的运行来源

1. person 框渲染包含两次 CUDA bilinear `interpolate`，其 backward 与卷积
   backward 是候选非确定性算子。
2. CGR 使用 `torch.linalg.svd` 构造行空间。近似奇异值、并列子空间或 CUDA
   求解器数值差异可能使投影结果变化；应比较行空间 projector，而不是直接
   比较可能发生符号翻转的奇异向量。
3. P1-A/P1-B 共享一个 `SDHObservationEngine` 和 YOLO 实例。虽然模型处于
   eval 且现有代码会取走 capture 记录，仍需排除 counter、hook records、TAL
   缓存、buffer 或残留 grad 的跨重放污染。
4. TAL/相关路径中的 `topk` 若遇到相等或近似相等得分，正样本索引可能发生
   不稳定选择。现有审计没有 hash `fg_mask`、`target_labels`、`target_scores`
   与 `target_gt_idx`。

### 3.3 不在本次修改范围内的机制风险

1. 当前 CGR 只保证 `projected_target_gradient` 与非目标行空间正交；随后显式
   加入的 NLA/protection gradient 使最终更新不必正交。这是当前实现的明确
   语义，不应在 determinism 审计中悄悄修改，但后续必须确认它是否符合论文
   方法表述。
2. R4 中 target reveal、NLA 和 DGCAIP distribution 校准权重触及上限 100，
   表明组件尺度可能失衡。只有 determinism 恢复后才允许另立 Spec 讨论。
3. P2/P4 的混合 probability/IoU/alignment 拒绝问题仍存在，但本次不得调整
   权重、容差、步长或 backtracking。

## 4. 研究问题与冻结假设

### Q1

在同一 R4 首批次、同一完整初始状态且不更新参数时，P1 的第一个差异发生在
计算图的哪一层？

### Q2

差异能否由 RNG/状态完全恢复、fresh engine 隔离或 PyTorch 严格确定性模式
消除或显式报出对应非确定性算子？

### H1：未跟踪状态或 RNG 依赖

如果普通共享 engine 重放失败，而完整 RNG/状态恢复或 fresh engine 重放通过，
则支持 H1。

### H2：CUDA 非确定性算子

如果完整状态和 fresh engine 仍失败，但严格确定性子进程通过，或
`torch.use_deterministic_algorithms(True)` 明确报出算子，则支持 H2。

### H3：SVD/离散选择数值不稳定

如果上游 target/NLA component gradients 相同，首次差异出现在 SVD projector、
rank 或 TAL assignment/top-k 索引，则按对应预注册标签记录。

H1-H3 是定位假设，不是方法效果假设。

## 5. 冻结输入与不变量

除本 Spec 明确减少执行范围外，继承 R4：

- seed `0`、VOC20、person id `14`、image size `640`；
- 相同 surrogate checkpoint 及 SHA256；
- 相同 secret manifest、primary secret 及 tensor SHA256；
- 相同 hiding checkpoint、P1 source state、D0 report 及各自 SHA256；
- 相同 carrier、目标组件、D-LFC、CICR、NLA、CGR、`eps=16/255`；
- 相同 `batch_size=4`、P1 学习率、容差和 backtracking 定义；
- 使用 R4 calibration split 的**第一个完整 batch**，样本次序不得变化；
- P1 target/component calibration 与 prototype banks 使用 R4 逻辑产生一次，随后
  冻结并 hash；
- `abs_tol=1e-6`、`rel_tol=1e-4` 保持不变。

每个比较 pair 必须复用同一个已经加载完成的 CPU batch snapshot，再分别复制到
GPU；不得在两个 repeat 中独立重新读取 JPEG。磁盘文件 hash 与 CPU/CUDA 张量
hash 都必须记录。

## 6. 最小审计设计

整个 GPU 审计只有一个 boot，包含两个受控子进程；每个 lane 只计算一个 P1
首步，不接受更新，不进入下一 batch。

### Lane N：正常后端，定位状态依赖

在 R4 默认后端设置下执行：

0. 按 R4 顺序重放 16 个 `dist` calibration batch、4 个权重 warm-up batch 与
   24 个只读 held-out batch，使首个 P1 前的 engine counter/TAL cache 与 R4
   一致；不保留 held-out feature tensor，不生成 held-out 指标；

1. `N-shared-A/B`：保持 R4 的共享 engine 顺序，复现现象并输出完整 trace；
2. `N-reset-A/B`：每次 repeat 前恢复完整 snapshot、全部 RNG、`.grad=None`、
   train/eval flags 和 engine/capture 空状态；
3. `N-fresh-A/B`：从同一冻结 snapshot 分别建立 fresh YOLO/engine/carrier，
   恢复同一 RNG 后运行。

若 `N-shared` 本身未复现 R4 的首步 mismatch，仍继续其余 lane，并标记
`baseline_not_reproduced`，不得据此宣称问题已解决。

### Lane D：严格确定性子进程

在独立子进程启动 CUDA 前固定：

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- `torch.backends.cudnn.benchmark=False`；
- cuDNN deterministic 开启；
- CUDA matmul/cuDNN TF32 关闭；
- `torch.use_deterministic_algorithms(True)`。

随后执行与 `N-fresh-A/B` 相同的两次首步计算。若 PyTorch 因非确定性算子
失败，保存完整 operator 名称和 stack，正常终止为诊断结果，不做替代实现或
自动降级。

### 禁止事项

- 不执行 P2、P3、P4；
- 不做第二个 optimization step；
- 不写入 accepted candidate，不改变 carrier 参数；
- 不生成 poisoned dataset，不训练 clean/poison victim，不算 AP50；
- 不改变 loss、权重、阈值、SVD tolerance、step size 或 backtracking；
- 不自动修复、不自动重跑。

## 7. 有序 trace 契约

每个 repeat 按以下固定顺序记录 tensor SHA256、shape、dtype、device、有限性；
比较时同时给出 exact match、max absolute error、max relative error 和冻结
allclose 结论：

1. `input.cpu`：图像、batch indices、class ids、normalized/pixel boxes；
2. `input.cuda`：实际送入模型的对应张量与 secret；
3. `state.pre`：完整 carrier/model state、buffers、training flags、prototype
   banks、校准权重、RNG/backend、engine counter、capture/cache/grad 状态；
4. `render`：canonical delta、box-resized patch、合并 perturbation、poisoned image；
5. `clean.forward`：P3/P4/P5 分类/回归 tower features、decoded predictions、
   `pred_scores_logits`、`pred_bboxes`；
6. `clean.tal`：`fg_mask`、`target_labels`、`target_scores`、`target_gt_idx`；
7. `poison.forward`：与 clean.forward 对应的全部张量；
8. `loss.components`：easy、reveal、RMS、D-LFC、CICR、floor、逐类 NLA；
9. `grad.components`：每个目标组件梯度、目标合成梯度、逐类归一化 NLA 梯度、
   NLA 总梯度；
10. `route.matrix`：active class order、constraint matrix、singular values、rank；
11. `route.projector`：小维度 `constraint_matrix @ constraint_matrix.T`、
   projected target gradient 与被移除的 target 分量；禁止显式构造参数维度的
   `V_r^T V_r`，以避免平方级显存开销；
12. `route.final`：显式 NLA 分量、最终 routed gradient；
13. `candidate.eval`：只构造首个候选并评估逐类 probability drop，不写回；
14. `state.post`：完整 state/RNG/cache/grad 状态及意外 mutation 列表。

独立 comparator 必须返回 `first_divergent_stage`。浮点 tensor 的 hash 不同但
通过冻结 allclose 时记录为 `bitwise_only_drift`，仍保留首差异，但不跳过后续
trace。

## 8. 预注册主标签

每次审计必须按以下优先级输出**恰好一个**主标签；可以同时记录非主标签 flags：

1. `invalid_input_tensor_mismatch`：repeat 前 CPU/CUDA 输入张量不同；
2. `invalid_initial_state_mismatch`：完整初始模型/carrier/bank/calibration 状态
   未能恢复一致；
3. `rng_state_dependency`：`N-shared` 失败而 `N-reset` 通过；
4. `shared_engine_state_dependency`：`N-reset` 失败而 `N-fresh` 通过；
5. `cuda_nondeterministic_operator`：strict 模式报出明确算子，或 normal fresh
   失败而 strict fresh 通过；
6. `tal_assignment_instability`：上游 clean prediction 通过而 TAL assignment
   首先不同；
7. `svd_subspace_instability`：上游 component gradients 通过而 SVD rank/
   projector 首先不同；
8. `upstream_forward_backward_drift`：首次差异位于 render、YOLO forward 或
   component gradient，且 strict 模式未给出更具体标签；
9. `baseline_not_reproduced`：无法复现 R4 mismatch，且所有有效 pair 均通过；
10. `unresolved_first_divergence`：证据完整但以上规则均不满足。

若输入或初始状态无效，结果只能使用前两个 `invalid_*` 标签，不能对 CUDA、
TAL 或 SVD 作因果解释。

## 9. 成功与失败信号

### 机械成功

只有同时满足以下条件才算审计完成：

1. 所有计划 lane 均有 terminal record，或 strict lane 有明确 deterministic-op
   error record；
2. 输入和完整初始状态比较完整；
3. 所有已产生 tensor 有限；
4. trace 顺序完整且 comparator 给出首个分歧点；
5. 输出恰好一个预注册主标签；
6. 运行没有更新 carrier，也没有产生 victim/artifact 越界输出；
7. controller 在硬上限内退出并留下关机请求证据。

机械完成不要求 replay 通过。`unresolved_first_divergence` 是有效但未解决的
诊断结果，所有结果都必须保留并报告。

### 独立失败

任一情况使本次运行无效或失败：

- 缺少 required trace、state hash、comparator 或 terminal；
- NaN、Inf、CUDA OOM、未捕获 Traceback；
- normal lane 改变了方法参数或 strict error 被静默忽略；
- artifact root 与控制/cache/tmp 路径落到系统盘；
- 达到 300 秒硬上限；
- controller 未请求关机。

## 10. 成本、存储与关机硬门禁

- GPU boot：最多一次；GPU 开启前先完成无卡测试和 pre-run review。
- 预期 GPU 计算：初始化后 1–2 分钟。
- **总 wall-clock 硬上限：300 秒，包含 controller 启动后的初始化与两个子进程。**
- 单进程异常立即停止，不尝试第二轮或自动修复。
- success、diagnostic error、failure、timeout 均必须执行 AutoDL shutdown。
- controller、artifact、cache、CUDA cache、TMP 全部位于已验证的数据盘
  `/root/autodl-tmp` 下；系统盘只允许只读系统依赖。
- 证据集硬上限 100 MiB；不保存 checkpoint、完整 feature tensor 或图片副本，
  只保存 hash、摘要、最小差异切片和必要 stack。

## 11. 最小实现范围与测试门禁

批准后只能新增独立 audit runner、trace/comparator helper、配置和测试。原 P1
算法路径默认行为必须不变；若为了观察必须加 hook，必须 default-off 且固定
回归证明 feature-off 输出不变。

无卡测试至少覆盖：

1. batch tensor hash schema 用合成张量验证字段遗漏会 fail closed；GPU 端实际
   张量一致性留给唯一一次受控运行验证；
2. full-state snapshot/restore 能检测一个 buffer、training flag、grad 或 RNG
   的单点变化；
3. fresh/shared engine 分类逻辑；
4. ordered first-divergence comparator；
5. SVD 用行空间 Gram 与 projector 对 target 的作用结果比较，而非比较可能
   符号翻转的奇异向量；
6. strict deterministic operator error 能转成有效诊断 terminal；
7. exactly-one-label 决策树；
8. feature-off regression；
9. 300 秒 controller、数据盘路径、unique artifact root 与 auto-shutdown；
10. focused pytest、配置验证、Bash syntax 和 stray-token scan 全部通过。

## 12. 必需产物

- frozen resolved config、source commit、config SHA256；
- normal/strict backend 与 CUDA/PyTorch/cuDNN/GPU 环境清单；
- `input_state_manifest.json`；
- `p1_trace_normal.json`；
- `p1_trace_strict.json` 或 `strict_operator_error.json`；
- `first_divergence_report.json`；
- `determinism_audit_summary.json`，含恰好一个主标签；
- controller/wrapper terminal、outer log、shutdown request evidence；
- minimal pull manifest 与 SHA256；
- 本地 H→E→N 结果说明。

## 13. 结论边界与后续决策

本审计最多可以说明 P1 首个不确定性来自哪一计算阶段，以及哪种受控设置能否
消除它。它不能说明 person 不可学习、非目标保持、DG-CAIP 有效、鲁棒性、
迁移性或 AP50 改善。

审计后只允许以下单步决策：

- 若定位到明确状态/RNG 问题：另立最小修复 Spec；
- 若定位到明确 CUDA 算子：另立确定性替代或生产容差策略 Spec；
- 若定位到 TAL/SVD：另立对应的稳定化 Spec；
- 若仍 unresolved：停止调参，依据首分歧 trace 设计一个更窄的 CPU/CUDA
  单算子复现；
- 在 determinism 修复并通过配对重放前，不恢复 P2/P4 调参或 victim 实验。

## 14. 执行门禁

用户明确批准本 Spec 后，才可：

1. 生成独立执行 CSV；
2. 在新分支实施最小 audit runner；
3. 通过本地无卡测试与独立 pre-run review；
4. 绑定 exact commit/config hash 并普通推送；
5. 完成远端无卡 checkout 和数据盘路径核验；
6. 请求用户开启一次 GPU，执行唯一一次 300 秒硬上限审计；
7. 自动关机并拉回最小证据集。
