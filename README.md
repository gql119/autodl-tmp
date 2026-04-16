下面我按你现在**已经实现并在跑的代码**，把整套实验方法整理成一版适合组会汇报的说明。你当前的主实现都集中在 `tausb_universal.py` 里，包括：实例级拓扑构建、Fourier 通用扰动、真实 TAL assignment 截获后的 strict 分支、ACGT 相关的 bbox 重标定与 RTP、`M_AL` 上的 prototype 提取，以及 ALCE 的 entanglement/anchor/flat/preserve 总损失。

---

# 1. 你的实验框架是什么

你现在的方法可以概括成三层：

## ALSD：总框架

**Assignment-Localized Shortcut Decoupling**

核心思想是：

> 不再把目标检测里的不可学习样本理解成“简单把 detector 打坏”，而是只在 detector 内部**真实负责目标类预测的单元**上，构造一种**错误 shortcut**。
> 这个 shortcut 让模型对 target 还保留一定语义响应，但逐渐失去可泛化的定位能力。

---

## ACGT：关键算子

**Assignment-Consistent Geometric Transport**

核心作用是：

> 保证图像、padding 后的 bbox、物理 mask、strict assignment、FPN 三层网格在同一几何语义下严格对齐。

你们现在代码里已经包含了：

* padding-aware bbox renormalization
* strict gate 的 1D→2D 反投影
* `M_AL = M_topology * M_assign`

这说明 ACGT 已经不是概念，而是代码主干的一部分。

---

## ALCE：最终算法

**Assignment-Localized Confounder Entanglement**

这一步是你当前代码主目标的核心：

> 不再只做空间坍缩，而是把 `M_AL` 上的 target prototype 拉向**局部背景/上下文 confounder prototype**，同时保住 target 语义方向和能量，再用轻量的空间平坦化辅助。

---

# 2. 你的方法流程是什么

下面按训练流程顺着讲，导师最容易听懂。

---

## 模块 1：物理拓扑构建

### 做什么

对每张训练图像，先构造目标实例的物理 support：

* `inner_t`
* `ring_t`

代码对应：

* `_build_support(...)`
* `build_support_mask(...)`

如果有离线 instance mask，就优先用；没有就 fallback 到伪实例 mask。

### 为什么好

这一步先解决“目标在图像里到底在哪里”的问题。
比直接用 bbox 更细，因为 bbox 会把大量背景和共现目标一起卷进去。

### 为什么会起作用

后面所有局部化干预都建立在这个物理拓扑上。
如果第一步就不准，后面 strict assignment 再精确也没意义。

---

## 模块 2：全局共享扰动载体

### 做什么

你现在不是每张图单独学一个 noise，而是用一组共享 Fourier 系数生成通用扰动：

* `_build_global_freq_pattern(...)`
* `_compose_delta_batched(...)`

实际流程是：

1. 采样中频 Fourier 坐标
2. 用 `fourier_coeff` 生成全局频域模式
3. 结合 JND 和 support mask
4. 得到 `adv` 图像

### 为什么好

这比逐图像噪声更像“类级 shortcut 载体”。

### 为什么会起作用

victim 在训练时更容易把这类 shared pattern 学成一个固定 shortcut，而不是只对某一张图过拟合。

---

## 模块 3：真实 TAL 分配截获

### 做什么

clean 图像先经过 surrogate detector，然后通过 hijacked loss 路径拿到真实 assign 结果：

* `fg_mask`
* `target_labels`
* `target_gt_idx`

在 `train_universal()` 里可以看到：

* `self.hijacked.get_assigned_targets_and_loss(...)`
* `self.hijacked.last_real_assign`

然后构造：

[
G_{strict}=fg_mask \land (target_labels = y_t)
]

也就是 `strict_gate_1d`。

### 为什么好

这和普通基于 bbox/mask 的方法最大不同：

> 你不是猜 detector 关注哪里，而是直接读取 detector 自己的 target assignment。

### 为什么会起作用

这样保留下来的单元，才是 detector 内部**真实负责 target 类预测**的正样本单元。

---

## 模块 4：ACGT 几何一致性

这一块是你们工程上最硬核、也最值得讲的部分。

### 4.1 padding-aware bbox renormalization

#### 做什么

batch 内图像 pad 到统一尺寸后，对 YOLO 归一化框重新缩放。代码里明确调用：

* `renorm_yolo_bbox_after_padding(...)`

#### 为什么好

这修复了以前最容易导致 strict assignment 漂移的 bug：
图像 pad 了，但 bbox 还是原图归一化坐标。

#### 为什么会起作用

修完后，high-resolution 层上的 strict gate 才能回到正确的网格位置，尤其是 `model.15` 不再错位。

---

### 4.2 strict RTP：1D → 2D 反投影

#### 做什么

把 `strict_gate_1d` 按三层 FPN 切片，并 reshape 成：

* `80×80`
* `40×40`
* `20×20`

代码里通过：

* `project_strict_gate_to_fpn(...)`

得到 `M_assign_2d`。

#### 为什么好

因为 assigner 给的是 `[B, 8400]` 的一维结果，而你的 feature 操作是在二维 FPN 上完成的。

#### 为什么会起作用

没有 RTP，你没法在 feature map 上精确找到 target-assigned 区域。

---

### 4.3 assignment-localized 干预区域

#### 做什么

用：

[
M_{AL}=M_{topology}\odot M_{assign}
]

代码里直接是：

* `M_topology = F.adaptive_avg_pool2d(inner_t, ...)`
* `M_AL = M_topology * M_assign_2d`

#### 为什么好

这比 topology-only 或 bbox-only 更精确。

#### 为什么会起作用

因为最终参与干预的位置必须同时满足：

* 物理上属于目标
* detector 内部真实被分配给 target 类

这就把攻击缩成了“微创式干预”。

---

## 模块 5：局部背景/上下文 confounder prototype

这是你现在方法从 strict AL-SVC 升级到 ALCE 的核心。

### 做什么

你当前已经不走 memory bank，也不把 target 拉向某个具体非目标类。
而是做**强化版 A**：

1. `build_all_objects_mask(...)`
2. `build_local_context_mask(...)`
3. `build_confounder_mask(...)`

然后得到：

* `M_local_ctx`
* `M_conf`

再在 clean feature 上提 confounder prototype：

[
c_{conf}
========

\frac{\sum (Z_{clean}\odot M_{conf})}{\sum M_{conf}}
]

代码中就是：

* `c_conf, valid_conf = masked_prototype(Z_clean, M_conf, ...)`

### 为什么好

它不是把 person 拉向 dog/car 这种具体类别，而是拉向：

> 当前图像中，与该 target 共现的**局部背景/上下文 shortcut 原型**

### 为什么会起作用

这更像 UE 的机制：
不是类别级伪装，而是诱导 detector 学到 **local co-occurrence shortcut**。

---

## 模块 6：ALCE 主损失

这一步是你当前代码里已经替换完成的主项。

### 做什么

当前代码里已经在每层上提取三个 prototype：

* `z_t_adv`：adv target prototype
* `mu_t_clean`：clean target prototype
* `c_conf`：clean confounder prototype

然后计算：

* `compute_entangle_loss(...)`
* `compute_anchor_losses(...)`
* `compute_collapse_loss(...)`

并分别累加到：

* `L_entangle_bg`
* `L_semantic_anchor`
* `L_collapse_aux`

### 总损失

现在你的训练主线已经是：

[
L_{total}
=========

\lambda_{ent}L_{entangle}^{bg}
+
\lambda_{anchor}(L_{cos}+L_{energy})
+
\lambda_{flat}L_{collapse}
+
\lambda_{preserve}L_{preserve}
+
\lambda_{tv}L_{tv}
+
\lambda_{budget}L_{budget}
]

代码里对应：

* `self.lambda_ent * L_ent_final`
* `self.lambda_anchor * L_anchor_final`
* `self.lambda_flat * L_flat_final`
* `+ preserve + tv + budget`

### 为什么好

这和旧版 strict AL-SVC 最大的区别是：

* 以前是 `L_collapse` 主导
* 现在是 `L_entangle_bg` 主导
* `L_collapse` 只做辅助平坦化

### 为什么会起作用

因为现在不是简单把 target feature 压坏，而是：

> 把 target-assigned representation 往局部背景 shortcut 拉，同时保住 target 的语义活性。

这更像真正的不可学习样本，而不是 suppression。

---

## 模块 7：语义锚定 + 非目标保护

### 做什么

你现在保留了三类辅助约束：

#### 1. `L_cos`

保持 adv target prototype 和 clean target prototype 的语义方向一致

#### 2. `L_energy`

保持 target prototype 的能量，不让它变成 dead feature

#### 3. `L_preserve`

保护 non-target 浅层 feature 和 non-target logits

### 为什么好

如果只做 entangle 或 collapse，target feature 很可能直接死掉，non-target 也容易被拖下去。

### 为什么会起作用

这些约束让你的方法变成：

* target 语义还活着
* target 空间几何被 shortcut 化
* non-target 尽量稳住

---

# 3. 每个模块对应哪些代码

下面是你组会最适合展示的“代码对应表”。

## 物理拓扑

* `_build_support(...)`
* `build_support_mask(...)`

## 通用扰动

* `_build_global_freq_pattern(...)`
* `_compose_delta_batched(...)`

## strict TAL 截获

* `self.hijacked.get_assigned_targets_and_loss(...)`
* `self.hijacked.last_real_assign`
* `strict_gate_1d = real_fg & (real_labels == self.target_class_id)`

## ACGT

* `renorm_yolo_bbox_after_padding(...)`
* `project_strict_gate_to_fpn(...)`

## `M_AL`

* `M_topology = F.adaptive_avg_pool2d(...)`
* `M_AL = M_topology * M_assign_2d`

## Confounder 原型

* `build_all_objects_mask(...)`
* `build_local_context_mask(...)`
* `build_confounder_mask(...)`
* `masked_prototype(Z_clean, M_conf, ...)`

## ALCE 主项

* `compute_entangle_loss(...)`

## 语义锚定

* `compute_anchor_losses(...)`

## 辅助平坦化

* `compute_collapse_loss(...)`

## 非目标保护

* strict 分支后面的 `L_preserve` 计算

## 训练目标

* `total_loss = self.lambda_ent * ... + ...`

---

# 4. 为什么你的方法比其他方法更好

这是你汇报时最重要的一段。

## 1. 它不是简单全图加噪

很多方法只是在输入空间制造困难样本。
你现在不是全图乱打，而是：

* 先找 target 的物理拓扑
* 再找 detector 内部真实负责 target 的单元
* 最后只在交集 `M_AL` 上操作

所以你的方法更精准。

---

## 2. 它不是简单压制 detector

如果只做 collapse，审稿人会说你只是让 detector 变差。
但你现在用的是：

* 背景/上下文 confounder prototype
* semantic anchor
* preserve

所以你讲的是：

> **诱导 detector 在 target 类上学到一条错误 shortcut**

而不是简单 feature erasure。

---

## 3. 它比拉向具体非目标类更干净

把 person 拉向 dog/car prototype 会像类别混淆。
你现在拉向的是**局部背景/上下文原型**，不是具体类。

所以更像：

* context bias
* local co-occurrence shortcut
* UE 风格的 shortcut induction

---

## 4. 它比 memory bank 更可控

你们当前不用 memory bank，是正确的。
因为图内局部背景原型：

* 不会跨图漂移
* 不会越积越脏
* 更容易控制 non-target 污染

所以更适合作为第一版论文主线。

---

# 5. 你现在实际是怎么操作的

这个部分你可以直接对导师说“实验实际是这样跑的”。

## 操作步骤

1. 从训练集中筛出包含 target 类的图像
2. 对每张图构造 `inner_t / ring_t`
3. batch 内做 padding，并同步做 bbox renorm
4. 用 Fourier carrier 生成 `adv`
5. clean 图走 detector，拿到 strict assignment
6. strict gate 做 RTP，得到 `M_assign_2d`
7. 用 `M_topology * M_assign_2d` 得到 `M_AL`
8. 构造局部背景/上下文 `M_conf`
9. 提取：

   * `z_t_adv`
   * `mu_t_clean`
   * `c_conf`
10. 优化：

* `L_entangle_bg`
* `L_cos + L_energy`
* `L_collapse`
* `L_preserve`

11. 更新 Fourier 系数，训练出 universal poison
12. 用训练好的 poison 去生成 poisoned dataset，再训练 victim detector

---

# 6. 最后一句可以怎么总结

你可以这样收尾：

> 我们现在的方法不是简单在目标上加噪，而是通过 ALSD 框架，在 ACGT 保证的 assignment-consistent 几何闭环下，只对真实 target-assigned detector units 施加 ALCE 式的局部背景/上下文纠缠。这样模型在 target 类上学到的不是可泛化的目标几何，而是一条局部 shortcut，因此它仍可能保留一定语义响应，但会逐步失去定位能力；同时由于我们显式约束 non-target，整体方法比传统粗暴 poisoning 更微创、更 detector-specific，也更符合不可学习样本的机制本质。

