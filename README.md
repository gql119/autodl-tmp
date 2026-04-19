## 最终修改方向

### 1. RLCP：采用

**现有 mid-range annulus 保留 + 非目标 core 剔除 + Trimmed Mean 原型**

不做：

* 多环带可靠性加权
* Medoid
* 全图 ASDC 主线替换

### 2. PAG：采用

**Hard Percentile，前 30% 硬截断**

不做：

* soft gate
* sigmoid temperature 权重化

### 3. DSNP-lite：采用

**现有 Feature MSE 托底 + Non-target Logit Margin 保护**

不做：

* Gram 矩阵关系保护
* 回归头刚性保护
* full OSS

### 4. 这轮不动

* 动态频带
* D-FAS
* full OSS
* 生成阶段加噪逻辑

---

# 为什么这么定

## 一句话判断

这三项改动都是**沿着你当前已经跑通的 ALCE 主闭环做“局部强化”**，不会破坏现有架构：

* `inner_t / ring_t` 继续做输入空间局部化
* hijacked TAL 继续给真实 assignment
* `M_AL` 继续做精准干预
* `L_entangle_bg` 继续做主项
* 只是把

  * `c_conf` 变稳
  * `strict_gate` 变精
  * `L_preserve` 变强

这和目标检测里已知的 context bias、背景误分、prototype contamination 问题是吻合的【Dreyer et al., 2023; Ong et al., 2025】。
*Dreyer, M., Achtibat, R., Wiegand, T., & Samek, W. (2023). Revealing Hidden Context Bias in Segmentation and Object Detection Through Concept-Specific Explanations. CVPRW. [https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Dreyer_Revealing_Hidden_Context_Bias_in_Segmentation_and_Object_Detection_Through_CVPRW_2023_paper.html](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Dreyer_Revealing_Hidden_Context_Bias_in_Segmentation_and_Object_Detection_Through_CVPRW_2023_paper.html)*
*Ong, D. S., Liu, Y., Shang, C., & Ding, G. et al. (2025). Fixing Background Misclassification in Few-Shot Object Detection via Product of Experts. IEEE Transactions on Circuits and Systems for Video Technology. IEEE Xplore abstract result.*

---

# 这轮就完成的内容

## 第一轮：只改 3 个点

### A. RLCP

最终实现定义：

> **中距离上下文环带 + 非目标 core 剔除 + trimmed mean confounder prototype**

你代码里对应两处：

1. **构造 `conf_np` 的地方改**

   * 保留 `mid-range annulus`
   * `all_objects` 改成 **non-target core mask**
   * core 收缩系数先固定 **0.8**

2. **提取 `c_conf` 的地方改**

   * `masked_prototype(...)` 改成 **trimmed mean**
   * 先固定 trim ratio = **10%**
   * 只对 `c_conf` 用 trimmed mean，`z_t_adv / mu_t_clean` 先不动

### 这一步的最终目的

解决 crowded scene 下：

* `M_conf` 太小
* `c_conf` 不稳
* `L_entangle_bg` 退化

---

### B. PAG

最终实现定义：

> **在 strict gate 内部，只保留 target-assigned units 中 target score 前 30% 的高能单元**

你代码里对应一处：

1. **`strict_gate_1d` 之后**

   * 读取 `target_scores[:, :, target_class_id]`
   * 在 `strict_gate_1d == True` 的位置里算分位数
   * 只保留前 30%
   * 再送去 `project_strict_gate_to_fpn(...)`

### 这一步的最终目的

解决：

* 硬二值门控太“平均”
* 低质量 target units 稀释梯度
* 攻击预算浪费在边缘锚点上

---

### C. DSNP-lite

最终实现定义：

> **现有 non-target Feature MSE 保留 + 新增 non-target logit margin preserve**

你代码里对应一处：

1. **preserve 分支**

   * 保留 `L_preserve_feat`
   * 保留 `L_preserve_logits`
   * 新增 `L_margin`
   * 最后合成新的 `L_preserve`

推荐第一版直接写成：

[
L_{preserve}
============

L_{feat}
+
L_{logit}
+
\lambda_{margin}L_{margin}
]

其中 `L_margin` 用 non-target classes 的：

[
(top1-top2)^{adv} \approx (top1-top2)^{clean}
]

### 这一步的最终目的

让 non-target 保护从“值不变”升级成“判别边界不变”。

---

## 第二轮：补日志

这轮我们也一起定掉，但只补**必要日志**，不扩张。

### RLCP 日志

加：

* `rlcp_conf_mass`
* `rlcp_trim_keep_ratio`
* `rlcp_core_exclusion_ratio`
* `confounder_purity_ratio`

### PAG 日志

加：

* `pag_positive_ratio`
* `pag_threshold`
* `pag_mean_target_score`

### DSNP-lite 日志

加：

* `L_margin`
* `margin_clean_mean`
* `margin_adv_mean`

---

# 最终不做的内容

这轮明确不做：

* 多环带可靠性加权
* soft gate
* Gram preserve
* regression rigid preserve
* 动态频带
* full ASDC
* full OSS
* D-FAS

这些都留到后面，不进这轮主线。

---

# 代码修改落点，最终定版

## 必改文件

* `ue_framework/methods/tausb_universal.py`

## 同步扩展

* `ue_framework/methods/alce_acgt.py`
* `ue_framework/methods/alce_losses.py`
* `ue_framework/methods/alce_metrics.py`

## 这轮不改

* `ue_framework/ultra/hijacked_loss.py`
* `_compose_delta_batched()`
* `TAUSBMaskGenerator.generate()`

---

# 我给你的最终执行顺序

## 先改

1. `conf_np / c_conf` → RLCP
2. `strict_gate_1d` → PAG
3. `L_preserve` → DSNP-lite

## 再补

4. RLCP / PAG / DSNP-lite 的必要日志

## 然后直接跑

做第一轮 sanity + 短程训练，不再加第四个模块。

---

# 最后的最终版本

你这轮方法就叫：

> **ALCE + RLCP + PAG + DSNP-lite**

对应含义：

* **ALCE**：主攻击框架不变
* **RLCP**：稳健局部上下文原型
* **PAG**：百分位感知门控
* **DSNP-lite**：轻量结构化非目标保护

