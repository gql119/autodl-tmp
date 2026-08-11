# STATE（当前真相）

更新时间：2026-08-11

本文件只记录当前仍成立的结论与索引。完整过程和原始数字留在 artifacts、
实验台账和单次分析中。

## 当前主线

| Key | Current Truth | Evidence |
|---|---|---|
| 研究任务 | Pascal VOC / YOLOv8n 下，选择性压低 `person` 检测，同时保持其他 19 类性能 | `ue_project/AGENTS.md` |
| 当前活跃方法方向 | 经人工内容、低频占比、VOC20 低响应和数据去重联合筛选的固定 secret image + 以每个 person GT bbox 为宿主/嵌入区域的 sample-adaptive deep hiding + 独立 D-LFC + 检测实例 CICR + 目标攻击梯度对逐类非目标梯度正交 + 显式 clean/poison 非目标 assigned-logit 对齐 | 用户 2026-08-10 方向修订；`docs/research/specs/TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3.md` |
| 当前 hiding 门禁 | r2（高频缩放 1.0）可恢复 secret，但高频能量超标；SB25（高频缩放 0.25）把高频能量降至 `0.034223`，却使 retrieval 降至 `0.424479`、L1 identity margin 降至 `0.067159`。两者均失败，mechanism/victim/AP50 继续阻断 | `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25/analysis/result-TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25.md` |
| 历史单-seed AP50 参照（非当前主方法） | 旧 `tausb_mask`：`mAP50_target=0.0877587`，`mAP50_non_target=0.7173273`，`mAP50_all=0.6858489`；仅 seed 0，标记 tentative，用户已放弃作为当前方案 | `ue_project/runs/artifacts/tausb_mask/steps40/seed0/metrics/metrics.json` |
| clean evaluation | clean VOC validation；鲁棒性变换不属于主 clean protocol | `ue_project/AGENTS.md` |
| OA-CLR-CCD v2 | mechanical PASS，但 held-out gain/dependency/specificity/joint 均未通过；暂停 fresh-victim 推进 | `ue_project/artifacts/oa_clr_ccd_mechanism_v2/report.md` |

## 活跃假设

| Priority | Hypothesis | Status | Next Evidence |
|---|---|---|---|
| P0 | 当前选定的固定语义 secret 能否在 `eps=16/255`、person bbox 宿主下同时保持可辨识恢复、非固定像素纹理和受控频谱 | `scale=1.0` 与 `scale=0.25` 均已失败；前者高频超标，后者 secret identity 恢复失败 | 如继续，仅以新批准 Spec 做一次其他条件完全匹配的 `hf_subband_scale=0.50` hiding-only 判别实验 |
| P0 | D-LFC 集中扰动隐特征与 CICR 对齐 clean-to-poison person 检测残差能否共同形成稳定捷径 | 待批准/验证 | matched T0/T1 surrogate mechanism gate；不以 mechanical smoke 替代 victim AP50 |
| P0 | 显式逐类 non-target logit 对齐在正交 target attack 之上能否降低 collateral drift，且不抹除 CICR/pattern 更新 | 待批准/验证 | matched P0/P1 protection gate；逐类 logit/probability、attack retention、CICR、D-pattern |

## 已废弃或非主线

- true instance mask 作为当前 best 叙述：与现有证据不符。
- cooccur-specific protect、class-aware preserve、FDACB、DES-R 作为当前 best
  有效组件：现有主配置/证据不支持。
- 将 OA-CLR-CCD mechanism v2 mechanical PASS 表述为最终 UE 有效：禁止。
- 旧 `tausb_mask`/ALCE/PAG/late-repair 作为当前要继续优化的方法：用户已明确
  放弃；仅保留历史证据。
- 固定 Fourier SIRC bank + 48 维共享系数作为当前 carrier：与用户重申的
  sample-adaptive deep hiding 载体定义不符。

## 下一步优先级

- **P0**：如继续 carrier 调优，先单独冻结并批准
  `hf_subband_scale=0.50` 的 matched hiding-only Spec；不得直接启动 GPU。
- **P0**：只有新的 hiding hard gate 全部通过后，才允许对
  D-LFC/CICR/NLA/CGR 做独立 mechanism pre-run review。
- **P0**：mechanism 未通过前继续禁止 fresh-victim C0/P1-V 与 AP50 训练。

## 声明纪律

- 没有真实训练/评估 artifacts，不声称实验完成或指标提升。
- 单 seed 高点只能标 `tentative`，不能直接成为多 seed Current Best。
- checkpoint 重复评估不是 seed audit。
- 失败实验必须入账，防止下一轮重复走已证伪路线。
- `HIDING-S0-SB25-R1` 只提供单 seed hiding mechanics 证据；不得外推为不可学习、
  非目标 AP50 保持、视觉自然性、鲁棒性或迁移性证据。
