# STATE candidate — TAUSB-MALC-GRAD-GEOMETRY-S0

## 建议

- **不修改 Current Best**：本实验没有 fresh-victim、AP50 或数据集 materialization 证据。
- 将本实验登记为失败/无效的非主线机制审计，避免以后把 raw 冲突信号误写成正式 first boundary。
- 在用户批准前不直接改写 `research_workspace/STATE.md`。

## 可供用户批准的 STATE 条目

> TAUSB-MALC gradient-geometry seed0 probe 完成 16/24/8 的 surrogate-only 诊断，但三个尺度的 prototype coverage 均低于预注册 0.80，导致 effective primary metric 缺失；结果为 `invalid / first_bad_boundary=null`。Raw gradient 中存在 cross-batch Q25 和 MALC–RMS 冲突线索，CGR 对 MALC 的 median retention 约 0.981，但这些信号在 validity 修复前不可用于选择方法模块。该分支继续禁止 victim 训练。

证据：[H→E→N](result-TAUSB-MALC-GRAD-GEOMETRY-S0.md)；[diagnostic summary](diagnostic_summary.md)；[decision artifact](../remote_artifacts/geometry/diagnostic_decision.json)。

## 唯一后续候选

`prototype coverage measurement audit`：只审计并修复 prototype pooling coverage 的测量口径，然后原样重跑 read-only geometry probe；不改变方法参数，不启动 victim。
