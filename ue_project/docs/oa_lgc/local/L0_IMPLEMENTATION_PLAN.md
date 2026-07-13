# L0 实施计划

## 边界与假设

1. 本地 Gate 是 engineering chain，不是方法有效性 Gate。
2. mini VOC smoke 使用真实本地图像和 YOLO 标注；为避免 RTX 2070 上对完整 YOLO 做高阶 functional meta-update，首个可验证 adapter 是轻量 object-crop detector proxy。
3. `first_order: true` 将明确记为局部/截断近似；为保持 poison inner update 到 `delta_obj` 的梯度，仍保留必要的 mixed derivative，不称为完整二阶 MAML。
4. YOLO TAL/FPN 现有接口只用于审计与诊断兼容设计；完整 YOLO virtual adapter 属于云端后续，不在本地 smoke 中伪造 pass。

## 阶段与成功标准

1. L1：新增 carrier 与单元测试；验证 overlap 排除、soft mask、边界、小目标、插值、面积和仅 delta 梯度。
2. L2：新增 episode sampler；所有 clean/poison 配对保留 source ID，support/query 交集为 0，数据不足明确报错。
3. L3：以 `torch.func.functional_call` 实现参数副本的 J=1/3/5 clean/poison 轨迹；base state hash 不变，outer gradient 到 delta。
4. L4：实现 target ratio、invalid clean gain、carrier query 与逐类 authorized gain；synthetic 正反例验证符号、缺失类排除和梯度。
5. L5：组合 `L_core`、梯度裁剪、L∞ projection、checkpoint roundtrip 和完整 loss schema。
6. L6：用 mini VOC 唯一 run id 执行 J=1/3/5（J=3 为主要 2–5 outer steps），写完要求 artifact；结论只写 engineering chain。
7. L7：逐文件引用审计；默认不删除；运行历史+新增全套测试并生成最终报告。

## 实现结构

- `oa_lgc/carrier.py`
- `oa_lgc/episodes.py`
- `oa_lgc/model.py`
- `oa_lgc/virtual_update.py`
- `oa_lgc/gains.py`
- `oa_lgc/objective.py`
- `oa_lgc/artifacts.py`
- `oa_lgc/smoke.py`
- `oa_lgc/cli.py`
- `configs/oa_lgc/local/smoke.yaml`
- `tests/test_oa_lgc_*.py`

## 验证命令

所有命令显式使用 `F:/autodl-tmp/ue_project/.venv/Scripts/python.exe`。每阶段先运行对应 OA-LGC tests，再运行历史 DCSS tests；提交前运行 `git diff --check`。每个 artifact 目录创建前必须不存在。

## 不做事项

不实现或启用 RCDS、广义特征子空间、PCA/random subspace、QP、PCGrad、ALCE、背景 collision、feature push/pull、ensemble、完整 victim、Stage 2 或防御实验。

