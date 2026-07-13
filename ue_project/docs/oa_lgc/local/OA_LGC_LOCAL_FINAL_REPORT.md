# OA-LGC 本地工程链路最终报告

Overall status：**local engineering chain partial pass**。detector-proxy 核心链通过；真实 YOLO functional adapter、TAL 与 DFL 未验证。

## 1. 工作区

- start branch：`codex/dcss-stage0-stage1`
- final branch：`codex/oa-lgc-local-chain`
- start commit：`b72672a1505a6ea76acbbedca4f404b38ab4b021`
- final implementation/report commit：`fdb921e`；最终元数据 checkpoint 见最终回复中的 HEAD
- historical dirty files：6 个 `ue_framework` tracked 修改及原有 untracked 目录/文件，均保留
- historical artifacts overwritten：no
- GitHub push status：L0–L7 commits 已成功 push 到 `origin/codex/oa-lgc-local-chain`

## 2. Stage 状态

| Stage | Status | Key evidence | Commit |
| --- | --- | --- | --- |
| L0 | pass | `L0_REPOSITORY_AUDIT.md`；历史 39 tests | `f1f3d0f` |
| L1 | pass | `artifacts/oa_lgc/local/20260713_220255_751416_L1_seed0/` | `37c3c24` |
| L2 | pass | `artifacts/oa_lgc/local/20260713_220613_368856_L2_seed0/` | `ecdeda1` |
| L3 | partial pass | `artifacts/oa_lgc/local/20260713_221051_396241_L3_seed0/` | `6a9f47d` |
| L4 | pass | `artifacts/oa_lgc/local/20260713_221448_428352_L4_seed0/` | `22034a3` |
| L5 | pass | `artifacts/oa_lgc/local/20260713_221939_077652_L5_seed0/` | `161e529` |
| L6 | partial pass | `artifacts/oa_lgc/local/20260713_223253_761596_L6_seed0/` | `1a54572` + final correction |
| L7 | pass | `L7_CLEANUP_REPORT.md`；91 tests | L7 finalization commit |

## 3. 工程结果

- object-aligned carrier：正确；单/多 person、soft edge、边界、小目标、插值、non-target 排除与 instance skip 通过。
- support/query：严格互斥；真实 episode overlap=0，clean/poison ID 配对一致。
- J=1：通过。
- J=3：通过（主 smoke 两 episodes）。
- J=5：通过（单 episode）。
- outer gradient to delta_obj：通过；L3 synthetic 0.006874，L6 每步非零。
- target gain：主 smoke 2/2 可计算，invalid ratio=0。
- per-class gain：主 smoke有效 classes 6、15、17。
- core objective：可更新 delta；各项独立记录。
- checkpoint：保存/恢复一致，存在时拒绝覆盖。
- artifact：L1–L6 schema 全部完整，唯一 run id。
- reproducibility：相同 seed 的 J=3 全链 IDs/loss/final delta 完全一致。
- base model：未原地修改，无模型参数梯度。

本地虚拟 detector 是 object-crop proxy。真实 YOLO TAL/DFL/full-model functional update 未验证；本地记录的 assignment/box/logit drift 为 proxy diagnostics，`target_dfl_available=false`。

## 4. 不应做出的结论

- 未证明 target AP 会下降。
- 未证明 non-target AP 会保持。
- 未证明 learning gain 可预测完整 victim。
- 未验证跨模型。
- 未验证 RCDS。
- 未验证 QP。
- 未验证恢复鲁棒性。
- 未运行完整 victim training 或 mAP evaluation。

## 5. 删除文件

- 删除文件：无。
- 删除原因：无安全候选；所有不确定项保留。
- 测试证据：91 passed；引用审计与 CLI help 通过。
- 是否影响历史工作：否。

## 6. 失败与限制

- L0：当前副本 `.venv` 缺 torch/Ultralytics；显式使用同机只读解释器，历史测试通过。environment failure，已绕过。
- L1：首轮 soft support 面积混用权重/二值口径；拆分几何面积与 weight mass 后通过。implementation/test failure，已修复。
- L5：首轮 float32 `0.2` 无容差断言失败；加入 1e-7 表示容差后通过。numerical/test failure，已修复。
- 本地资源限制：真实 YOLO mixed-derivative、TAL/DFL diagnostics 和 full_model update 未运行，需云端验证。

## 7. 云端下一步

只建议后续入口，不在本地直接运行：

1. 将 `ObjectCropDetector` adapter 替换为 YOLOv8 functional detection-head adapter，保持同一 Episode/Gain/Objective schema。
2. 验证真实 TAL assignment coverage、box/DFL loss 与 non-target assignment/box/logit drift。
3. 运行 learning-gain 与完整 victim 指标的 correlation study。
4. 在 Gate 通过后再运行完整 victim、多类别、多模型。
5. RCDS 和约束方法仅作为独立消融，不进入当前 OA-LGC core。

## 8. 复现命令

解释器：`F:/autodl-tmp/ue_project/.venv/Scripts/python.exe`。

```powershell
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli carrier --config configs\oa_lgc\local\carrier.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli episode --config configs\oa_lgc\local\episode.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli virtual --config configs\oa_lgc\local\virtual_update.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli gain --config configs\oa_lgc\local\gain.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli objective --config configs\oa_lgc\local\objective.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m oa_lgc.cli smoke --config configs\oa_lgc\local\smoke.yaml
& 'F:\autodl-tmp\ue_project\.venv\Scripts\python.exe' -m pytest tests\test_dcss.py tests\test_dcss_stage15.py tests\test_oa_lgc_carrier.py tests\test_oa_lgc_episode.py tests\test_oa_lgc_virtual_update.py tests\test_oa_lgc_gain.py tests\test_oa_lgc_objective.py tests\test_oa_lgc_smoke.py -q
```
