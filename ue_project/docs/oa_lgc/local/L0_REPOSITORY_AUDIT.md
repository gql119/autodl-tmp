# L0 仓库审计

日期：2026-07-13  
起始分支：`codex/dcss-stage0-stage1`  
实施分支：`codex/oa-lgc-local-chain`  
起始 commit：`b72672a1505a6ea76acbbedca4f404b38ab4b021`

## 工作区与 Git

- origin：`https://github.com/gql119/autodl-tmp.git`（fetch/push 均已配置）。
- 起始 dirty tracked 文件共 6 个：`ue_framework/launch_one.py`、`paths.py`、`runtime.py`、`stages/aggregate.py`、`evaluate.py`、`train_victim.py`。
- 起始 untracked 内容包括 `.venv/`、`AGENTS.md`、`artifacts/`、`configs/dcss/`、部分 `docs/dcss/`、`figures/`、`runs/`、脚本和本地资料；均视为历史/用户工作，不纳入无关修改。
- 基线测试：`39 passed in 4.80s`，命令为外部只读解释器运行 `tests/test_dcss.py tests/test_dcss_stage15.py`。

## 环境、数据与 checkpoint

- 当前副本 `.venv`：Python 3.12.13，但缺 PyTorch 与 Ultralytics。
- 可用同机只读解释器：`F:/autodl-tmp/ue_project/.venv/Scripts/python.exe`。
- PyTorch：`2.11.0+cu128`；CUDA runtime：12.8；CUDA available：true；GPU：NVIDIA GeForce RTX 2070。
- Ultralytics：`8.4.90`。
- mini VOC：`F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset`；800 train / 200 val，YOLO labels 数量一致。
- 当前副本 checkpoint：`F:/autodl-tmp - 副本/ue_project/checkpoints/voc20_surrogate.pt`。
- 历史审计 SHA256：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`。
- checkpoint 训练 split metadata 不足；本任务仅把它视为 frozen engineering surrogate，不作独立性或方法有效性结论。

## 1. 当前 YOLOv8 结构

`configs/voc_yolov8n_20cls.yaml` 定义 20 类 YOLOv8n 风格模型。Detect 接收模型层 15/18/21，对应 P3/P4/P5；历史实际张量为 `[B,64,80,80]`、`[B,128,40,40]`、`[B,256,20,20]`（640 输入）。目标类为 person/id 14。

## 2. 当前 FPN hook

`dcss/feature_hooks.py::FeatureHookBank` 提供按模块名注册和清理 hook。`TAUSBUniversalTrainer` 也维护 `shape_layers = [model.15, model.18, model.21]` 和多层 feature cache。两者可复用作未来 YOLO adapter，不应复制第三套 hook。

## 3. 当前 TAL unit selection

`ue_framework/ultra/hijacked_loss.py::HijackedV8Loss` 暴露真实 `fg_mask`、`target_scores`、`target_labels`、`target_gt_idx`。`dcss/unit_partition.py` 将 target positives 经 PAG 选择并逐类划分 non-target positives，包含 TAL/FPN anchor 数一致性检查。

## 4. 当前 poison materialization

`ue_framework/methods/tausb_universal.py` 和 `dcss/stage1.py::materialize_dataset` 已能读取 VOC mini、写 poisoned images/labels/manifest。它们属于 TAUSB/DCSS 历史协议；OA-LGC smoke 不调用会覆盖旧目录的入口，而是在独立 run id 内按 episode 在线构造 clean/poison tensor。

## 5. 当前 object-aligned carrier

`dcss/stage15.py::object_aligned_warp` 已实现标准对象纹理插值和 non-target box dilation 排除，且历史测试覆盖基础形状与 overlap=0。缺口包括：插值配置、soft edge、实例跳过原因、真实 overlap/area 分解、边界与小目标诊断、artifact schema、明确的唯一参数和 checkpoint 接口。因此复用数学思路，在独立 `oa_lgc/` 命名空间做最小完整实现，不改历史函数。

## 6. 当前 functional virtual update

未发现 `torch.func.functional_call`、stateless parameter copy 或可靠可微 inner-loop。Stage 1 是直接优化 universal carrier，不是 clean/poison 双轨虚拟训练。OA-LGC 必须新增。

## 7. 当前 clean/poison 双分支

现有代码有 clean/adv forward 对照，但没有 `S_c/S_p/Q_c/Q_p` 四集合，也没有原始 image ID 的 support/query 互斥保证。需要新增显式 episode 数据结构与 sampler。

## 8. 当前 per-class metrics

DCSS 可逐类记录 feature leakage/logit drift，统一 evaluation 可提取 per-class AP；尚无逐类 `G_k^c/G_k^p`、有效性门槛和 invalid reason。需要新增 learning-gain schema，缺失类不得补零平均。

## 9. 当前 evaluation 入口

正式入口为 `ue_framework/launch_one.py --stage train_victim/evaluate`，aggregate 独立执行。本地 OA-LGC smoke 明确不训练 victim、不计算 mAP；只输出 engineering metrics。

## 10. 当前测试入口

历史测试为 `tests/test_dcss.py` 与 `tests/test_dcss_stage15.py`，基线 39/39 通过。OA-LGC 将增加 carrier、episode、virtual update、gain、objective、smoke 六个测试文件，并与历史测试一起运行。

## 11. 可复用模块

- VOC 图像/标签读取：`ue_framework/data_utils.py`。
- object-aligned 几何参考：`dcss/stage15.py`。
- FPN/TAL/PAG：`FeatureHookBank`、`HijackedV8Loss`、`partition_tal_units`。
- 环境和历史 provenance 文档。
- 当前 mini VOC 与 VOC20 checkpoint。

## 12. 需要新增或重构的模块

- 独立 `oa_lgc/` 包：carrier、episode、functional virtual update、gain、objective、artifact/smoke CLI。
- 本地轻量 detector proxy adapter，用真实 mini VOC 对象 crop 验证数学与梯度链；future YOLO functional adapter 保留清晰接口但不伪称已验证。
- 独立 configs/docs/artifacts 路径和 no-overwrite 检查。

## 13. 待删除冗余模块

L0 未识别到可 100% 安全删除的文件。DCSS、RCDS、QP、Stage 0/1/1.5 均保留；它们不进入 OA-LGC 默认主链路。L7 默认输出“无删除”，除非届时存在本分支新建且明确无引用的临时文件。

## 14. 核心文件保护列表

详见 `docs/oa_lgc/local/CORE_FILE_PROTECTION_LIST.md`。该列表在任何清理前已建立。

## 文档审计总结

Stage 0 证明 P3/rank8 子空间工程 Gate，但不证明 victim 效果；Stage 1 和修复筛选因 non-target leakage/迁移失败而停止；Stage 1.5 证明 object-aligned carrier 能排除直接 non-target overlap，但相对 random 的选择性仍未通过，且 clean mini victim 欠拟合。OA-LGC 因而只验证全新 learning-gain 工程链，不复用或重新包装 DCSS 方法结论，不启用 RCDS/QP/ALCE。

