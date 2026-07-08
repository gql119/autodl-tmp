# Minimal Victim Report

## 1. Git 与代码版本

- HEAD: `aacab4d2b42cb27c237c87e39dcfcfb363c58807`
- tag: `det-crad-p1-p2-local-validation-pass`
- branch: `codex/det-crad-p1-p2`
- working tree status: `M docs/local_validation_report.md
 M tests/run_tests.py
 M ue_framework/launch_one.py
 M ue_framework/methods/learning_trajectory/method.py
 M ue_framework/paths.py
 M ue_framework/runtime.py
 M ue_framework/stages/aggregate.py
 M ue_framework/stages/evaluate.py
 M ue_framework/stages/train_victim.py
?? .venv/
?? AGENTS.md
?? ImageShortcutSqueezing-master/
?? "Towards_Effective_and_Robust_Unlearnable_Examples_Against_Object_Detection (1).pdf"
?? configs/subsets/
?? docs/local_validation_report_final.md
?? docs/minimal_victim_report.md
?? docs/virtual_lr_sweep_report.md
?? figures/
?? outputs/
?? p1_p2_before_local_validation.patch
?? p1_p2_untracked_before_local_validation.txt
?? runners/run_minimal_victim_experiment.py
?? scripts/
?? tests/test_p2_inner_full_loss.py
?? ue_framework/datasets/
?? ue_framework/tools/
?? ue_framework/utils/robust_transforms.py
?? "\345\210\235\347\250\277/"`
- key file hashes: `{'method_py': 'b91514191528f7f50e75aaca660e970b6a93ed2561ee8015826b94d909d5b039', 'adapter_py': 'e641cd09ac370902dc4355dbf8078a3182c1034e30e19acf43f6b33abad75acc', 'minimal_runner': 'df6da97226dba0b1044234834999f8a68317d237000a5055b2124bf9dbd695ba', 'train_config': '939f6c94f924b8551578640c17692a01b53bb1061e3de4347b54650649cfa621'}`

## 2. P2 full-loss 审计

- support loss 来源: `YOLOv8TALAdapter.compute_detection_loss(... class_filter=None)` -> `ultralytics.utils.loss.v8DetectionLoss`
- support cls/box/dfl components: logged as `support_full_cls_loss`, `support_full_box_loss`, `support_full_dfl_loss`
- 是否为完整 Ultralytics loss: yes, class_filter=None path uses full BCE over all classes plus box and DFL.
- 外层类别条件 loss: `compute_class_conditioned_detection_loss`
- victim loss: Ultralytics `YOLO.train()` default detection loss
- 新增测试: `......                                                                   [100%]
6 passed in 8.05s`

## 3. Virtual LR sweep

- chosen LR: `1e-05`
- LR 1e-05: protected_gap_mean=-1.6609827677408854e-06, authorized_gap_mean=1.4050801595052083e-05, selectivity_mean=-1.571178436279297e-05, clean_delta_mean=-6.642341613769532e-05
- LR 3e-05: protected_gap_mean=-6.079673767089844e-06, authorized_gap_mean=4.19775644938151e-05, selectivity_mean=-4.805723826090495e-05, clean_delta_mean=-0.00019749005635579428
- LR 0.0001: protected_gap_mean=-2.0333131154378255e-05, authorized_gap_mean=0.00013992786407470703, selectivity_mean=-0.00016026099522908527, clean_delta_mean=-0.0006552060445149739
- LR 0.0003: protected_gap_mean=-6.376902262369791e-05, authorized_gap_mean=0.000407870610555013, selectivity_mean=-0.00047163963317871095, clean_delta_mean=-0.0020613988240559896
- LR 0.001: protected_gap_mean=-0.00043203035990397134, authorized_gap_mean=0.0013739109039306641, selectivity_mean=-0.0018059412638346355, clean_delta_mean=-0.006187661488850912
- LR 0.003: protected_gap_mean=-0.001160407066345215, authorized_gap_mean=0.0031547307968139648, selectivity_mean=-0.00431513786315918, clean_delta_mean=-0.010196900367736817

## 4. Minimal subset

- train/val: `240` / `100`
- train person-only/cooccur/authorized-only: `48` / `120` / `72`
- val person-only/cooccur/authorized-only: `20` / `50` / `30`
- manifest hash: `00ed237929bd01bebdf17e8002896dbd95c6034e7694a4e7e298c92dc69ebedf`
- scale note: `Using 240 train / 100 val / 3 epochs to keep four seed-0 victim runs feasible on local RTX 2070; all four methods share the exact same setting.`

## 5. Poisoned datasets

- cs_em_det: images=240, linf=0.0627451241016388, PSNR=28.20114010698184, LPIPS=0.3013414843939245, support_area=1.0, perturbed_area=0.992864990234375, path=F:\autodl-tmp\ue_project\outputs\minimal\poisoned_datasets\cs_em_det, label_mismatch=0, count_mismatch=False
- meta_only: images=240, linf=0.0627451241016388, PSNR=28.839664742120682, LPIPS=0.2786115601658821, support_area=1.0, perturbed_area=0.9919974009195963, path=F:\autodl-tmp\ue_project\outputs\minimal\poisoned_datasets\meta_only, label_mismatch=0, count_mismatch=False
- p1_meta: images=240, linf=0.0627451241016388, PSNR=29.406060600555406, LPIPS=0.24884191000213227, support_area=1.0, perturbed_area=0.9916146596272787, path=F:\autodl-tmp\ue_project\outputs\minimal\poisoned_datasets\p1_meta, label_mismatch=0, count_mismatch=False

## 6. Victim training

- Clean: mAP50_t=0.68987081865262, mAP50_a=0.6211019324788024, mAP50_all=0.6245403767874933, time=34.547528982162476, peak_mem=396637696.0
- CS-EM-Det: mAP50_t=0.6889096241045356, mAP50_a=0.6484522959190987, mAP50_all=0.6504751623283705, time=27.785813331604004, peak_mem=412931072.0
- Meta-only: mAP50_t=0.6946192687075462, mAP50_a=0.6507298562757381, mAP50_all=0.6529243268973286, time=26.280751943588257, peak_mem=415134720.0
- P1+Meta: mAP50_t=0.6936177389249893, mAP50_a=0.6214661994742933, mAP50_all=0.625073776446828, time=24.41699981689453, peak_mem=413168640.0

## 7. 方法比较

1. 四组能否完整运行: `yes`
2. 三组 poisoned datasets 是否正确生成: `yes`
3. Meta-only 是否优于 CS-EM-Det: `False`
4. P1+Meta 是否优于 Meta-only: `False`
5. P1 是否明显伤害 authorized classes: `False` under 95% Meta-only threshold; it still removes most of Meta-only's authorized AP gain.
6. 是否建议进入更大子集: `False`
7. 下一步: `先重新校准 virtual update`

## 8. 下一步结论

- 所有最低有效性条件均未满足。
- Virtual LR sweep 中所有 LR 的平均 meta_selectivity 均为负，说明当前单步虚拟更新目标方向没有通过小规模验证。
- P1+Meta 的 protected AP 仍高于 Clean，protected_unlearnability 为负，不能视为有效 target collapse。
- 最终结论: `先重新校准 virtual update`
