# L6 End-to-End Local Smoke 报告

状态：**local engineering chain pass**。证据：`artifacts/oa_lgc/local/20260713_222449_255485_L6_seed0/`。

## 已完成链路

真实 mini VOC image/YOLO annotations → shared `delta_obj` → object-aligned poison/non-target exclusion → disjoint support/query → clean/poison functional virtual trajectories → target gain/per-class authorized gain → core objective → backward/update/project → checkpoint/metrics。

## 主 J=3 结果

- 2 episodes；每个 4 support / 4 query；最大 ID overlap=0。
- forward/backward 完成；delta change norm=1.004614。
- final mean/max abs delta=0.017453/0.024954 < 0.0627451；saturation ratio=0。
- base model unchanged=true；checkpoint restored equal=true。
- target gain 2/2 可计算；invalid ratio=0。
- authorized valid classes：episode 0 为 class 6；episode 1 为 classes 15、17；总数 3。
- target gain ratio：1.000007、0.998028。此数值只验证可计算性，不表示 target learning 已被抑制。
- target assignment coverage proxy：0.6667、0.7586；未覆盖实例是 overlap 后 valid support 太小而按协议跳过。
- non-target logits/assignment/box drift proxy：该 run 为 0；只反映 object-crop proxy 的直接 non-target 区域排除，不外推到 YOLO。
- peak Python traced memory：5,441,308 bytes；主链 runtime 0.583 s（CPU proxy）。

## J 与复现 Gate

- J=1 单 episode：pass。
- J=3 两 episodes：pass。
- J=5 单 episode：pass。
- 同 seed J=3 全链独立复跑：IDs、loss rows 与 final delta 完全一致。
- artifact required files：完整。
- 全套测试：`91 passed in 5.83s`，0 failed，0 skipped。

## 结论边界

这只证明 engineering chain pass。未使用真实 YOLO TAL/DFL；`target_dfl_available=false`，值 0 是 unavailable 占位而非真实 DFL=0。未训练 victim、未计算 mAP，不能声称 person AP 会下降、non-target AP 会保持或 learning gain 能预测完整 victim。

