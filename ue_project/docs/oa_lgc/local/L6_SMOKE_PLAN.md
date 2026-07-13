# L6 End-to-End Local Smoke 计划

真实 mini VOC 链路：clean image/YOLO objects → shared object-space delta → strict disjoint episode → clean/poison functional virtual update → target/per-class gain → core objective → delta update → checkpoint/artifact。

主配置 J=3、2 episodes、head_only、first_order local approximation；另运行 J=1/J=5 单 episode。主 J=3 用相同 seed 独立复跑，比较 IDs、loss rows 与 final delta。

本地 adapter 为 object-crop detector proxy。记录 class/box/logit/assignment drift；DFL 不存在，字段值为 0 且 `target_dfl_available=0`，不得解释成真实 YOLO DFL 为零。

