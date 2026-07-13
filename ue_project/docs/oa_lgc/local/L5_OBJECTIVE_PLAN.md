# L5 OA-LGC Core Objective 计划

组装 `L_protect + lambda_carrier L_carrier + lambda_auth L_auth + lambda_reg L_delta`，所有 lambda/eps/clip/lr 写入 YAML。只由 optimizer 更新 `delta_obj`，每步检查 finite、裁剪梯度并投影到 L∞ budget。

Gate：组件与日志 schema、delta 更新、base model frozen、空 authorized、invalid target protect 跳过、budget、checkpoint roundtrip 和 no-overwrite 全部通过。

