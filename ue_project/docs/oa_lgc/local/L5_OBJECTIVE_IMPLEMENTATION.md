# L5 OA-LGC Core Objective 实现

`oa_lgc/objective.py` 定义 core objective、固定日志字段、`update_delta`、L∞ projection、delta metrics 与 checkpoint save/load。默认正则仅为 mean squared delta；没有加入 RCDS、subspace、QP、assignment/box/DFL preservation loss。

checkpoint 保存 delta、eps、shape 和 metadata；目标已存在时明确报错。加载时检查 schema、shape 和 eps budget。

