# L7 安全清理计划

本阶段先审计、后决定。禁止目录级或批量删除；任何删除必须是一个明确文件并有引用/测试证据。

| 文件 | tracked/untracked | 当前引用 | 历史用途 | 是否核心 | 删除理由 | 验证命令 |
| --- | --- | --- | --- | --- | --- | --- |
| 本分支临时 debug 文件 | 未发现 | n/a | n/a | n/a | 无候选，不删除 | `git status --short`; `rg --files oa_lgc configs/oa_lgc docs/oa_lgc tests` |
| `dcss/`、`configs/dcss/`、`docs/dcss/`、`scripts/dcss_*` | tracked 与历史 untracked 混合 | 多处 | Stage 0/1/1.5 | 是/受保护 | retained due to uncertain dependency；禁止删除 | `git grep dcss`; `rg -n dcss`；历史 tests |
| 6 个 dirty `ue_framework` 文件 | tracked dirty | 正式 CLI/train/eval | 用户历史工作 | 是/受保护 | 明确禁止删除或修改 | `git status --short` |
| `artifacts/oa_lgc/local/*` | untracked local artifact | 新报告引用 | OA-LGC smoke evidence | 证据 | 保留；不提交 checkpoint/PNG | artifact schema check |
| `.venv/`、`runs/`、`checkpoints/`、数据 | untracked/ignored/tracked 混合 | 运行环境与历史实验 | 历史资源 | 是/受保护 | retained due to uncertain dependency | 只读存在性审计 |

预期结论：files removed=0。若引用审计未发现孤立的新文件，也不创建 cleanup-only 删除提交；L7 提交只包含计划、报告和最终汇总。

