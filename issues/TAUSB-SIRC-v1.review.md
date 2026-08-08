# TAUSB-SIRC-v1 Pre-run Review

## PRERUN-REVIEW-01

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-PROBE-01` / `TAUSB-SIRC-MECH-S0`
- Code snapshot: branch `codex/tausb-sirc-v1`, commit
  `baaabcddd2fb76c892ee0b21c971987b50fe4560`
- Intent: 在相同预算、support、实例 renderer、频谱幅度、参数容量和优化预算下，
  比较保留共同 phase/shape 的四变体语义载体与逐频幅度匹配的 phase-scrambled
  controls；Phase B 只比较同一 I-SV 的 TRC off/on。该 probe 不生成 poisoned
  dataset、不训练 fresh victim、不产生 mAP，也不启用 LFC 或 non-target 梯度投影。
- Code location:
  - carrier：`ue_project/ue_framework/methods/semantic_residual_carrier.py`；
  - variant renderer：
    `ue_project/ue_framework/methods/instance_canonical_carrier.py`；
  - TAL/P3-P5/route adapter：
    `ue_project/ue_framework/methods/bsc_icmo_probe.py`；
  - protocol/workflow/gates：`ue_project/ue_framework/methods/sirc_probe.py`；
  - CLI/config：`ue_project/ue_framework/tools/probe_tausb_sirc.py` 与
    `ue_project/ue_framework/configs/exp_voc_person_tausb_sirc_probe.yaml`。
- Parameter data flow:
  - CLI `--config` → `load_config` → `validate_sirc_config`；
  - source manifest/local map → SHA/dimension/person-free checks → anchor/donor tensors；
  - frozen anchor phase + donor amplitude → 4 semantic variants；每个 variant 的逐频
    amplitude 生成匹配 control；
  - 4 radial × 4 modulo-pi orientation masks → 16 个共轭对称 unit-L2 bases 与固定
    signed reconstruction scales；
  - seed 2103 的共享 `16x3` coefficients 经 `1+tanh(z)` 正值调制，seed 2104 的
    256-direction pooled gamma 标定到 `0.35*eps`；
  - seed 2102 的 `hash(image_id) mod 4` → 每图唯一 variant → 同图全部 person 共用；
  - forced pseudo ellipse support → instance-canonical warp → JND/clamp；
  - frozen YOLOv8 → clean TAL `fg_mask/target_labels/target_scores/target_gt_idx` → PAG
    P3/P4/P5 → per-person residual → Instance-CICR/route/TRC/protection diagnostics；
  - arm metrics、prototype/coefficient/variant hashes、phase gates → `metrics.json` 与
    `status.json`。
- Runtime state: surrogate 参数全部 `requires_grad=False`；正式 arm 只有 carrier 的
  48 个 coefficients 可训练；prototype 仅由 calibration residual 初始化，held-out
  state 逐项精确比对；E0/E1 都执行两个相同 deterministic EOT forwards，只有 TRC
  权重分别为 0/1。
- Sink effect: local real smoke 已越过真实 VOC 加载、source screen、640 carrier bank、
  共享 gamma、YOLO 初始化、真实 TAL assignment、P3/P4/P5 residual、variant renderer、
  I-SPC-V/I-SV matched backward 和 I-SV-E1 TRC backward。三个 arm 的 coefficient
  gradient 均 finite，prototype/variant hash 已落盘，support 外最大扰动为 0。
- Baseline/disable path: 新 workflow 是独立 CLI/config；原 ICMO engine 新参数全部有
  默认值，single-pattern legacy apply 与新 single-variant path 做了 `torch.equal`
  回归；全仓 92 tests 通过。
- Local validation:
  - `py_compile`：通过；
  - formal validate-only：通过，config hash
    `4fc00518c1793c098bb160dc4827d66952187e51c720f5b7c16551fd6b1378c8`；
  - repository pytest：`92 passed`（使用独立 basetemp）；
  - real smoke：
    `ue_project/runs_research_local/TAUSB-SIRC-v1-smoke-20260808-b/`，status
    `completed`，明确 `mechanism_claim=not_evaluated_by_smoke`。
- Minimal probe:
  - source screen PASS；最高 person confidence `0.02836 < 0.05`，最高 VOC20
    confidence `0.09543 < 0.25`；与 VOC train/val 文件 hash 无重复；
  - semantic/control spectrum relative error 最大 `7.44e-10`，basis minimum rank 16，
    gamma family RMS ratio `1.03988`；
  - 静态结构：semantic pair gradient NCC `0.72798`、semantic-anchor `0.76108`、
    control-anchor `0.06283`、amplitude diversity `0.71640`；
  - smoke coverage：I-SPC-V `0.80`、I-SV `1.00`、I-SV-E1 `1.00`；E1 TRC loss
    `0.74816`，全部 finite。
- Run command binding:

  ```bash
  cd /root/autodl-tmp/ue_project
  python -u -m ue_framework.tools.probe_tausb_sirc \
    --config ue_framework/configs/exp_voc_person_tausb_sirc_probe.yaml \
    --stage all \
    --device 0
  ```

  该命令尚未绑定到可由远端 checkout 的 commit：本地 commit 已存在，但向
  `https://github.com/gql119/autodl-tmp.git` 推送因缺少用户对该具体目的地的明确授权
  被拒绝，未尝试绕过。
- Experiment validity: VOC20、person id 14、seed 0、640、shared split/hash、label hash、
  surrogate hash、source hash、semantic bank hash、epsilon、arm 集合、bootstrap 和
  success/failure/stop/claim boundary 均在 config/validator 中 fail closed。该 probe
  不调用 clean validation 或 robustness-as-main-metric 路径。
- Output non-overwrite: formal root 固定为
  `/root/autodl-tmp/ue_project/runs_research/TAUSB-SIRC-v1`；构造 workflow 时路径存在即
  `FileExistsError`，无删除、覆盖或 resume 语义。远端路径新鲜性尚未核验。
- Recoverability/secrecy: 正式命令尚未生成 tmux session/log，因为 review 未通过；
  CSV、Spec、代码和本地 artifacts 未记录 SSH 主机、端口、用户名、密钥或令牌。
- Blockers:
  1. 12 个本地已保存 AutoDL profiles 均只读连接失败；远端 checkout、Python/CUDA/GPU、
     VOC、checkpoint、source local map 与 fresh artifact root 均未核验。
  2. reviewed branch 尚未获用户明确授权推送到上述具体 GitHub 仓库，远端无法保证
     checkout `baaabcddd2fb76c892ee0b21c971987b50fe4560`。
  3. frozen `semantic_proxy_cosine` 门禁存在判别力风险：smoke 后 I-SV global-P5
     cosine 为 `0.85938`，I-SPC-V 为 `0.84060`，delta `0.01878`，低于 Spec 的
     failure threshold `0.03`；保留空间布局的 raw/centered P5 cosine delta 也只有
     `0.02071/0.02488`。同时像素结构 NCC 对比为 `0.671/0.061`。这表明 P5 proxy
     可能对 phase 不敏感。用户必须决定：保持冻结门禁并接受 Phase A 很可能立即
     FAIL，或先修订 Spec 的该诊断/阈值再正式运行；不得在执行中暗改。
- Validation gaps:
  - smoke 只证明机械链路，不证明 Phase A/B 机制成功或 fresh-victim UE；
  - formal held-out cue/JPEG/bootstrap 全量路径尚未运行；
  - 没有远端输入与环境证据；
  - 分支未推送，未生成正式 tmux 命令。

结论：`REMOTE-PROBE-01` 保持关闭。只有用户对具体 GitHub push 给出明确授权、恢复
可连接 AutoDL、完成全部远端只读输入核验，并对 global-P5 proxy 门禁作出明确决定后，
才可重建独立 review packet 并考虑 `allow_run`。
