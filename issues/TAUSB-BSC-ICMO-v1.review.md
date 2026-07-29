# TAUSB-BSC-ICMO-v1 Pre-run Implementation Review

## PRERUN-REVIEW-01

- Result: `blocked`
- Decision: `do_not_run`
- Gated run: `REMOTE-PROBE-01`
- Code snapshot: branch `codex/tausb-bsc-rc-gr-v1`, commit
  `023a75f20ccbd12cdec4701a12b90ae5c499a1cb`
- Intent: run the frozen matched 2×2 surrogate ICMO mechanism probe, without
  making victim-unlearnability, clean-mAP, perceptual-quality, or true-mask
  claims.
- Code location: isolated workflow
  `ue_project/ue_framework/methods/bsc_icmo_probe.py`, CLI
  `ue_project/ue_framework/tools/probe_tausb_bsc_icmo.py`, and formal config
  `ue_project/ue_framework/configs/exp_voc_person_tausb_bsc_icmo_probe.yaml`.
- Parameter data flow: CLI → frozen YAML → `ICMOProbeWorkflow` → verified
  inputs/bases/shared gamma/z0 → matched renderer → real YOLOv8/TAL
  `target_gt_idx` → Instance-CICR/easy-cls/RMS loss → per-arm
  metrics/bootstrap/decision/status.
- Runtime state: VOC20, `person=14`, `eps=16/255`, seed 0, frozen surrogate,
  detached train-only prototype bank, and only 48 carrier coefficients
  trainable per arm.
- Sink effect: the changed parameterization and renderer directly affect the
  input seen by the surrogate and the active loss; their held-out effects reach
  `arms/*_metrics.json`, `metrics.json`, and `status.json`.
- Baseline/disable path: four matched control arms share all frozen factors;
  old TAUSB/BSC entry points remain unchanged; 77 repository tests pass.
- Local validation: 18 new tests, 22 relevant regressions, 77 full-suite tests,
  formal config validation, and one completed real VOC/YOLO mechanical smoke.
- Minimal probe: smoke `TAUSB-BSC-ICMO-v1-smoke-20260730-b` crossed real TAL,
  one I-C2LM backward, and one G-C0 forward with finite loss/gradient, matched
  amplitude, full-rank bases, and zero outside-support perturbation.
- Run command binding: inner command is frozen below but is not executable until
  the remote-input row and a second pre-run review pass.
- Experiment validity: mechanism-only, calibration/held-out shared split,
  no victim training, clean evaluation, aggregation, or robustness transforms.
- Output non-overwrite: workflow raises on an existing artifact root; formal
  remote root freshness remains unverified.
- Recoverability/secrecy: tmux/session/log identifiers are planned below; no
  connection host, port, username, or key is persisted.
- Blockers: no connected authorized AutoDL profile; remote checkout, environment,
  frozen inputs/hashes, and artifact-root freshness are unverified.
- Validation gaps: no formal four-arm metrics, no victim mAP, no clean mAP, and
  no dataset-level PSNR/LPIPS evidence.
- Review date: 2026-07-30
- Formal config hash:
  `685333e6f6268a0a108e5b415d55fb3406605b8c9e07ab81c86284cd04f9d9bc`

The implementation is locally ready for remote input verification, but the formal
GPU probe is not authorized to start yet. No currently configured SSH profile
could connect to the AutoDL environment, so the remote inputs, exact checkout,
GPU environment, and fresh artifact root could not be independently verified.

## Intent and claim boundary

This is the frozen 2×2 surrogate mechanism probe from
`docs/research/specs/TAUSB-BSC-ICMO-v1.md`:

| Arm | Basis family | Renderer |
|---|---|---|
| G-C0 | matched synthetic Fourier | global coordinate |
| G-C2LM | phase-scrambled natural low+mid | global coordinate |
| I-C0 | matched synthetic Fourier | instance canonical |
| I-C2LM | phase-scrambled natural low+mid | instance canonical |

The probe can support only a surrogate-level mechanism claim about instance
canonicalization, natural low/mid-frequency bases, residual consistency, and
their interaction. It cannot establish fresh-victim unlearnability, clean VOC
mAP, dataset PSNR/LPIPS, or a true instance-mask claim. The support remains the
documented forced pseudo ellipse fallback.

## Parameter and data-flow audit

The active path was re-read from the frozen config and code snapshot:

1. `probe_tausb_bsc_icmo.py` parses the YAML, validates the frozen schema, and
   passes the resolved config and device to `ICMOProbeWorkflow`.
2. `ICMOProbeWorkflow` fails closed on an existing artifact root, loads VOC20
   with `person=14`, verifies the shared split and label hashes, resolves the
   authorized background source manifest through a local path map, verifies the
   C2-LM source-basis hash, verifies the C0 coordinate-pack hash, and verifies
   the surrogate checkpoint hash.
3. C0 and C2-LM bases use the same 16-dimensional coefficient vector
   initialization, shared gamma calibration, epsilon, optimizer, batch order,
   four-step route warmup, and 40 optimization steps.
4. The global and instance-canonical renderers share the same canonical
   pattern. The instance path maps it into each person box, applies the exact
   forced pseudo ellipse primitive, averages overlaps, then applies JND and
   clamping. Outside-support perturbation is audited.
5. The real YOLOv8/TAL path exposes `target_gt_idx`; target positives are grouped
   per person GT before Instance-CICR. Missing positive assignments remain in the
   coverage denominator. The optimization sink is
   `CICR + easy_cls route + canonical RMS`.
6. The only trainable state is the arm's 48 carrier coefficients. Surrogate
   parameters are frozen. CICR prototypes are detached train-only state and are
   not updated by held-out evaluation.
7. Held-out outputs reach per-arm metrics, affine audits, 10,000-iteration
   stratified paired bootstrap contrasts, decision gates, `metrics.json`, and
   `status.json`. Non-finite values, failed mechanical prerequisites, and
   frozen-hash mismatches fail closed.

## Baseline, isolation, and regression audit

- All four arms are created by the same workflow and differ only by basis family
  and renderer selected from the frozen 2×2 definition.
- Existing TAUSB/BSC entry points are not redirected to the new workflow.
- The support refactor reuses one bbox-ellipse drawing primitive for both the
  legacy fallback and the new per-instance helper.
- The formal artifact root is creation-only: an existing directory raises
  `FileExistsError`; there is no automatic deletion, overwrite, or resume path.
- The formal run does not invoke victim training, clean evaluation, aggregation,
  or robustness transforms.

## Local validation evidence

- New ICMO tests: 18 passed.
- Existing relevant background/BSC/CICR/hooks/routes regressions: 22 passed.
- Full repository suite: 77 passed.
- Formal CLI `--validate-only`: passed with config hash
  `685333e6f6268a0a108e5b415d55fb3406605b8c9e07ab81c86284cd04f9d9bc`.
- Real local VOC/YOLO smoke:
  `ue_project/runs_research_local/TAUSB-BSC-ICMO-v1-smoke-20260730-b/`.
  It crossed model initialization, real TAL assignment, shared gamma
  calibration, one I-C2LM backward, and one G-C0 forward.
- Smoke evidence: 4 images, 6 person instances, valid instance coverage
  `0.833333`, finite total loss `2.734033`, finite carrier gradient norm
  `7.149893`, initial matched RMS ratio `1.005250`, renderer NCC median
  `0.999965`, outside-support maximum `0` for both renderers, basis ranks
  `16/16`, zero trainable model parameters, and 48 trainable carrier
  coefficients.
- The earlier smoke root ending in `-a` is preserved as a failed diagnostic. It
  exposed a CPU placeholder/CUDA residual device mismatch; the fixed `-b` run
  passed. This is mechanical smoke evidence only.

## Frozen formal command

The inner command to be run only after a second pre-run review passes is:

```bash
cd /root/autodl-tmp/ue_project
python -u -m ue_framework.tools.probe_tausb_bsc_icmo \
  --config ue_framework/configs/exp_voc_person_tausb_bsc_icmo_probe.yaml \
  --stage all \
  --device 0
```

Planned persistent identifiers, subject to remote verification:

- tmux session: `tausb-bsc-icmo-s0-023a75f`
- driver log:
  `/root/autodl-tmp/ue_project/runs_research/TAUSB-BSC-ICMO-v1.driver.log`
- artifact root:
  `/root/autodl-tmp/ue_project/runs_research/TAUSB-BSC-ICMO-v1`

## Blocking findings

1. No currently configured SSH profile connected to the authorized AutoDL
   environment during this review.
2. The remote checkout of branch `codex/tausb-bsc-rc-gr-v1` at code commit
   `023a75f20ccbd12cdec4701a12b90ae5c499a1cb` has not been verified.
3. Remote Python/CUDA/GPU readiness has not been verified.
4. Remote VOC dataset, label set, shared split, source manifest/local map,
   surrogate checkpoint, C0 coordinate pack, and C2-LM basis hashes have not
   been recomputed against the frozen values.
5. The formal artifact root's absence/freshness has not been verified.

These findings block only formal execution. They do not invalidate the local
implementation or the scientific hypothesis.

## Required remediation and re-review

- `PRERUN-REMOTE-INPUTS-01`: restore an authorized AutoDL connection and perform
  read-only verification of checkout, environment, inputs, hashes, and artifact
  root without persisting connection secrets.
- `PRERUN-REVIEW-02`: independently review that evidence, bind the exact
  branch/commit/session/log/command, and issue `pass / allow_run` before
  `REMOTE-PROBE-01`.

Until both rows close, `REMOTE-PROBE-01` remains `not_started`.
