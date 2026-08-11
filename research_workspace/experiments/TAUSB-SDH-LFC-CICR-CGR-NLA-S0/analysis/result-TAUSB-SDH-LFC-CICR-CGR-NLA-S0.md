# TAUSB-SDH-LFC-CICR-CGR-NLA-S0 H→E→N analysis

## Outcome

The r2 single-secret hiding pilot is mechanically complete but the pre-registered hiding gate
**failed**. `PRERUN-MECHANISM-01`, the detector mechanism pilot, and every victim/AP50 stage
remain blocked. This result makes no detector-efficacy or unlearnability claim.

## Hypothesis

The approved Spec hypothesizes that a fixed semantic secret, hidden through different person
hosts, can remain recoverable while producing non-trivial sample-wise perturbations that do not
collapse to a fixed pixel code, do not encode non-target co-occurrence, and do not shift most
energy into high spatial frequencies.

For this hiding-only gate, success required all held-out checks in the approved Spec, including
`retrieval_top1 >= 0.90`, primary recovery `SSIM >= 0.50`, relative L1 margin `>= 0.20`,
pairwise pixel cosine `< 0.98`, every channel RMS CV `>= 0.05`, high-frequency energy ratio
`<= 0.40`, bounded/support-localized finite perturbations, and the frozen D-LFC leakage probe.

## Evidence

- Reviewed method commit: `20c35b6b69bca8a04a69608ddaba315e0ab88325`.
- Run identity: `HIDING-S0-R2`; deterministic split hash
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`.
- Transfer report:
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/remote_artifacts/transfer-report.json`.
- Hiding metrics (SHA-256
  `c7d1b120ffbadeb7385be41669dda704b00a2cee60940e3c3d97112e24e59246`):
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/remote_artifacts/hiding-20c35b6-r2/ready/hiding_metrics.json`.
- Formal status snapshot (SHA-256
  `ecd5a62662c087fd60f249c5257e3362d1b037620483e61f00e07eeb9b07c3e3`):
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/remote_artifacts/hiding-20c35b6-r2/ready/status_hiding.json`.
- Split manifest (SHA-256
  `0cc6bd107b33b842b9fbc511bc567293c0f72a90e164058fab2ce3d689c319d0`):
  `research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/remote_artifacts/hiding-20c35b6-r2/ready/split_manifest.json`.
- Cost guard and log prove exit code 0, evidence snapshot, and shutdown request. The run took
  `17.3829569 s`; no mechanism or victim stage was started.
- Remote SHA-256 comparison showed that each `ready/` snapshot exactly matches the formal r2
  status/metrics/split file. The preserved r1 status still hashes to
  `a0df443edb1b5e68ce875a2ef96c6ea77680215842851f17d52d69710f3adc0a`.
- Five of five required files were pulled and verified; no checkpoint, weight, dataset, or
  poisoned image tree was transferred.

| Gate metric | r2 | Required | Result |
|---|---:|---:|---|
| finite | true | true | pass |
| Linf | 0.06274510 | <= 0.06666667 | pass |
| support outside max | 0.0 | = 0 | pass |
| secret retrieval top-1 | 1.000000 | >= 0.90 | pass |
| primary recovery SSIM median | 0.667435 | >= 0.50 | pass |
| primary relative L1 margin median | 0.355765 | >= 0.20 | pass |
| pairwise pixel cosine median | 0.958555 | < 0.98 | pass |
| channel RMS CV (R/G/B) | 0.018599 / 0.010226 / 0.015434 | each >= 0.05 | **fail** |
| high-frequency energy median | 0.671792 | <= 0.40 | **fail** |
| co-occurrence balanced accuracy | 0.531250 | <= 0.60 | pass |
| non-target macro AUROC | 0.550237 | <= 0.60 | pass |

The reveal loss decreased from `0.332540` at step 1 to `0.191409` at step 120, so the result is
not explained by a crashed or completely untrained decoder. There were no NaN, Inf, OOM, or
Traceback findings in the pulled r2 log.

No `viz/` images were produced or pulled. Consequently, this report makes no visual-quality
or perceptual-imperceptibility claim. `exp-results-ingest-local` was intentionally not run:
this hiding-only pilot has no canonical `metrics/metrics.json`, victim, poisoned count, PSNR,
LPIPS, or AP50 values, and fabricating a detection metrics summary would violate the claim
boundary.

## Interpretation

The carrier learned a decodable, support-localized and non-target-leakage-safe secret signal,
but its encoding is not the intended robust sample-adaptive shortcut carrier. About `67.18%`
of non-DC perturbation energy lies beyond radius 64, exceeding the cap by `0.2718`. At the same
time, the minimum channel RMS CV is only `0.01023`, about one fifth of the required `0.05`.
Together these measurements support a narrow diagnosis: the current unconstrained residual
adapter uses a high-frequency, nearly fixed-energy steganographic channel across hosts. The
pixel-cosine pass shows that it is not literally the same pixel tensor, but the RMS failure
prevents describing it as sufficiently host-adaptive.

This is a pre-registered Failure Signal. Per the Spec, no mechanism checkpoint may consume this
hiding checkpoint and no victim/AP50 experiment is authorized.

## Next

The smallest discriminating experiment is one matched hiding-only variant against this frozen
r2 baseline: insert a fixed one-level Haar spectral bottleneck before `tanh`, preserving the LL
subband and scaling LH/HL/HH by `0.25`. No loss, split, secret, step count, epsilon, detector,
support, or downstream mechanism is changed. `scale=1.0` must be an output-equivalent rollback.

This tests one causal explanation: suppressing the cheap high-frequency channel should force the
existing host-conditioned network to express the secret through lower-frequency host structure,
potentially fixing both failed metrics. If high-frequency energy passes while RMS CV still fails,
the two failures are separable and a later host-diversity objective—not more arbitrary training
steps—is warranted. The proposed experiment is frozen separately in
`docs/research/specs/TAUSB-SDH-HIDING-SB-v1.md` and requires user approval before code changes.

