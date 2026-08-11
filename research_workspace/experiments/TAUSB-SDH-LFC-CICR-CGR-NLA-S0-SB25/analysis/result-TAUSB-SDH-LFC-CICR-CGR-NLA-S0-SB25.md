# TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25 H→E→N analysis

## Outcome

The seed-0 SB25 hiding-only pilot completed normally, but the pre-registered revised hiding gate
**failed**. The spectral bottleneck fixed the high-frequency failure, while secret identity
recovery regressed below two independent hard thresholds. Mechanism, victim training and AP50
evaluation remain blocked.

## Hypothesis

With the r2 protocol held fixed, preserve the Haar LL residual subband and multiply LH/HL/HH by
`0.25` before `tanh`. Success required all revised hard checks: high-frequency energy median
`<= 0.40`, retrieval top-1 `>= 0.90`, primary recovery SSIM median `>= 0.50`, relative L1 margin
median `>= 0.20`, pairwise pixel cosine median `< 0.98`, finite/Linf/support checks, and the
frozen non-target leakage probe. Channel RMS CV was pre-registered as descriptive only.

## Evidence

- Spec: `docs/research/specs/TAUSB-SDH-HIDING-SB-v1.md`.
- Run: `HIDING-S0-SB25-R1`; branch
  `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`; exact code commit
  `d244c3270eb24d7a6515e79ff643cb015ebb0bb9`.
- Canonical hiding metrics SHA-256:
  `9d2e804975d2cbe020dc0fc7b03254f9d2ebb56d3224995f168266d27ab8f67b`.
- Status SHA-256: `08eb0685398c0aaeeaa2295f89a026b90a35a8155029454ddfbb37e49bb8357a`.
- Split SHA-256: `0cc6bd107b33b842b9fbc511bc567293c0f72a90e164058fab2ce3d689c319d0`;
  the embedded split identity is
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`, matching r2.
- Input-audit SHA-256:
  `83aa3db3430eabc10e88a2f1b880cc177873fd6c640ffbd5f104f13b243daad9`.
- Nine required small files were pulled. Canonical status, metrics and split files exactly match
  their shutdown-ready snapshots. No checkpoint, dataset, weight or image tree was transferred.
- The status is `completed`, the cost guard records `hiding_exit=0` and shutdown requested, and
  elapsed optimization time was `22.2687 s`. No mechanism or victim command ran.

| Metric | Frozen r2 | SB25 | Required | SB25 result |
|---|---:|---:|---:|---|
| finite | true | true | true | pass |
| Linf | 0.06274510 | 0.06274510 | <= 0.06666667 | pass |
| support outside max | 0.0 | 0.0 | = 0 | pass |
| high-frequency energy median | 0.671792 | 0.034223 | <= 0.40 | pass |
| secret retrieval top-1 | 1.000000 | 0.424479 | >= 0.90 | **fail** |
| primary recovery SSIM median | 0.667435 | 0.641143 | >= 0.50 | pass |
| primary relative L1 margin median | 0.355765 | 0.067159 | >= 0.20 | **fail** |
| pairwise pixel cosine median | 0.958555 | 0.682427 | < 0.98 | pass |
| channel RMS CV (R/G/B) | 0.018599 / 0.010226 / 0.015434 | 0.013114 / 0.015315 / 0.012007 | descriptive | report only |
| co-occurrence balanced accuracy | 0.531250 | 0.458333 | <= 0.60 | pass |
| non-target macro AUROC | 0.550237 | 0.477039 | <= 0.60 | pass |

The high-frequency ratio fell by `0.637568` absolute (about 94.9% relative), so the parameter
reached the intended spectral sink and decisively removed the r2 high-frequency failure. At the
same time, retrieval fell by `0.575521` and the relative L1 margin fell by `0.288606`. The reveal
loss decreased from `0.339798` at step 1 to `0.257343` at step 120, which rules out a crashed or
completely static optimization but does not rescue the failed identity gates.

No visualization artifacts were produced or pulled, so no visual-quality or naturalness claim is
made. PSNR, LPIPS, victim AP50, poisoned counts and detector outputs do not exist for this
hiding-only pilot. `exp-results-ingest-local` was intentionally not run because it requires a
canonical detector `metrics/metrics.json`; `hiding-metrics-summary.json` records only the
observed hiding metrics and does not populate the AP50 ledger.

## Interpretation

This is an attack/recovery trade-off, not a Pareto improvement. The 0.25 bottleneck prevents the
encoder from relying on the r2 high-frequency channel and also yields less pixel-aligned residuals,
but it suppresses too much discriminative secret information. SSIM alone is insufficient here:
the decoder reconstructs a broadly similar image while failing to distinguish the primary secret
from the other same-semantic bank members. Consequently, the result supports the causal role of
high-frequency capacity but rejects `hf_subband_scale=0.25` as the approved carrier setting.

The leakage probe remains below both caps, but this is only a feature-probe result. It is not
evidence that 19 non-target detector AP50 values are preserved.

## Next

The smallest discriminating next experiment is one matched hiding-only run with
`hf_subband_scale=0.50`, keeping the secret bank, split, seed, 120 steps, epsilon, losses and every
gate unchanged. It directly tests whether an intermediate spectral capacity can retain the large
high-frequency safety margin while restoring retrieval and L1 identity margin. It requires a new
approved Spec; no GPU run is authorized by this analysis.
