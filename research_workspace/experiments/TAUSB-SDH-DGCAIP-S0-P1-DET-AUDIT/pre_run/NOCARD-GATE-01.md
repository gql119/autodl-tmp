# TAUSB P1 Determinism Audit — Remote No-card Gate

## Verdict

`PASS` for the approved GPU P1 determinism audit only.

No GPU audit or effectiveness experiment was started during this gate.

## Frozen identity

- SpecID: `TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT`
- Branch: `codex/tausb-sdh-dgcaip-p1-determinism-audit-v1`
- Implementation commit: `067fd35c3a3a71f4905bcfc613d8492a301796a9`
- Config SHA256:
  `064f4ee3a9cbfeacdd141c59e754cf1ca926249952cb3773014582a0402d1679`
- Remote checkout:
  `/root/autodl-tmp/tausb-dgcaip/preflight-checkouts/cc55bf2-p1-det-audit`

The remote checkout was detached at the exact commit and clean after adding
the tracked secret assets to its sparse path set.

## Test evidence

Environment controls used for the large CPU-only host:

```text
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
pytest cache provider disabled
```

- P1 audit suite: `22 passed in 4.57s`, exit 0.
- Adjacent regression suite: `59 passed in 6.32s`, exit 0.
- Total: `81 passed`, no failures.

The initially observed 60-second timeout in the renderer observation test was
reproduced as CPU thread over-subscription. The identical test passed in 4.41
seconds after bounding OpenMP/MKL to one thread. No production method change
was required.

## Frozen input evidence

The production input-validation path completed on CPU without creating the
experiment artifact root:

- train images: 16,551;
- train labels: 16,551;
- person-containing images: 6,095 (validated internally);
- calibration plus held-out audit split: 160;
- split SHA256:
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`;
- first batch IDs: `000321`, `000777`, `001362`, `001686`;
- frozen surrogate, secret, hiding, P1 state/metrics, and D0 bindings: PASS;
- planned GPU artifact root: absent.

## Remote terminal state

- GPU: unavailable in no-card mode (`No devices were found`).
- Audit tmux session: absent.
- System disk: 21 GiB available (31% used).
- Data disk: 4.2 GiB available (92% used).
- No cleanup, dataset write, checkpoint write, parameter update, or GPU launch
  occurred.

## Next action

After the user opens GPU mode, recheck the frozen identity and launch the
reviewed 300-second controller once. Retain all terminal evidence regardless
of result, then stop; no automatic retry or downstream experiment is allowed.
