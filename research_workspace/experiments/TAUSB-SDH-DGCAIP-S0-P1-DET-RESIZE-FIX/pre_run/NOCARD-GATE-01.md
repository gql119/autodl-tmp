# TAUSB P1 Deterministic Resize Repair — Remote No-card Gate

## Verdict

`PASS` for the approved bounded GPU G0→G1→G2 gate only.

No GPU runner, mechanism update, poisoned-dataset generation, victim training,
or AP50 evaluation was started during this gate.

## Frozen identity

- SpecID: `TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-P1-DET-RESIZE-FIX`
- Branch: `codex/tausb-sdh-dgcaip-p1-det-resize-fix-v1`
- Reviewed runtime implementation commit:
  `f758355ea50b44f1576c6efdee47ea721342de75`
- Config SHA256:
  `0294f29190b60b168afc54ac25e41eb5509a6103ceddf095bc713281a9480900`
- Remote detached checkout:
  `/root/autodl-tmp/tausb-dgcaip/preflight-checkouts/f758355-p1-det-resize-fix`

The checkout matched the reviewed commit and remained clean after testing.
The pre-existing `/root/autodl-tmp` worktree had 2,102 tracked changes and was
therefore not used or modified.

## Test evidence

The large no-card CPU host was bounded with:

```text
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
pytest cache provider disabled
```

- Final focused and adjacent suite: `98 passed in 5.71s`, exit 0.
- The suite covers deterministic resize forward/gradient parity, host/adapter
  reachability, carrier support/Linf/overlap invariants, the new config and
  controller gates, the parent P1 audit, CGR, DG-CAIP experiment routing,
  target objective, E2E config, and non-target logit alignment.
- Controller and tmux-launcher `bash -n`: PASS.
- Local Python compile, CSV schema, AST/static checks, and `git diff --check`:
  PASS before the candidate snapshot.

An initial unbounded CPU run timed out after completing tests without an
assertion failure. This reproduced the parent audit's documented OpenMP/MKL
thread over-subscription. The exact same tests passed under the frozen
single-thread CPU test controls; no production code was changed for it.

## Frozen input evidence

The production input-validation path completed on CPU:

- train images: 16,551;
- train labels: 16,551;
- person-containing images: 6,095 (validated internally);
- calibration images: 64;
- held-out images: 96;
- split SHA256:
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`;
- first batch IDs: `000321`, `000777`, `001362`, `001686`;
- primary secret tensor shape: `[1, 3, 256, 256]`;
- surrogate, secret, hiding checkpoint, source P1 state/metrics, and D0 report
  bindings: PASS;
- planned GPU artifact root: absent.

## Remote terminal state

- AutoDL data disk mounted at `/root/autodl-tmp`;
- approximately 18 GiB available at review time;
- CUDA device unavailable in no-card mode;
- no GPU controller or experiment session started;
- no dataset, checkpoint, parameter, or historical evidence was deleted.

## Next action

After GPU mode is enabled, recheck the exact execution commit, unchanged config
hash, mounted data disk, absent artifact/control roots, and available CUDA
device. Then launch the reviewed controller once. Every terminal path requests
shutdown, and the total hard cap is 480 seconds. Preserve failures and do not
retry automatically.
