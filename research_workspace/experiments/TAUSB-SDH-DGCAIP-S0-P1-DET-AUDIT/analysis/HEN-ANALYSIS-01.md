# TAUSB P1 Determinism Audit — H→E→N Analysis

## Registered outcome

- Primary label: `cuda_nondeterministic_operator`
- Mechanical audit result: `PASS`
- Scientific determinism result: `FAIL`
- Retry performed: no
- Parameter update performed: no
- Downstream P2/P4/victim/AP50 work performed: no

`mechanical_pass=true` means the bounded diagnostic completed with valid
evidence. It does not mean the P1 optimization is deterministic.

## H — Hypothesis

The previously unstable P1 decision is driven primarily by a nondeterministic
CUDA backward operator in the differentiable carrier-resize path, rather than
by different batches, dirty model/carrier state, failed RNG restoration, or an
accepted parameter update.

## E — Evidence

### Execution validity

- Execution commit:
  `067fd35c3a3a71f4905bcfc613d8492a301796a9`
- Config SHA256:
  `064f4ee3a9cbfeacdd141c59e754cf1ca926249952cb3773014582a0402d1679`
- Split SHA256:
  `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`
- First batch: `000321`, `000777`, `001362`, `001686`
- R4 read-only prelude observations: 44
- Normal lane elapsed time: 52.61 seconds
- Controller, wrapper, and summary exit codes: 0
- Artifact size: 370,757 bytes; all eight manifest entries verified locally
  and remotely.
- Controller JSON and outer-log SHA256 values also matched between local and
  remote copies.

### Controls

- All three normal pairs used identical input tensors.
- Reset and fresh pairs had identical initial state hashes.
- All pairs preserved parameter state; no update was accepted.
- The shared pair intentionally advances shared engine/RNG state between A and
  B, so its initial-state hashes differ and it is not used for the state-valid
  decision.
- Rendered hosts, canonical deltas, resized patches, final perturbation,
  poisoned images, clean/poison forward tensors, TAL assignments, and scalar
  losses were exact in the shared trace before gradient computation.

### First divergence

All shared/reset/fresh pairs produced the same ordered diagnosis:

1. First bitwise divergence: `grad.components/cicr`.
2. First divergence exceeding the registered numerical tolerance:
   `grad.components/dlfc`.
3. The drift propagated into the composed target gradient and the projected
   final update direction.

For the state-controlled reset/fresh pairs:

| quantity | reset max abs | fresh max abs |
|---|---:|---:|
| CICR gradient | 2.7381e-7 | 1.3933e-6 |
| DLFC gradient | 2.1342e-5 | 1.9602e-5 |
| target gradient | 1.1063e-3 | 7.2670e-4 |
| projected target gradient | 8.3208e-4 | 6.8474e-4 |
| combined routed gradient | 8.3590e-4 | 6.8474e-4 |
| candidate probability drop | 3.7134e-6 | 7.1644e-6 |

### Strict lane

Strict deterministic algorithms stopped at:

```text
upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation
```

The error occurred during `torch.autograd.grad` in the target-component
gradient-norm calibration path. It was captured as the pre-registered
`strict_operator_error.json` terminal rather than being silently downgraded.

## N — Narrow conclusion and next action

The audit supports a narrow causal conclusion: a CUDA bilinear-upsample
backward operation is an identified nondeterministic source, and normal-mode
gradient drift begins in CICR/DLFC before being amplified by target-objective
composition and constraint projection. The evidence does not show that CICR,
DLFC, or CGR is conceptually invalid; it shows that their current differentiable
geometry path is not reproducible enough for a stable pass/fail gate.

The next action should be a separate minimal repair Spec that:

1. locates every bilinear interpolation whose backward reaches carrier adapter
   parameters;
2. replaces only the implicated resize-backward path with a CUDA-deterministic
   alternative verified under `torch.use_deterministic_algorithms(True)`;
3. keeps the carrier, losses, weights, batch, split, and thresholds frozen;
4. reruns this same bounded P1 audit once before allowing P2/P4 or any victim
   training.

Using `warn_only=True` is not an acceptable repair because it would retain the
nondeterministic operator and merely suppress the fail-closed evidence.
