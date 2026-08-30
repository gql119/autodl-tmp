# TAUSB P1 Deterministic Resize Repair Attestation

## Verdict

`REPAIR_PASS` for the bounded P1 G0 -> G1 -> G2 execution gate.

- execution commit: `83f43070f04e2a98401ad17ec098c01d83d96665`
- config SHA256:
  `0294f29190b60b168afc54ac25e41eb5509a6103ceddf095bc713281a9480900`
- controller elapsed time: 233 seconds of the 480-second hard cap
- controller exit code: 0
- wrapper terminal stage: `summarize`
- shutdown requested: true

This gate did not run P2/P4, dataset materialization, victim training, AP50,
E20, or E200.

## G0 - deterministic resize

- status: passed
- three forward replays: bitwise identical
- three input-gradient replays: bitwise identical
- maximum forward absolute error: `8.828938007354736e-07`
- approved threshold: `2e-6`
- source maximum absolute value: `0.06274502724409103`
- frozen epsilon: `0.06274509803921569` (`16/255`)
- 32-iteration benchmark: `0.11861307546496391` seconds
- peak CUDA memory: 98,186,752 bytes
- no unsupported deterministic operator, NaN, Inf, OOM, or traceback

## G1 - P1 replay

- primary label: `strict_replay_pass`
- input/state validation: passed
- strict fresh A/B: bitwise identical
- strict deterministic-operator error: false
- normal reset diagnostic: not bitwise identical

The normal reset mismatch is diagnostic only under the approved repair Spec.
The strict fresh lane is the scientific gate and passed exactly.

## G2 - two-step writeback

- status: passed
- accepted steps: 2/2
- adapter changed: true
- adapter SHA256 before:
  `4f8faeb84a0edc5544426321231fe2b6fc1bc13fb591100ec9eca08ae3e0d591`
- adapter SHA256 after:
  `944fe64b7eb6951ef9a22a3e7d35682e4809982192cd8840865fce7d94ea39a0`
- state loadable: true
- finite: true
- support valid: true
- perturbation Linf: `0.06274443864822388`
- hiding trunk, reveal decoder, YOLO model, D-LFC bank, CICR bank,
  DG-CAIP weights, NLA weight, and target weights: unchanged

The P1 smoke state SHA256 is
`743f48073a58a6687d5e01a01251d6187ff21070024f12184eb976e50416eb35`.
It remains an experiment checkpoint and is excluded from Git.

## Evidence integrity

- all 10 files listed by the remote artifact manifest matched their local
  SHA256 values;
- all four control JSON files and the outer log matched remote SHA256 values;
- the first failed G0 evidence remains preserved separately;
- no dataset, model weight, historical evidence, or unrelated dirty worktree
  content was modified or deleted.

## Consequence

The deterministic resize/P1 execution blocker is closed. This attestation
permits the next bounded paired E20 execution gate; it is not P4 or AP50
effectiveness evidence.
