import json
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch



def sample_midfreq_coords(
    h: int,
    w: int,
    num_bases: int,
    seed: int,
    enable_search: bool = True,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)

    candidates = np.argwhere((rr >= 0.08) & (rr <= 0.26))
    if len(candidates) == 0:
        candidates = np.argwhere(rr > 0)

    if len(candidates) == 0:
        return [(1 % h, 1 % w)] * num_bases

    if enable_search:
        picks = rng.sample(list(map(tuple, candidates.tolist())), k=min(num_bases, len(candidates)))
    else:
        # deterministic fixed mid-frequency picks
        candidates = sorted(candidates.tolist(), key=lambda t: (t[0], t[1]))
        picks = [tuple(candidates[i % len(candidates)]) for i in range(num_bases)]

    return [(int(y), int(x)) for y, x in picks]



def build_fourier_pattern(
    h: int,
    w: int,
    coords: Sequence[Tuple[int, int]],
    amplitudes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    spec = torch.zeros((h, w), dtype=torch.complex64, device=device)
    for i, (yy, xx) in enumerate(coords):
        amp = amplitudes[i]
        c = torch.complex(amp, torch.zeros_like(amp))
        spec[yy % h, xx % w] = spec[yy % h, xx % w] + c
        spec[(-yy) % h, (-xx) % w] = spec[(-yy) % h, (-xx) % w] + c

    pattern = torch.fft.ifft2(spec).real
    denom = pattern.abs().amax().clamp_min(1e-6)
    pattern = pattern / denom
    return pattern.unsqueeze(0).unsqueeze(0)



def spectrum_to_numpy(
    h: int,
    w: int,
    coords: Sequence[Tuple[int, int]],
    amplitudes: np.ndarray,
) -> np.ndarray:
    spec = np.zeros((h, w), dtype=np.complex64)
    for i, (yy, xx) in enumerate(coords):
        amp = float(amplitudes[i])
        spec[yy % h, xx % w] += amp
        spec[(-yy) % h, (-xx) % w] += amp

    mag = np.log1p(np.abs(np.fft.fftshift(spec)))
    mag = mag - mag.min()
    if mag.max() > 1e-8:
        mag = mag / mag.max()
    return mag.astype(np.float32)



def save_coords_json(path: str, coords: Sequence[Tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"coords": [[int(y), int(x)] for y, x in coords]}, f, ensure_ascii=False, indent=2)

