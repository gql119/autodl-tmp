import json
import random
from typing import Any, Dict, List, Sequence, Tuple

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


def sample_bandfreq_coords(
    h: int,
    w: int,
    band_names: Sequence[str],
    band_num_bases: Sequence[int],
    band_radius_ranges: Dict[str, Sequence[float]],
    seed: int,
    enable_search: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if len(band_names) != len(band_num_bases):
        raise ValueError("band_names and band_num_bases must have the same length.")

    rng = random.Random(seed)

    fy = np.fft.fftfreq(h) * float(h)
    fx = np.fft.fftfreq(w) * float(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)

    all_nonzero = np.argwhere(rr > 0)
    if len(all_nonzero) == 0:
        all_nonzero = np.array([[1 % h, 1 % w]], dtype=np.int64)

    out_coords: List[Tuple[int, int]] = []
    out_meta: List[Dict[str, Any]] = []
    global_index = 0
    num_bands = len(band_names)

    for band_id, (band_name, num_bases) in enumerate(zip(band_names, band_num_bases)):
        k = int(num_bases)
        if k < 0:
            raise ValueError(f"band_num_bases for '{band_name}' must be >= 0, got {k}.")
        if k == 0:
            continue

        radius_range = band_radius_ranges.get(str(band_name))
        if radius_range is None or len(radius_range) != 2:
            raise ValueError(f"Missing or invalid radius range for band '{band_name}'.")

        lo = float(radius_range[0])
        hi = float(radius_range[1])
        if hi < lo:
            raise ValueError(f"Invalid radius range for band '{band_name}': [{lo}, {hi}]")

        if band_id < num_bands - 1:
            mask = (rr >= lo) & (rr < hi)
        else:
            mask = (rr >= lo) & (rr <= hi)
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            candidates = all_nonzero

        candidate_list = [tuple(map(int, t)) for t in candidates.tolist()]
        if len(candidate_list) == 0:
            candidate_list = [(1 % h, 1 % w)]

        if enable_search:
            if len(candidate_list) >= k:
                picks = rng.sample(candidate_list, k=k)
            else:
                picks = list(candidate_list)
                while len(picks) < k:
                    picks.append(candidate_list[rng.randrange(len(candidate_list))])
        else:
            candidate_list = sorted(candidate_list, key=lambda t: (t[0], t[1]))
            picks = [candidate_list[i % len(candidate_list)] for i in range(k)]

        for local_index, (y, x) in enumerate(picks):
            y_i = int(y)
            x_i = int(x)
            out_coords.append((y_i, x_i))
            out_meta.append(
                {
                    "index": int(global_index),
                    "band": str(band_name),
                    "band_id": int(band_id),
                    "local_index": int(local_index),
                    "y": int(y_i),
                    "x": int(x_i),
                    "radius": float(rr[y_i, x_i]),
                }
            )
            global_index += 1

    return out_coords, out_meta


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

