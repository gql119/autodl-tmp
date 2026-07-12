import os
from typing import Dict

import torch


def save_subspaces(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_subspaces(path: str) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "layers" not in payload:
        raise ValueError("invalid DCSS subspace payload")
    for layer_name, layer_data in payload["layers"].items():
        for source, ranks in layer_data.get("subspaces", {}).items():
            for rank, basis in ranks.items():
                q = basis.to(dtype=torch.float64)
                error = (q.T @ q - torch.eye(q.shape[1], dtype=q.dtype)).abs().max().item()
                if error > 1e-4:
                    raise ValueError(f"non-orthonormal subspace {layer_name}/{source}/{rank}: {error:.3e}")
    return payload
