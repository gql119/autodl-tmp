import os
import random
import subprocess
from datetime import datetime
from typing import Dict

import numpy as np
import torch



def detect_platform(cfg: Dict) -> str:
    mode = cfg["platform"].get("mode", "auto")
    if mode != "auto":
        return mode
    if os.path.isdir("/kaggle/input"):
        return "kaggle"
    return "local"



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def select_device(gpu_id: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")



def resolve_workers(platform_mode: str, cfg: Dict) -> int:
    if platform_mode == "local":
        return 0
    return int(cfg["victim"].get("workers", 2))



def collect_runtime_info(seed: int, method: str, steps: int) -> Dict:
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "method": method,
        "steps": steps,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        info["git_commit"] = proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except Exception:
        info["git_commit"] = "unknown"
    return info

