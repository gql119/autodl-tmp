from datetime import datetime
from typing import Dict

from .io_utils import atomic_write_json, read_json



def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"



def load_or_init_status(path: str, method: str, steps: int, seed: int) -> Dict:
    status = read_json(path, default=None)
    if status is None:
        status = {
            "method": method,
            "steps": steps,
            "seed": seed,
            "created_at": _now(),
            "updated_at": _now(),
            "current_stage": "",
            "completed_stages": [],
            "stage_state": {},
        }
        atomic_write_json(path, status)
    return status



def save_status(path: str, status: Dict) -> None:
    status["updated_at"] = _now()
    atomic_write_json(path, status)



def mark_stage_running(path: str, status: Dict, stage: str) -> Dict:
    status["current_stage"] = stage
    state = status.setdefault("stage_state", {}).setdefault(stage, {})
    state["status"] = "running"
    state["start_time"] = _now()
    save_status(path, status)
    return status



def mark_stage_completed(path: str, status: Dict, stage: str, extra: Dict = None) -> Dict:
    state = status.setdefault("stage_state", {}).setdefault(stage, {})
    state["status"] = "completed"
    state["end_time"] = _now()
    if extra:
        state.update(extra)
    if stage not in status.setdefault("completed_stages", []):
        status["completed_stages"].append(stage)
    status["current_stage"] = ""
    save_status(path, status)
    return status



def stage_completed(status: Dict, stage: str) -> bool:
    return stage in status.get("completed_stages", [])

