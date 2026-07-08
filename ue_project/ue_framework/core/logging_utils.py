from __future__ import annotations

import csv
import os
from typing import Dict, Iterable


def append_metrics_csv(path: str, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(fieldnames)
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
