import csv
import json
import os
import tempfile
from typing import Dict, Iterable, List



def atomic_write_text(path: str, text: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=folder or None, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, path)



def atomic_write_json(path: str, obj: Dict) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))



def atomic_write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=folder or None, encoding="utf-8", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = tmp.name
    os.replace(tmp_path, path)



def read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def read_csv_rows(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def append_text(path: str, line: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

