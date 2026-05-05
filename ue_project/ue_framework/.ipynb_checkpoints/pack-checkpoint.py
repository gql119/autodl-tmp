import os
import zipfile
from typing import List

from .paths import RunPaths


INCLUDE_RELATIVE = [
    "noise",
    "logs",
    "metrics",
    "checkpoints",
    "status.json",
]



def pack_run_artifacts(paths: RunPaths) -> str:
    os.makedirs(paths.bundle_dir, exist_ok=True)

    include_files: List[str] = []

    for rel in INCLUDE_RELATIVE:
        p = os.path.join(paths.artifact_root, rel)
        if os.path.isfile(p):
            include_files.append(p)
        elif os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for name in files:
                    include_files.append(os.path.join(dirpath, name))

    # include poisoning metadata only (no poisoned images)
    for p in [paths.manifest_csv, paths.poisoned_status_json]:
        if os.path.isfile(p):
            include_files.append(p)

    with zipfile.ZipFile(paths.bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in include_files:
            if src.startswith(paths.artifact_root):
                arc = os.path.relpath(src, start=paths.artifact_root)
                arcname = os.path.join("artifacts", arc)
            else:
                arcname = os.path.join("poison_meta", os.path.basename(src))
            zf.write(src, arcname)

    return paths.bundle_path

