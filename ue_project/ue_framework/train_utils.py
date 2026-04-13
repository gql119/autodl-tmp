import os
import shutil
from typing import Iterable



def remove_yolo_cache_files(paths: Iterable[str]) -> int:
    deleted = 0
    for root in paths:
        if not root or not os.path.exists(root):
            continue
        if os.path.isfile(root) and root.endswith(".cache"):
            try:
                os.remove(root)
                deleted += 1
            except OSError:
                pass
            continue

        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith(".cache"):
                    p = os.path.join(dirpath, name)
                    try:
                        os.remove(p)
                        deleted += 1
                    except OSError:
                        pass
    return deleted



def copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True

