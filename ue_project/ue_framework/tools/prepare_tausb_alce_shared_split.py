from __future__ import annotations

import argparse
import json
from pathlib import Path

from ue_framework.io_utils import atomic_write_json
from ue_framework.methods.bsc_rc_gr_probe import (
    build_alce_shared_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen TAUSB ALCE/BSC shared VOC split."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    manifest = build_alce_shared_split_manifest(
        image_dir=dataset_root / "images" / "train",
        label_dir=dataset_root / "labels" / "train",
    )
    if output.exists():
        if not args.verify_existing:
            raise FileExistsError(f"Refusing to overwrite shared split: {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing shared split differs from recomputed manifest.")
        status = "verified"
    else:
        if args.verify_existing:
            raise FileNotFoundError(f"Shared split does not exist: {output}")
        atomic_write_json(str(output), manifest)
        status = "created"
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "split_hash": manifest["split_hash"],
                "label_hash": manifest["label_hash"],
                "group_counts": manifest["group_counts"],
                "validation_gaps": manifest["validation_gaps"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
