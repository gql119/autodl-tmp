from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ue_framework.methods.bsc_icmo_probe import (
    ICMOProbeWorkflow,
    canonical_hash,
    validate_icmo_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TAUSB BSC-ICMO matched surrogate mechanism probe."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--source-local-map", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen schema without loading data, model, or artifacts.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ICMO config must be a YAML mapping.")
    validate_icmo_config(config)
    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "spec_id": config["spec"]["spec_id"],
                    "exp_id": config["spec"]["exp_id"],
                    "config_hash": canonical_hash(config),
                    "stage": args.stage,
                    "claim_boundary": (
                        "schema validation only; no surrogate or mechanism evidence"
                    ),
                },
                indent=2,
            )
        )
        return

    workflow = ICMOProbeWorkflow(
        config,
        config_path=config_path,
        source_manifest=args.source_manifest,
        source_local_map=args.source_local_map,
        device_override=args.device,
    )
    try:
        status = workflow.run()
    finally:
        workflow.close()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
