from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ue_framework.methods.bsc_rc_gr_probe import (
    BSCProbeWorkflow,
    canonical_hash,
    validate_probe_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAUSB-BSC-RC-GR surrogate-only mechanism probes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=("A", "B", "C", "all"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--source-local-map", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate the frozen config without touching data or artifacts.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Probe config must be a YAML mapping.")
    validate_probe_config(config)
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
                    "phase": args.phase,
                    "claim_boundary": (
                        "schema validation only; no surrogate or mechanism evidence"
                    ),
                },
                indent=2,
            )
        )
        return

    workflow = BSCProbeWorkflow(
        config,
        config_path=config_path,
        source_manifest=args.source_manifest,
        source_local_map=args.source_local_map,
        device_override=args.device,
    )
    try:
        status = workflow.run(args.phase)
    finally:
        workflow.close()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
