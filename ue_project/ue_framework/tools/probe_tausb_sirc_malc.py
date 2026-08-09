from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ue_framework.methods.sirc_malc_mechanism import (  # noqa: E402
    SIRCMALCMechanismWorkflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched no-EOT SIRC MALC A0/A1 mechanism gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--source-local-map", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    workflow = SIRCMALCMechanismWorkflow(
        config,
        config_path=config_path,
        device_override=args.device,
        source_manifest=args.source_manifest,
        source_local_map=args.source_local_map,
    )
    try:
        status = workflow.run(smoke=bool(args.smoke))
    finally:
        workflow.close()
    print(status)


if __name__ == "__main__":
    main()
