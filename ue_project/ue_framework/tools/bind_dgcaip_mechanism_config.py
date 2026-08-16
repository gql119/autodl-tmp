from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import yaml

from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind a passed DG-CAIP D0 report to the mechanism config."
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--d0-report", required=True)
    parser.add_argument(
        "--d0-reference",
        default="",
        help="Runtime-visible D0 path to embed; defaults to the read path.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_dgcaip_mechanism_config(
    template_path: Path,
    d0_report_path: Path,
    output_path: Path,
    *,
    d0_reference: str = "",
) -> Dict[str, Any]:
    template_path = template_path.resolve()
    d0_report_path = d0_report_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError("DG-CAIP bound config already exists: %s" % output_path)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    d0_report = json.loads(d0_report_path.read_text(encoding="utf-8"))
    if not isinstance(template, Mapping) or not isinstance(d0_report, Mapping):
        raise ValueError("DG-CAIP binding inputs must be mappings.")
    if not bool(d0_report.get("decision", {}).get("pass")):
        raise ValueError("DG-CAIP D0 gate did not pass; mechanism remains blocked.")
    bound = copy.deepcopy(template)
    dgcaip = bound.get("dgcaip", {})
    if dgcaip.get("run_mode") != "mechanism":
        raise ValueError("DG-CAIP template must use mechanism run_mode.")
    if d0_report.get("spec_id") != bound["spec"]["spec_id"]:
        raise ValueError("DG-CAIP D0 SpecID does not match the template.")
    if d0_report.get("split_hash") != dgcaip.get("expected_split_sha256"):
        raise ValueError("DG-CAIP D0 split hash does not match the template.")
    if d0_report.get("source_p1_state_sha256") != dgcaip.get(
        "source_p1_state_sha256"
    ):
        raise ValueError("DG-CAIP D0 source P1 hash does not match the template.")
    dgcaip["d0_report"] = str(d0_reference).strip() or str(d0_report_path)
    dgcaip["d0_report_sha256"] = _file_sha256(d0_report_path)
    validate_sdh_experiment_config(bound)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(bound, sort_keys=False), encoding="utf-8")
    report = {
        "schema": "tausb.dgcaip-mechanism-binding.v1",
        "spec_id": bound["spec"]["spec_id"],
        "template": str(template_path),
        "template_sha256": _file_sha256(template_path),
        "d0_report": str(d0_report_path),
        "d0_report_sha256": dgcaip["d0_report_sha256"],
        "source_p1_state_sha256": dgcaip["source_p1_state_sha256"],
        "source_p1_metrics_sha256": dgcaip["source_p1_metrics_sha256"],
        "output": str(output_path),
        "output_sha256": _file_sha256(output_path),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".binding.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    args = _arguments()
    report = bind_dgcaip_mechanism_config(
        Path(args.template),
        Path(args.d0_report),
        Path(args.output),
        d0_reference=args.d0_reference,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
