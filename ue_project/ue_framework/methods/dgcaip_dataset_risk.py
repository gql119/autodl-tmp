from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


RISK_BANK_SCHEMA = "tausb.dgcaip-dataset-risk-bank.v1"


@dataclass(frozen=True, order=True)
class DGCAIPInstanceKey:
    image_id: str
    gt_index: int
    class_id: int

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("DG-CAIP instance image_id must be non-empty.")
        if self.gt_index < 0 or self.class_id < 0:
            raise ValueError("DG-CAIP instance indices must be non-negative.")

    def as_tuple(self) -> Tuple[str, int, int]:
        return self.image_id, self.gt_index, self.class_id


@dataclass(frozen=True)
class DGCAIPRiskRecord:
    key: DGCAIPInstanceKey
    snapshot_id: str
    js_divergence: float
    clean_to_poison_kl: float

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("DG-CAIP risk snapshot_id must be non-empty.")
        values = (self.js_divergence, self.clean_to_poison_kl)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("DG-CAIP KL/JS risk values must be finite and non-negative.")


@dataclass(frozen=True)
class DGCAIPRiskEntry:
    key: DGCAIPInstanceKey
    risk: float
    mean_risk: float
    rank_variance: float
    snapshot_coverage: float
    snapshot_risks: Mapping[str, float]
    high_risk: bool


@dataclass(frozen=True)
class DGCAIPDatasetRiskBank:
    schema: str
    spec_id: str
    snapshot_ids: Tuple[str, ...]
    js_weight: float
    kl_weight: float
    top_fraction: float
    expected_instance_count: int
    covered_instance_count: int
    coverage: float
    class_counts: Mapping[int, int]
    entries: Tuple[DGCAIPRiskEntry, ...]
    canonical_sha256: str

    def rank_mapping(self) -> Dict[Tuple[str, int, int], float]:
        return {entry.key.as_tuple(): entry.risk for entry in self.entries}

    def high_risk_keys(self) -> Tuple[Tuple[str, int, int], ...]:
        return tuple(
            entry.key.as_tuple() for entry in self.entries if entry.high_risk
        )


def _stable_mid_ranks(
    values: Mapping[DGCAIPInstanceKey, float],
) -> Dict[DGCAIPInstanceKey, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    if count == 1:
        return {ordered[0][0]: 0.0}
    output: Dict[DGCAIPInstanceKey, float] = {}
    start = 0
    while start < count:
        end = start + 1
        while end < count and ordered[end][1] == ordered[start][1]:
            end += 1
        mid_position = 0.5 * (start + end - 1)
        percentile = mid_position / float(count - 1)
        for index in range(start, end):
            output[ordered[index][0]] = percentile
        start = end
    return output


def _canonical_payload(
    *,
    spec_id: str,
    snapshot_ids: Sequence[str],
    js_weight: float,
    kl_weight: float,
    top_fraction: float,
    expected_instance_count: int,
    covered_instance_count: int,
    coverage: float,
    class_counts: Mapping[int, int],
    entries: Sequence[DGCAIPRiskEntry],
) -> Dict[str, object]:
    return {
        "schema": RISK_BANK_SCHEMA,
        "spec_id": spec_id,
        "snapshot_ids": list(snapshot_ids),
        "js_weight": js_weight,
        "kl_weight": kl_weight,
        "top_fraction": top_fraction,
        "expected_instance_count": expected_instance_count,
        "covered_instance_count": covered_instance_count,
        "coverage": coverage,
        "class_counts": {
            str(class_id): int(class_counts[class_id])
            for class_id in sorted(class_counts)
        },
        "entries": [
            {
                "key": asdict(entry.key),
                "risk": entry.risk,
                "mean_risk": entry.mean_risk,
                "rank_variance": entry.rank_variance,
                "snapshot_coverage": entry.snapshot_coverage,
                "snapshot_risks": {
                    name: entry.snapshot_risks[name]
                    for name in sorted(entry.snapshot_risks)
                },
                "high_risk": entry.high_risk,
            }
            for entry in sorted(entries, key=lambda item: item.key)
        ],
    }


def _payload_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_dataset_risk_bank(
    records: Iterable[DGCAIPRiskRecord],
    *,
    spec_id: str,
    expected_snapshot_ids: Sequence[str],
    expected_instance_keys: Sequence[DGCAIPInstanceKey] | None = None,
    js_weight: float = 0.7,
    kl_weight: float = 0.3,
    top_fraction: float = 0.25,
    minimum_coverage: float = 0.90,
) -> DGCAIPDatasetRiskBank:
    """Build a frozen class-wise KL/JS risk bank from dataset-wide scans."""

    snapshot_ids = tuple(str(item) for item in expected_snapshot_ids)
    if not spec_id.strip() or not snapshot_ids or len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("Risk bank requires a SpecID and unique snapshot IDs.")
    if any(not item.strip() for item in snapshot_ids):
        raise ValueError("Risk bank snapshot IDs must be non-empty.")
    if js_weight < 0 or kl_weight < 0 or not math.isclose(
        js_weight + kl_weight, 1.0, abs_tol=1.0e-12
    ):
        raise ValueError("Risk bank KL/JS weights must be non-negative and sum to one.")
    if not 0 < top_fraction <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("Risk-bank fractions must lie in (0,1].")

    frozen_records = tuple(records)
    if not frozen_records:
        raise ValueError("Risk bank requires at least one record.")
    allowed_snapshots = set(snapshot_ids)
    by_snapshot_class: Dict[Tuple[str, int], list[DGCAIPRiskRecord]] = {}
    seen = set()
    for record in frozen_records:
        if record.snapshot_id not in allowed_snapshots:
            raise ValueError("Risk record references an unexpected snapshot.")
        identity = (record.snapshot_id, record.key)
        if identity in seen:
            raise ValueError("Duplicate DG-CAIP snapshot/instance risk record.")
        seen.add(identity)
        by_snapshot_class.setdefault(
            (record.snapshot_id, record.key.class_id), []
        ).append(record)

    snapshot_risk: Dict[Tuple[str, DGCAIPInstanceKey], float] = {}
    for (snapshot_id, _class_id), group in sorted(by_snapshot_class.items()):
        js_ranks = _stable_mid_ranks(
            {record.key: record.js_divergence for record in group}
        )
        kl_ranks = _stable_mid_ranks(
            {record.key: math.log1p(record.clean_to_poison_kl) for record in group}
        )
        for record in group:
            snapshot_risk[(snapshot_id, record.key)] = (
                js_weight * js_ranks[record.key]
                + kl_weight * kl_ranks[record.key]
            )

    keys = sorted({record.key for record in frozen_records})
    expected_input = (
        tuple(expected_instance_keys)
        if expected_instance_keys is not None
        else tuple(keys)
    )
    if len(expected_input) != len(set(expected_input)):
        raise ValueError("Expected DG-CAIP instance keys must be unique.")
    expected_keys = tuple(sorted(expected_input))
    unexpected = set(keys).difference(expected_keys)
    if unexpected:
        raise ValueError("Risk records contain unexpected instance keys.")
    coverage = len(keys) / float(len(expected_keys)) if expected_keys else 0.0
    if coverage + 1.0e-12 < minimum_coverage:
        raise ValueError("Dataset risk-bank instance coverage is below the gate.")

    provisional = []
    for key in keys:
        values = {
            snapshot_id: snapshot_risk[(snapshot_id, key)]
            for snapshot_id in snapshot_ids
            if (snapshot_id, key) in snapshot_risk
        }
        if not values:
            raise ValueError("Covered risk-bank key has no snapshot risk.")
        ordered_values = tuple(values[name] for name in sorted(values))
        mean = sum(ordered_values) / len(ordered_values)
        variance = sum((value - mean) ** 2 for value in ordered_values) / len(
            ordered_values
        )
        provisional.append(
            DGCAIPRiskEntry(
                key=key,
                risk=max(ordered_values),
                mean_risk=mean,
                rank_variance=variance,
                snapshot_coverage=len(values) / float(len(snapshot_ids)),
                snapshot_risks=values,
                high_risk=False,
            )
        )

    by_class: Dict[int, list[DGCAIPRiskEntry]] = {}
    for entry in provisional:
        by_class.setdefault(entry.key.class_id, []).append(entry)
    high_risk_keys = set()
    for class_entries in by_class.values():
        count = max(1, math.ceil(len(class_entries) * top_fraction))
        ordered = sorted(class_entries, key=lambda item: (-item.risk, item.key))
        high_risk_keys.update(entry.key for entry in ordered[:count])
    entries = tuple(
        DGCAIPRiskEntry(
            key=entry.key,
            risk=entry.risk,
            mean_risk=entry.mean_risk,
            rank_variance=entry.rank_variance,
            snapshot_coverage=entry.snapshot_coverage,
            snapshot_risks=entry.snapshot_risks,
            high_risk=entry.key in high_risk_keys,
        )
        for entry in sorted(provisional, key=lambda item: item.key)
    )
    class_counts = {
        class_id: len(class_entries)
        for class_id, class_entries in sorted(by_class.items())
    }
    payload = _canonical_payload(
        spec_id=spec_id,
        snapshot_ids=snapshot_ids,
        js_weight=js_weight,
        kl_weight=kl_weight,
        top_fraction=top_fraction,
        expected_instance_count=len(expected_keys),
        covered_instance_count=len(keys),
        coverage=coverage,
        class_counts=class_counts,
        entries=entries,
    )
    return DGCAIPDatasetRiskBank(
        schema=RISK_BANK_SCHEMA,
        spec_id=spec_id,
        snapshot_ids=snapshot_ids,
        js_weight=js_weight,
        kl_weight=kl_weight,
        top_fraction=top_fraction,
        expected_instance_count=len(expected_keys),
        covered_instance_count=len(keys),
        coverage=coverage,
        class_counts=class_counts,
        entries=entries,
        canonical_sha256=_payload_sha256(payload),
    )


def risk_bank_payload(bank: DGCAIPDatasetRiskBank) -> Dict[str, object]:
    payload = _canonical_payload(
        spec_id=bank.spec_id,
        snapshot_ids=bank.snapshot_ids,
        js_weight=bank.js_weight,
        kl_weight=bank.kl_weight,
        top_fraction=bank.top_fraction,
        expected_instance_count=bank.expected_instance_count,
        covered_instance_count=bank.covered_instance_count,
        coverage=bank.coverage,
        class_counts=bank.class_counts,
        entries=bank.entries,
    )
    actual_hash = _payload_sha256(payload)
    if actual_hash != bank.canonical_sha256:
        raise ValueError("DG-CAIP risk-bank canonical hash mismatch.")
    payload["canonical_sha256"] = actual_hash
    return payload


def write_risk_bank(path: Path, bank: DGCAIPDatasetRiskBank) -> None:
    path.write_text(
        json.dumps(
            risk_bank_payload(bank),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_risk_bank(
    path: Path,
    *,
    expected_spec_id: str | None = None,
    expected_sha256: str | None = None,
) -> DGCAIPDatasetRiskBank:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RISK_BANK_SCHEMA:
        raise ValueError("Unsupported DG-CAIP risk-bank schema.")
    recorded_hash = str(payload.pop("canonical_sha256", "")).lower()
    actual_hash = _payload_sha256(payload)
    if not recorded_hash or recorded_hash != actual_hash:
        raise ValueError("DG-CAIP risk-bank canonical hash mismatch.")
    if expected_sha256 is not None and actual_hash != expected_sha256.lower():
        raise ValueError("DG-CAIP risk-bank expected hash mismatch.")
    spec_id = str(payload.get("spec_id", ""))
    if expected_spec_id is not None and spec_id != expected_spec_id:
        raise ValueError("DG-CAIP risk-bank SpecID mismatch.")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("DG-CAIP risk bank contains no entries.")
    entries = []
    seen = set()
    for raw in raw_entries:
        key_payload = raw.get("key", {})
        key = DGCAIPInstanceKey(
            image_id=str(key_payload.get("image_id", "")),
            gt_index=int(key_payload.get("gt_index", -1)),
            class_id=int(key_payload.get("class_id", -1)),
        )
        if key in seen:
            raise ValueError("DG-CAIP risk bank contains duplicate instance keys.")
        seen.add(key)
        snapshot_risks = {
            str(name): float(value)
            for name, value in dict(raw.get("snapshot_risks", {})).items()
        }
        numeric = (
            float(raw.get("risk", float("nan"))),
            float(raw.get("mean_risk", float("nan"))),
            float(raw.get("rank_variance", float("nan"))),
            float(raw.get("snapshot_coverage", float("nan"))),
            *snapshot_risks.values(),
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("DG-CAIP risk-bank entry contains invalid values.")
        entries.append(
            DGCAIPRiskEntry(
                key=key,
                risk=numeric[0],
                mean_risk=numeric[1],
                rank_variance=numeric[2],
                snapshot_coverage=numeric[3],
                snapshot_risks=snapshot_risks,
                high_risk=bool(raw.get("high_risk", False)),
            )
        )
    bank = DGCAIPDatasetRiskBank(
        schema=RISK_BANK_SCHEMA,
        spec_id=spec_id,
        snapshot_ids=tuple(str(item) for item in payload["snapshot_ids"]),
        js_weight=float(payload["js_weight"]),
        kl_weight=float(payload["kl_weight"]),
        top_fraction=float(payload["top_fraction"]),
        expected_instance_count=int(payload["expected_instance_count"]),
        covered_instance_count=int(payload["covered_instance_count"]),
        coverage=float(payload["coverage"]),
        class_counts={
            int(class_id): int(count)
            for class_id, count in dict(payload["class_counts"]).items()
        },
        entries=tuple(sorted(entries, key=lambda item: item.key)),
        canonical_sha256=actual_hash,
    )
    risk_bank_payload(bank)
    return bank


def build_balanced_replay_image_ids(
    bank: DGCAIPDatasetRiskBank,
    all_image_ids: Sequence[str],
    *,
    total_slots: int,
    high_risk_fraction: float = 0.50,
    seed: int = 0,
) -> Tuple[str, ...]:
    """Build deterministic 50/50 uniform and high-risk replay slots."""

    unique_images = tuple(sorted(set(str(item) for item in all_image_ids)))
    if len(unique_images) != len(all_image_ids) or any(
        not item.strip() for item in unique_images
    ):
        raise ValueError("Replay source image IDs must be unique and non-empty.")
    if total_slots < 1 or not 0.0 <= high_risk_fraction <= 1.0:
        raise ValueError("Replay slot count or high-risk fraction is invalid.")
    allowed_images = set(unique_images)
    high_images = tuple(
        sorted(
            {
                entry.key.image_id
                for entry in bank.entries
                if entry.high_risk and entry.key.image_id in allowed_images
            }
        )
    )
    if not unique_images or not high_images:
        raise ValueError("Replay requires high-risk images inside the scan population.")

    def ordered(values: Sequence[str], stream: str) -> Tuple[str, ...]:
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    hashlib.sha256(
                        ("%d:%s:%s" % (seed, stream, item)).encode("utf-8")
                    ).hexdigest(),
                    item,
                ),
            )
        )

    uniform_order = ordered(unique_images, "uniform")
    high_order = ordered(high_images, "high")
    high_slots = int(round(total_slots * high_risk_fraction))
    uniform_slots = total_slots - high_slots
    uniform = tuple(
        uniform_order[index % len(uniform_order)] for index in range(uniform_slots)
    )
    high = tuple(high_order[index % len(high_order)] for index in range(high_slots))
    output = []
    for index in range(max(len(uniform), len(high))):
        if index < len(uniform):
            output.append(uniform[index])
        if index < len(high):
            output.append(high[index])
    if len(output) != total_slots:
        raise RuntimeError("Replay construction produced the wrong number of slots.")
    return tuple(output)
