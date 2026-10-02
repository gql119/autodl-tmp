"""Compute the preregistered P/U/S and collateral comparison from AP50 JSONs."""
import argparse
import json
from pathlib import Path


NAMES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def read_metrics(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if "primary" in value:
        return value["primary"]
    return value


def pu_s(metrics, clean):
    p = (clean["mAP50_target"] - metrics["mAP50_target"]) / clean["mAP50_target"]
    u = metrics["mAP50_non_target"] / clean["mAP50_non_target"]
    score = 2 * p * u / (p + u) if p + u else None
    return {"P": p, "U": u, "S": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--formal-b2-to", required=True)
    parser.add_argument("--safe-last", required=True)
    parser.add_argument("--safe-best", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean, formal, safe_last, safe_best = map(read_metrics,
        (args.clean, args.formal_b2_to, args.safe_last, args.safe_best))
    if any(len(item["AP50_per_class"]) != 20 for item in (clean, formal, safe_last, safe_best)):
        raise RuntimeError("All comparisons require complete VOC20 per-class AP50 results")
    nt_ids = [i for i in range(20) if i != 14]
    dc = {NAMES[i]: clean["AP50_per_class"][i] - safe_last["AP50_per_class"][i] for i in nt_ids}
    sorted_drops = sorted(dc.items(), key=lambda item: (-item[1], item[0]))
    result = {
        "primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
        "B2-TO-Safe_last": safe_last, "B2-TO-Safe_best": safe_best,
        "clean_baseline": clean, "formal_B2_TO": formal,
        "PUS": {"B2-TO": pu_s(formal, clean), "B2-TO-Safe_last": pu_s(safe_last, clean),
                "B2-TO-Safe_best_secondary": pu_s(safe_best, clean)},
        "collateral_damage": {
            "definition": "D_c = clean AP50_c - method AP50_c for each non-target class; WCD=max D_c; TCD3=mean of the three largest D_c.",
            "D_c": dc, "WCD": sorted_drops[0][1],
            "WCD_class": sorted_drops[0][0], "TCD3": sum(x[1] for x in sorted_drops[:3]) / 3,
            "top_three": sorted_drops[:3],
            "bicycle_dog_horse": {name: dc[name] for name in ("bicycle", "dog", "horse")},
        },
        "per_class_delta_vs_formal_B2_TO": {
            "AP50": {NAMES[i]: safe_last["AP50_per_class"][i] - formal["AP50_per_class"][i] for i in range(20)},
            "Recall": {NAMES[i]: safe_last["Recall_per_class"][i] - formal["Recall_per_class"][i] for i in range(20)},
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
