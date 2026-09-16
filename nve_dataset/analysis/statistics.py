from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from nve_dataset.util import read_csv, write_csv, write_json


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def dominant_match_ratio(root: Path, interactions: list[dict[str, str]], manifest: dict[str, Any]) -> float | None:
    scenario = manifest["scenario"]
    parameters = manifest["scenario_parameters"]
    if scenario == "uniform":
        return None
    matches, eligible = 0, 0
    for row in interactions:
        timestamp, entity_id = float(row["timestamp"]), row["entity_id"]
        dominant: str | None = None
        if scenario == "concentrated":
            dominant = parameters["dominant_peer_by_entity"][entity_id]
        elif scenario == "shifting":
            for phase in parameters["phases"]:
                if phase["start"] <= timestamp < phase["end"]:
                    dominant = phase["dominant_peer_by_entity"][entity_id]
                    break
        elif scenario == "burst":
            if any(item["start"] <= timestamp < item["end"] for item in parameters["burst_intervals"]):
                dominant = parameters["dominant_peer_by_entity"][entity_id]
        if dominant is not None:
            eligible += 1
            matches += row["peer_id"] == dominant
    return matches / eligible if eligible else None


def generate_summary(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    import json
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    interactions = read_csv(root / "input" / "interactions.csv")
    network = read_csv(root / "input" / "network.csv")
    duration = float(manifest["duration"])
    rtts = [float(row["rtt_ms"]) for row in network if row["src_peer"] != row["dst_peer"]]
    peer_counts = Counter(row["peer_id"] for row in interactions)
    entity_counts = Counter(row["entity_id"] for row in interactions)
    type_counts = Counter(row["request_type"] for row in interactions)
    total = len(interactions)
    summary = {
        "dataset_id": manifest["dataset_id"],
        "total_requests": total,
        "requests_per_second": total / duration,
        "requests_per_peer": dict(sorted(peer_counts.items())),
        "requests_per_entity": dict(sorted(entity_counts.items())),
        "interaction_distribution": {
            key: {"count": value, "ratio": value / total if total else 0.0}
            for key, value in sorted(type_counts.items())
        },
        "dominant_peer_ratio_actual": dominant_match_ratio(root, interactions, manifest),
        "mean_rtt_ms": statistics.fmean(rtts) if rtts else 0.0,
        "median_rtt_ms": statistics.median(rtts) if rtts else 0.0,
        "p95_rtt_ms": _percentile(rtts, 0.95),
    }
    write_json(root / "analysis" / "summary.json", summary)
    return summary


AGGREGATE_COLUMNS = [
    "dataset_id", "scenario", "seed", "peer_count", "entity_count", "duration",
    "interaction_rate_per_entity", "network_model", "movement_model",
    "total_requests", "requests_per_second",
    "dominant_peer_ratio_expected", "dominant_peer_ratio_actual",
    "mean_rtt_ms", "median_rtt_ms", "p95_rtt_ms", "valid", "path",
]


def aggregate_summaries(roots: list[str | Path], output_path: str | Path) -> Path:
    """Collapse per-dataset summaries into one table for cross-condition tables."""
    import json
    rows = []
    for root in sorted(Path(item) for item in roots):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "analysis" / "summary.json").read_text(encoding="utf-8"))
        validation_path = root / "validation.json"
        valid = json.loads(validation_path.read_text(encoding="utf-8"))["valid"] if validation_path.exists() else ""
        rows.append({
            "dataset_id": manifest["dataset_id"],
            "scenario": manifest["scenario"],
            "seed": manifest["seed"],
            "peer_count": manifest["peer_count"],
            "entity_count": manifest["entity_count"],
            "duration": manifest["duration"],
            "interaction_rate_per_entity": manifest["interaction_rate_per_entity"],
            "network_model": manifest["network_model"],
            "movement_model": manifest["movement_model"],
            "total_requests": summary["total_requests"],
            "requests_per_second": f"{summary['requests_per_second']:.6f}",
            "dominant_peer_ratio_expected": manifest["scenario_parameters"].get("dominant_peer_ratio", ""),
            "dominant_peer_ratio_actual": (
                f"{summary['dominant_peer_ratio_actual']:.6f}"
                if summary["dominant_peer_ratio_actual"] is not None else ""
            ),
            "mean_rtt_ms": f"{summary['mean_rtt_ms']:.6f}",
            "median_rtt_ms": f"{summary['median_rtt_ms']:.6f}",
            "p95_rtt_ms": f"{summary['p95_rtt_ms']:.6f}",
            "valid": valid,
            "path": root.as_posix(),
        })
    output = Path(output_path)
    write_csv(output, AGGREGATE_COLUMNS, rows)
    return output
