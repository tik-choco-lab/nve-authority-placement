from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from nve_dataset.analysis.statistics import dominant_match_ratio
from nve_dataset.util import read_csv, sha256_file


def validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((root / "config_effective.yaml").read_text(encoding="utf-8"))
    peers = read_csv(root / "input" / "peers.csv")
    entities = read_csv(root / "input" / "entities.csv")
    network = read_csv(root / "input" / "network.csv")
    positions = read_csv(root / "input" / "peer_positions.csv")
    interactions = read_csv(root / "input" / "interactions.csv")
    request_types = read_csv(root / "input" / "request_types.csv")
    peer_ids, entity_ids = {r["peer_id"] for r in peers}, {r["entity_id"] for r in entities}

    timestamps = [float(row["timestamp"]) for row in interactions]
    if timestamps != sorted(timestamps):
        errors.append("interaction timestamps are not monotonically nondecreasing")
    if any(timestamp < 0 or timestamp >= float(manifest["duration"]) for timestamp in timestamps):
        errors.append("interaction timestamp outside [0, duration)")
    request_ids = [row["request_id"] for row in interactions]
    if len(request_ids) != len(set(request_ids)):
        errors.append("request_id values are not unique")
    if any(row["peer_id"] not in peer_ids for row in interactions):
        errors.append("interaction references unknown peer_id")
    if any(row["entity_id"] not in entity_ids for row in interactions):
        errors.append("interaction references unknown entity_id")
    if any(row["initial_authority"] not in peer_ids for row in entities):
        errors.append("entity initial_authority references unknown peer")
    if any(float(row["rtt_ms"]) < 0 for row in network):
        errors.append("negative RTT found")
    pairs = {(row["src_peer"], row["dst_peer"]) for row in network}
    expected_pairs = {(src, dst) for src in peer_ids for dst in peer_ids}
    if pairs != expected_pairs:
        errors.append("network matrix is incomplete or references unknown peers")
    if any(float(row["rtt_ms"]) != 0 for row in network if row["src_peer"] == row["dst_peer"]):
        errors.append("self RTT must be zero")
    if config["network"].get("symmetric", True):
        rtt_map = {(r["src_peer"], r["dst_peer"]): float(r["rtt_ms"]) for r in network}
        if any(rtt_map[src, dst] != rtt_map[dst, src] for src, dst in expected_pairs):
            errors.append("RTT matrix is not symmetric")
    if any(row["peer_id"] not in peer_ids for row in positions):
        errors.append("position trace references unknown peer")
    if any(not (0 <= float(row["x"]) <= float(config["world"]["width"]) and 0 <= float(row["y"]) <= float(config["world"]["height"])) for row in positions):
        errors.append("peer position outside world bounds")
    if len(positions) != int(manifest["position_sample_count"]) * len(peer_ids):
        errors.append("position trace does not cover every peer at every recorded sample")
    if {row["timestamp"] for row in positions} and min(float(row["timestamp"]) for row in positions) != 0.0:
        errors.append("position trace must start at t=0 so replay can hold the last known value")
    declared = {row["request_type"]: float(row["interaction_weight"]) for row in request_types}
    configured = {name: float(value["weight"]) for name, value in config["interaction"]["request_types"].items()}
    if len(declared) != len(request_types):
        errors.append("request_types.csv contains duplicate request_type rows")
    if declared != configured:
        errors.append("request_types.csv does not match configured interaction weights")
    if any(row["request_type"] not in declared for row in interactions):
        errors.append("interaction references a request_type absent from request_types.csv")

    scenario = manifest["scenario"]
    observed: float | None = None
    expected: float | None = None
    sample_size = len(interactions)
    if scenario == "uniform" and interactions:
        expected = 1 / len(peer_ids)
        counts = Counter(row["peer_id"] for row in interactions)
        observed = max(counts.values()) / len(interactions)
        tolerance = _tolerance(expected, len(interactions), config)
        if any(abs(counts[peer] / len(interactions) - expected) > tolerance for peer in peer_ids):
            errors.append("uniform peer distribution exceeds statistical tolerance")
    elif scenario in {"concentrated", "shifting", "burst"}:
        observed = dominant_match_ratio(root, interactions, manifest)
        expected = float(manifest["scenario_parameters"]["dominant_peer_ratio"])
        if scenario == "burst":
            intervals = manifest["scenario_parameters"]["burst_intervals"]
            sample_size = sum(any(i["start"] <= float(r["timestamp"]) < i["end"] for i in intervals) for r in interactions)
        if observed is None:
            warnings.append("no eligible interactions for dominant-ratio validation")
        elif abs(observed - expected) > _tolerance(expected, sample_size, config):
            errors.append("dominant peer ratio exceeds statistical tolerance")

    for name, expected_hash in manifest["input_sha256"].items():
        if sha256_file(root / "input" / name) != expected_hash:
            errors.append(f"input checksum mismatch: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "peer_count": len(peer_ids),
            "entity_count": len(entity_ids),
            "interaction_count": len(interactions),
            "scenario_distribution_expected": expected,
            "scenario_distribution_observed": observed,
            "scenario_distribution_sample_size": sample_size,
        },
    }


def _tolerance(probability: float, sample_size: int, config: dict[str, Any]) -> float:
    if sample_size <= 0:
        return 1.0
    sigma = float(config["validation"]["distribution_sigma"])
    absolute = float(config["validation"]["distribution_absolute_tolerance"])
    return absolute + sigma * math.sqrt(probability * (1 - probability) / sample_size)


def check_reproducibility(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    config = yaml.safe_load((root / "config_effective.yaml").read_text(encoding="utf-8"))
    original = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["input_sha256"]
    from nve_dataset.generator.core import generate_dataset
    with tempfile.TemporaryDirectory(prefix="nve-reproduce-") as temporary:
        regenerated_root = generate_dataset(config, Path(temporary) / "dataset", analyze=False, validate=False)
        regenerated = json.loads((regenerated_root / "manifest.json").read_text(encoding="utf-8"))["input_sha256"]
    differences = sorted(name for name in original if original[name] != regenerated.get(name))
    return {"reproducible": not differences, "different_files": differences}
