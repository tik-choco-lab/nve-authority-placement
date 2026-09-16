from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nve_dataset.analysis.statistics import aggregate_summaries, generate_summary
from nve_dataset.analysis.visualization import generate_visualizations
from nve_dataset.config import config_for_run
from nve_dataset.mobility import generate_peer_positions, peer_ids_for, sample_steps
from nve_dataset.network import generate_network
from nve_dataset.scenario import build_pattern
from nve_dataset.util import rng_for, sha256_file, weighted_choice, write_csv, write_json
from nve_dataset.validation.validator import validate_dataset


def generate_dataset(config: dict[str, Any], output_dir: str | Path | None = None,
                     analyze: bool = True, validate: bool = True) -> Path:
    scenario, seed = config["scenario"], int(config["seed"])
    root = Path(output_dir) if output_dir else (
        Path(config["output_root"]) / config["dataset_prefix"] / scenario / f"seed_{seed:03d}"
    )
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)

    # peer_ids needs no RNG (it is a pure function of peers.count), so it can be
    # computed before positions exist. That lets entities and the scenario
    # pattern - both required by generate_peer_positions when
    # peers.target_coupling is enabled - be built first, without perturbing
    # anything: every generator call below draws from its own SHA-256-derived
    # namespace (nve_dataset/util.py rng_for), so reordering these calls does
    # not change any of their outputs (see docs/DATASET_DESIGN.md section 3).
    peer_ids = peer_ids_for(int(config["peers"]["count"]))
    entities = _generate_entities(config, peer_ids)
    entity_ids = [row["entity_id"] for row in entities]
    network = generate_network(config, peer_ids)
    pattern = build_pattern(config, peer_ids, entity_ids)
    peers, positions = generate_peer_positions(config, entities=entities, pattern=pattern)
    interactions = _generate_interactions(config, pattern, entity_ids)

    write_csv(input_dir / "peers.csv",
              ["peer_id", "initial_x", "initial_y", "movement_model", "movement_speed"], peers)
    write_csv(input_dir / "entities.csv",
              ["entity_id", "initial_x", "initial_y", "initial_authority", "state_size_bytes"], entities)
    write_csv(input_dir / "network.csv", ["src_peer", "dst_peer", "rtt_ms"], network)
    write_csv(input_dir / "peer_positions.csv", ["timestamp", "peer_id", "x", "y"], positions)
    write_csv(input_dir / "interactions.csv",
              ["timestamp", "request_id", "entity_id", "peer_id", "request_type"], interactions)
    # Interaction weight is an undecided research variable, so it is kept out of the
    # trace: changing a weight must not change interactions.csv byte-for-byte.
    write_csv(input_dir / "request_types.csv",
              ["request_type", "interaction_weight"], _request_type_rows(config))
    effective_path = root / "config_effective.yaml"
    effective_path.write_text(yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8")

    input_files = ["peers.csv", "entities.csv", "network.csv", "peer_positions.csv",
                   "interactions.csv", "request_types.csv"]
    manifest = _build_manifest(config, pattern, root, input_files)
    write_json(root / "manifest.json", manifest)
    (root / "output" / "README.txt").write_text(
        "Authority-placement policy outputs belong here. Input traces must not be modified.\n",
        encoding="utf-8",
    )

    if analyze:
        generate_summary(root)
        generate_visualizations(root)
    if validate:
        report = validate_dataset(root)
        write_json(root / "validation.json", report)
        if not report["valid"]:
            raise ValueError(f"dataset validation failed: {root}: {report['errors']}")
    return root


def generate_suite(base_config: dict[str, Any], analyze: bool = True, validate: bool = True) -> list[Path]:
    generated = []
    for scenario in base_config["suite"]["scenarios"]:
        for seed in base_config["suite"]["seeds"]:
            generated.append(generate_dataset(config_for_run(base_config, scenario, seed), analyze=analyze, validate=validate))
    if analyze and generated:
        suite_root = Path(base_config["output_root"]) / base_config["dataset_prefix"]
        aggregate_summaries(generated, suite_root / "summary_all.csv")
    return generated


def _request_type_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    types = config["interaction"]["request_types"]
    return [
        {"request_type": name, "interaction_weight": f"{float(types[name]['weight']):.6f}"}
        for name in sorted(types)
    ]


def _generate_entities(config: dict[str, Any], peer_ids: list[str]) -> list[dict[str, Any]]:
    rng = rng_for(config["seed"], "entity-initial-state")
    width, height = float(config["world"]["width"]), float(config["world"]["height"])
    count = int(config["entities"]["count"])
    size = int(config["migration"]["state_size_bytes"])
    rows = []
    for index in range(count):
        rows.append({
            "entity_id": f"E{index + 1:04d}",
            "initial_x": f"{rng.uniform(0, width):.6f}",
            "initial_y": f"{rng.uniform(0, height):.6f}",
            # Initial authority is neutral random input, not a policy decision.
            "initial_authority": rng.choice(peer_ids),
            "state_size_bytes": size,
        })
    return rows


def _event_times(config: dict[str, Any], entity_id: str) -> list[float]:
    rng = rng_for(config["seed"], f"interaction-arrivals-{entity_id}")
    duration = float(config["simulation"]["duration"])
    rate = float(config["interaction"]["rate_per_entity"])
    model = config["interaction"]["model"]
    times: list[float] = []
    if model == "fixed_interval":
        timestamp = 1.0 / rate
        while timestamp < duration:
            times.append(timestamp)
            timestamp += 1.0 / rate
    else:
        timestamp = rng.expovariate(rate)
        while timestamp < duration:
            times.append(timestamp)
            timestamp += rng.expovariate(rate)
    return times


def _generate_interactions(config: dict[str, Any], pattern: Any, entity_ids: list[str]) -> list[dict[str, Any]]:
    type_config = config["interaction"]["request_types"]
    # Canonical order keeps traces stable after YAML round-trips or mapping reordering.
    type_names = sorted(type_config)
    probabilities = [float(type_config[name]["probability"]) for name in type_names]
    raw: list[tuple[float, str, str, str]] = []
    for entity_id in entity_ids:
        peer_rng = rng_for(config["seed"], f"interaction-peers-{config['scenario']}-{entity_id}")
        type_rng = rng_for(config["seed"], f"interaction-types-{entity_id}")
        for timestamp in _event_times(config, entity_id):
            peer_id = pattern.choose_peer(entity_id, timestamp, peer_rng)
            request_type = weighted_choice(type_rng, type_names, probabilities)
            raw.append((timestamp, entity_id, peer_id, request_type))
    raw.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        {
            "timestamp": f"{timestamp:.6f}",
            "request_id": f"R{index:09d}",
            "entity_id": entity_id,
            "peer_id": peer_id,
            "request_type": request_type,
        }
        for index, (timestamp, entity_id, peer_id, request_type) in enumerate(raw, start=1)
    ]


def _build_manifest(config: dict[str, Any], pattern: Any, root: Path, input_files: list[str]) -> dict[str, Any]:
    network = config["network"]
    model = network["model"]
    return {
        "dataset_id": f"{config['dataset_prefix']}-{config['scenario']}-seed-{config['seed']:03d}",
        "dataset_kind": "controlled synthetic workload",
        "generator_version": config["generator_version"],
        "seed": config["seed"],
        "created_at": config["created_at"],
        "scenario": config["scenario"],
        "peer_count": config["peers"]["count"],
        "entity_count": config["entities"]["count"],
        "duration": config["simulation"]["duration"],
        "timestep": config["simulation"]["timestep"],
        "position_sample_interval": config["peers"]["position_sample_interval"],
        "position_sample_count": len(sample_steps(config)),
        "interaction_model": config["interaction"]["model"],
        "interaction_rate_per_entity": config["interaction"]["rate_per_entity"],
        "network_model": model,
        "rtt_range_ms": [network[f"{model}_rtt_min_ms"], network[f"{model}_rtt_max_ms"]],
        "movement_model": config["peers"]["movement_model"],
        "processing_delay_ms": config["processing"]["delay_ms"],
        "state_size_bytes": config["migration"]["state_size_bytes"],
        "scenario_parameters": pattern.metadata(float(config["simulation"]["duration"])),
        "input_sha256": {name: sha256_file(root / "input" / name) for name in input_files},
    }
