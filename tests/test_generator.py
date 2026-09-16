from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import sha256_file
from nve_dataset.validation import check_reproducibility, validate_dataset


def compact_config() -> dict:
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    config["simulation"]["duration"] = 4.0
    config["simulation"]["timestep"] = 0.5
    config["peers"]["position_sample_interval"] = 0.5
    config["peers"]["count"] = 4
    config["entities"]["count"] = 2
    config["interaction"]["rate_per_entity"] = 10.0
    config["interaction"]["shifting"]["phase_duration"] = 1.0
    config["interaction"]["burst"]["intervals"] = [{"start": 1.0, "duration": 2.0}]
    config["validation"]["distribution_absolute_tolerance"] = 0.20
    return config


def test_all_scenarios_validate(tmp_path: Path) -> None:
    base = compact_config()
    for scenario in ("uniform", "concentrated", "shifting", "burst"):
        root = generate_dataset(config_for_run(base, scenario, 7), tmp_path / scenario, analyze=False)
        assert validate_dataset(root)["valid"]


def test_network_and_mobility_do_not_change_interactions(tmp_path: Path) -> None:
    base = config_for_run(compact_config(), "concentrated", 11)
    changed = copy.deepcopy(base)
    changed["network"]["model"] = "homogeneous"
    changed["peers"]["movement_model"] = "static"
    first = generate_dataset(base, tmp_path / "first", analyze=False)
    second = generate_dataset(changed, tmp_path / "second", analyze=False)
    assert sha256_file(first / "input" / "interactions.csv") == sha256_file(second / "input" / "interactions.csv")


def test_scenarios_share_arrivals_types_and_weights(tmp_path: Path) -> None:
    base = compact_config()
    traces = []
    for scenario in ("uniform", "concentrated", "shifting", "burst"):
        root = generate_dataset(config_for_run(base, scenario, 13), tmp_path / scenario, analyze=False)
        with (root / "input" / "interactions.csv").open(newline="", encoding="utf-8") as stream:
            traces.append([
                (row["timestamp"], row["request_id"], row["entity_id"], row["request_type"])
                for row in csv.DictReader(stream)
            ])
    assert all(trace == traces[0] for trace in traces[1:])


def test_reproducibility_hashes(tmp_path: Path) -> None:
    root = generate_dataset(config_for_run(compact_config(), "uniform", 3), tmp_path / "source", analyze=False)
    assert check_reproducibility(root) == {"reproducible": True, "different_files": []}


def test_manifest_has_all_input_hashes(tmp_path: Path) -> None:
    root = generate_dataset(config_for_run(compact_config(), "burst", 5), tmp_path / "source", analyze=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["input_sha256"]) == {
        "peers.csv", "entities.csv", "network.csv", "peer_positions.csv",
        "interactions.csv", "request_types.csv",
    }


def test_interaction_weight_change_leaves_trace_untouched(tmp_path: Path) -> None:
    base = config_for_run(compact_config(), "concentrated", 17)
    reweighted = copy.deepcopy(base)
    for entry in reweighted["interaction"]["request_types"].values():
        entry["weight"] = float(entry["weight"]) * 7.5
    first = generate_dataset(base, tmp_path / "first", analyze=False)
    second = generate_dataset(reweighted, tmp_path / "second", analyze=False)
    assert sha256_file(first / "input" / "interactions.csv") == sha256_file(second / "input" / "interactions.csv")
    assert sha256_file(first / "input" / "request_types.csv") != sha256_file(second / "input" / "request_types.csv")


def test_static_peers_record_a_single_position_sample(tmp_path: Path) -> None:
    config = compact_config()
    config["peers"]["movement_model"] = "static"
    root = generate_dataset(config_for_run(config, "uniform", 19), tmp_path / "static", analyze=False)
    rows = list(csv.DictReader((root / "input" / "peer_positions.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == config["peers"]["count"]
    assert {row["timestamp"] for row in rows} == {"0.000000"}
    assert validate_dataset(root)["valid"]


def test_position_sampling_is_coarser_than_the_integration_step(tmp_path: Path) -> None:
    config = compact_config()
    config["simulation"]["timestep"] = 0.5
    config["peers"]["position_sample_interval"] = 2.0  # duration is 4.0s -> t = 0, 2, 4
    root = generate_dataset(config_for_run(config, "uniform", 23), tmp_path / "sampled", analyze=False)
    rows = list(csv.DictReader((root / "input" / "peer_positions.csv").open(newline="", encoding="utf-8")))
    assert sorted({float(row["timestamp"]) for row in rows}) == [0.0, 2.0, 4.0]
    assert validate_dataset(root)["valid"]
