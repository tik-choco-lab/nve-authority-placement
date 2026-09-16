from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from nve_dataset.config import config_for_run, validate_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import read_csv, sha256_file
from tests.test_generator import compact_config


def coupled_config(bias_probability: float = 1.0) -> dict:
    config = compact_config()
    config["peers"]["target_coupling"] = {"enabled": True, "bias_probability": bias_probability}
    return config


def test_target_coupling_absent_is_byte_identical_to_no_coupling(tmp_path: Path) -> None:
    """Regression: the option did not exist before this change, so its absence
    (as in every dataset generated so far) must reproduce the exact same
    bytes as explicitly turning it off."""
    base = compact_config()
    off = copy.deepcopy(base)
    off["peers"]["target_coupling"] = {"enabled": False}
    for scenario in ("uniform", "concentrated", "shifting", "burst"):
        absent_root = generate_dataset(config_for_run(base, scenario, 41), tmp_path / f"{scenario}-absent", analyze=False)
        off_root = generate_dataset(config_for_run(off, scenario, 41), tmp_path / f"{scenario}-off", analyze=False)
        for name in ("peers.csv", "entities.csv", "network.csv", "peer_positions.csv",
                     "interactions.csv", "request_types.csv"):
            assert sha256_file(absent_root / "input" / name) == sha256_file(off_root / "input" / name), name


# Captured from generate_dataset(config_for_run(compact_config(), "shifting", 41), ...)
# on the generator as it existed IMMEDIATELY BEFORE peers.target_coupling was
# added (i.e. before nve_dataset/generator/core.py started reordering entity/
# pattern/position generation and before generate_peer_positions gained the
# entities/pattern parameters). Pinned literally, not recomputed, so that a
# silent behavioural change in the reorder itself would fail this test even
# if some other test happened to compare two post-change runs to each other.
PRE_COUPLING_SHIFTING_SEED_41_HASHES = {
    "peers.csv": "e3fb595426a05a269d5dac54d4d780f23fbc6da9faf0e10609607a46bc4b13b9",
    "entities.csv": "e28eaaea4b99e79910f329393a9969ff0af76a9acb5d9246bfbfd5cc5147c655",
    "network.csv": "658a6aef8b6dd2a56546d50e71341929af8f7ad9ee5506fcc7a5d24f5de7ab60",
    "peer_positions.csv": "a7f500718d4fc9313e1cf2d4d1c259ea56659b5a4cfb19fd8a6e59223c78c417",
    "interactions.csv": "7cd400fcbec52ba3a70025597c6af3990257d53ae258ced34abd64ae6ed59af6",
    "request_types.csv": "fac7e96b3404c3300541b57f6e74677025cfc673c899863fac984c1dff12c103",
}


def test_target_coupling_off_matches_pre_existing_default_dataset(tmp_path: Path) -> None:
    """Pinned regression: a dataset generated today, with the option absent,
    must still match the exact bytes the generator produced BEFORE this
    option existed (see the task's step 2 requirement that every existing
    dataset stay byte-for-byte reproducible). Unlike a same-run-vs-same-run
    comparison, this catches a regression introduced by the entities/pattern
    reordering in generator/core.py even if it were somehow deterministic.
    """
    base = config_for_run(compact_config(), "shifting", 41)
    root = generate_dataset(base, tmp_path / "pinned", analyze=False)
    for name, expected in PRE_COUPLING_SHIFTING_SEED_41_HASHES.items():
        assert sha256_file(root / "input" / name) == expected, name


def test_target_coupling_requires_random_walk(tmp_path: Path) -> None:
    config = coupled_config()
    config["peers"]["movement_model"] = "static"
    with pytest.raises(ValueError, match="random_walk"):
        validate_config(config_for_run(config, "shifting", 1), require_run=True)


def test_target_coupling_requires_explicit_bias_probability() -> None:
    config = compact_config()
    config["peers"]["target_coupling"] = {"enabled": True}
    with pytest.raises(ValueError, match="explicitly"):
        validate_config(config_for_run(config, "shifting", 1), require_run=True)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_target_coupling_bias_probability_must_be_in_unit_interval(bad: float) -> None:
    config = coupled_config(bad)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_config(config_for_run(config, "shifting", 1), require_run=True)


def test_target_coupling_does_not_change_interaction_trace(tmp_path: Path) -> None:
    """The whole point of nve_dataset/generator/core.py separating interaction
    generation from mobility (docs/DATASET_DESIGN.md section 3) is that
    turning this on must not perturb interactions.csv, exactly like the
    existing network/mobility independence test in test_generator.py."""
    base = config_for_run(compact_config(), "shifting", 29)
    coupled = config_for_run(coupled_config(), "shifting", 29)
    plain_root = generate_dataset(base, tmp_path / "plain", analyze=False)
    coupled_root = generate_dataset(coupled, tmp_path / "coupled", analyze=False)
    assert sha256_file(plain_root / "input" / "interactions.csv") == \
        sha256_file(coupled_root / "input" / "interactions.csv")
    # But positions themselves must actually differ - otherwise bias_probability=1.0 did nothing.
    assert sha256_file(plain_root / "input" / "peer_positions.csv") != \
        sha256_file(coupled_root / "input" / "peer_positions.csv")


def test_target_coupling_pulls_dominant_peer_toward_its_entity(tmp_path: Path) -> None:
    """Direct geometric check: with bias_probability=1.0 every mobility step of
    a peer that is currently a shifting-scenario dominant peer heads exactly
    at the centroid of the entities it is dominant for. Over many steps that
    must land the peer much closer to its target, on average, than the
    uncoupled random walk does.
    """
    config = compact_config()
    config["simulation"]["duration"] = 20.0
    config["simulation"]["timestep"] = 0.5
    config["peers"]["position_sample_interval"] = 0.5
    config["peers"]["count"] = 8
    config["entities"]["count"] = 2
    config["interaction"]["shifting"]["phase_duration"] = 5.0
    plain = config_for_run(copy.deepcopy(config), "shifting", 5)
    coupled_cfg = copy.deepcopy(config)
    coupled_cfg["peers"]["target_coupling"] = {"enabled": True, "bias_probability": 1.0}
    coupled = config_for_run(coupled_cfg, "shifting", 5)

    plain_root = generate_dataset(plain, tmp_path / "plain2", analyze=False)
    coupled_root = generate_dataset(coupled, tmp_path / "coupled2", analyze=False)

    def final_distance_to_dominant(root: Path) -> float:
        import json
        import yaml
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entities = {row["entity_id"]: (float(row["initial_x"]), float(row["initial_y"]))
                    for row in read_csv(root / "input" / "entities.csv")}
        positions = read_csv(root / "input" / "peer_positions.csv")
        last_ts = max(float(row["timestamp"]) for row in positions)
        last = {row["peer_id"]: (float(row["x"]), float(row["y"]))
                for row in positions if float(row["timestamp"]) == last_ts}
        dominant_by_entity = manifest["scenario_parameters"]["phases"][-1]["dominant_peer_by_entity"]
        total = 0.0
        for entity_id, peer_id in dominant_by_entity.items():
            ex, ey = entities[entity_id]
            px, py = last[peer_id]
            total += math.hypot(px - ex, py - ey)
        return total / len(dominant_by_entity)

    plain_distance = final_distance_to_dominant(plain_root)
    coupled_distance = final_distance_to_dominant(coupled_root)
    assert coupled_distance < plain_distance
