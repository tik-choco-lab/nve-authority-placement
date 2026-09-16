from __future__ import annotations

import copy
import json
from pathlib import Path

from nve_dataset.config import config_for_run
from nve_dataset.generator import generate_dataset
from nve_dataset.util import sha256_file
from tests.test_generator import compact_config

INPUT_FILES = ("peers.csv", "entities.csv", "network.csv", "peer_positions.csv",
               "interactions.csv", "request_types.csv")


def test_shared_dominant_absent_is_byte_identical_to_off(tmp_path: Path) -> None:
    """Regression: the option did not exist before this change, so its absence
    (as in every dataset generated so far) must reproduce the exact same bytes
    as explicitly turning it off."""
    base = compact_config()
    off = copy.deepcopy(base)
    off["interaction"]["shared_dominant_peer"] = False
    for scenario in ("uniform", "concentrated", "shifting", "burst"):
        absent = generate_dataset(config_for_run(base, scenario, 41), tmp_path / f"{scenario}-absent", analyze=False)
        explicit = generate_dataset(config_for_run(off, scenario, 41), tmp_path / f"{scenario}-off", analyze=False)
        for name in INPUT_FILES:
            assert sha256_file(absent / "input" / name) == sha256_file(explicit / "input" / name), name


def test_shared_dominant_collapses_every_entity_onto_one_peer(tmp_path: Path) -> None:
    base = compact_config()
    shared = copy.deepcopy(base)
    shared["interaction"]["shared_dominant_peer"] = True

    off_root = generate_dataset(config_for_run(base, "concentrated", 41), tmp_path / "off", analyze=False)
    on_root = generate_dataset(config_for_run(shared, "concentrated", 41), tmp_path / "on", analyze=False)

    def dominant_map(root: Path) -> dict[str, str]:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return manifest["scenario_parameters"]["dominant_peer_by_entity"]

    off_dominant = dominant_map(off_root)
    on_dominant = dominant_map(on_root)
    assert len(off_dominant) > 1, "test needs more than one entity to be meaningful"
    assert len(set(off_dominant.values())) == len(off_dominant), "default assignment must be distinct"
    assert len(set(on_dominant.values())) == 1, "shared assignment must collapse onto one peer"
    # The peer kept is the one the round robin already gave the first entity,
    # so enabling the option never introduces a new peer into the workload.
    first_entity = sorted(off_dominant)[0]
    assert set(on_dominant.values()) == {off_dominant[first_entity]}
