from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


SCENARIOS = {"uniform", "concentrated", "shifting", "burst"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(config)
    return config


def config_for_run(base: dict[str, Any], scenario: str, seed: int) -> dict[str, Any]:
    result = copy.deepcopy(base)
    result["scenario"] = scenario
    result["seed"] = int(seed)
    validate_config(result, require_run=True)
    return result


def _validate_target_coupling(config: dict[str, Any]) -> None:
    """`peers.target_coupling` is OPT-IN and OFF by default: an absent section,
    or `enabled: false`, means nothing below is even inspected. Once enabled,
    it is a research variable exactly like the other knobs in this file (see
    docs/POLICY_EVALUATION.md section 2), so its parameter gets no hidden
    default - a missing `bias_probability` is a ValueError, not a silent 0.0.
    """
    coupling = config["peers"].get("target_coupling", {})
    if not coupling.get("enabled", False):
        return
    if config["peers"]["movement_model"] != "random_walk":
        raise ValueError(
            "peers.target_coupling.enabled requires peers.movement_model == 'random_walk' "
            "(static peers never move, so there is nothing to bias)")
    if "bias_probability" not in coupling:
        raise ValueError("peers.target_coupling.bias_probability must be set explicitly when enabled")
    bias_probability = float(coupling["bias_probability"])
    if not 0.0 <= bias_probability <= 1.0:
        raise ValueError("peers.target_coupling.bias_probability must be in [0, 1]")


def validate_config(config: dict[str, Any], require_run: bool = False) -> None:
    if require_run and config.get("scenario") not in SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
    positive = [
        ("simulation.duration", config["simulation"]["duration"]),
        ("simulation.timestep", config["simulation"]["timestep"]),
        ("world.width", config["world"]["width"]),
        ("world.height", config["world"]["height"]),
        ("peers.count", config["peers"]["count"]),
        ("entities.count", config["entities"]["count"]),
        ("peers.position_sample_interval", config["peers"]["position_sample_interval"]),
        ("interaction.rate_per_entity", config["interaction"]["rate_per_entity"]),
        ("migration.state_size_bytes", config["migration"]["state_size_bytes"]),
    ]
    for name, value in positive:
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if config["peers"]["movement_model"] not in {"static", "random_walk"}:
        raise ValueError("unsupported movement model")
    _validate_target_coupling(config)
    timestep = float(config["simulation"]["timestep"])
    sample_interval = float(config["peers"]["position_sample_interval"])
    if sample_interval < timestep:
        raise ValueError("peers.position_sample_interval must be >= simulation.timestep")
    if abs(sample_interval / timestep - round(sample_interval / timestep)) > 1e-9:
        raise ValueError("peers.position_sample_interval must be a multiple of simulation.timestep")
    if config["interaction"]["model"] not in {"poisson", "fixed_interval"}:
        raise ValueError("unsupported interaction model")
    if config["network"]["model"] not in {"homogeneous", "heterogeneous"}:
        raise ValueError("unsupported network model")
    types = config["interaction"]["request_types"]
    probability_sum = sum(float(item["probability"]) for item in types.values())
    if abs(probability_sum - 1.0) > 1e-9:
        raise ValueError("request type probabilities must sum to 1")
    for section in ("concentrated", "shifting", "burst"):
        ratio = float(config["interaction"][section]["dominant_peer_ratio"])
        if not 0 <= ratio <= 1:
            raise ValueError(f"{section}.dominant_peer_ratio must be in [0, 1]")
    duration = float(config["simulation"]["duration"])
    for interval in config["interaction"]["burst"]["intervals"]:
        if interval["start"] < 0 or interval["duration"] <= 0 or interval["start"] + interval["duration"] > duration:
            raise ValueError("burst intervals must lie inside simulation duration")
