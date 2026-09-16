from __future__ import annotations

import math
from typing import Any

from nve_dataset.util import rng_for


def peer_ids_for(count: int) -> list[str]:
    """Canonical peer-id sequence, shared so callers can compute it before
    (or without) generating any positions."""
    return [f"P{index + 1:04d}" for index in range(count)]


def sample_steps(config: dict[str, Any]) -> list[int]:
    """Simulation steps at which a peer position is recorded.

    The walk advances every ``simulation.timestep`` but is only sampled every
    ``peers.position_sample_interval`` seconds, so trace resolution is decoupled
    from integration resolution. Static peers never move, so one sample suffices.
    """
    timestep = float(config["simulation"]["timestep"])
    step_count = round(float(config["simulation"]["duration"]) / timestep)
    if config["peers"]["movement_model"] == "static":
        return [0]
    stride = max(1, round(float(config["peers"]["position_sample_interval"]) / timestep))
    return list(range(0, step_count + 1, stride))


def _current_targets(pattern: Any, entity_positions: dict[str, tuple[float, float]],
                      timestamp: float) -> dict[str, tuple[float, float]]:
    """peer_id -> centroid of the entities `pattern` currently calls that
    peer's dominant, at `timestamp`. Empty for peers nobody currently targets
    (e.g. every peer, always, in the ``uniform`` scenario, since
    ``dominant_at`` there always returns ``None``)."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for entity_id, position in entity_positions.items():
        dominant = pattern.dominant_at(entity_id, timestamp)
        if dominant is not None:
            grouped.setdefault(dominant, []).append(position)
    return {
        peer_id: (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
        for peer_id, points in grouped.items()
    }


def generate_peer_positions(config: dict[str, Any], entities: list[dict[str, Any]] | None = None,
                             pattern: Any | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate peer initial positions and their position trace.

    `entities` and `pattern` are optional and only consulted when
    `peers.target_coupling.enabled` is set (see `nve_dataset/config.py`):
    an OPT-IN, OFF-BY-DEFAULT correlation between interaction locality (which
    peer a scenario currently calls an entity's "dominant" peer) and spatial
    locality (where that peer's random walk actually is). With the option
    absent or disabled, this function is byte-for-byte identical to the
    unconditional random walk it has always been - `entities`/`pattern` are
    never even read in that case.
    """
    seed = config["seed"]
    count = int(config["peers"]["count"])
    width, height = float(config["world"]["width"]), float(config["world"]["height"])
    model = config["peers"]["movement_model"]
    speed = float(config["peers"]["movement_speed"])
    duration = float(config["simulation"]["duration"])
    timestep = float(config["simulation"]["timestep"])

    coupling = config["peers"].get("target_coupling", {})
    coupling_enabled = bool(coupling.get("enabled", False))
    bias_probability = 0.0
    bias_rng = None
    entity_positions: dict[str, tuple[float, float]] = {}
    if coupling_enabled:
        if entities is None or pattern is None:
            raise ValueError(
                "peers.target_coupling.enabled requires generate_peer_positions to be "
                "called with entities and pattern (see nve_dataset/generator/core.py)")
        bias_probability = float(coupling["bias_probability"])
        bias_rng = rng_for(seed, "peer-mobility-target-bias")
        entity_positions = {
            row["entity_id"]: (float(row["initial_x"]), float(row["initial_y"])) for row in entities
        }

    initial_rng = rng_for(seed, "peer-initial-positions")
    ids = peer_ids_for(count)
    peers: list[dict[str, Any]] = []
    current: dict[str, tuple[float, float]] = {}
    for peer_id in ids:
        x, y = initial_rng.uniform(0, width), initial_rng.uniform(0, height)
        current[peer_id] = (x, y)
        peers.append({
            "peer_id": peer_id,
            "initial_x": f"{x:.6f}",
            "initial_y": f"{y:.6f}",
            "movement_model": model,
            "movement_speed": f"{speed:.6f}",
        })

    movement_rng = rng_for(seed, "peer-mobility")
    recorded = set(sample_steps(config))
    last_step = max(recorded)
    rows: list[dict[str, Any]] = []
    step_count = round(duration / timestep)
    for step in range(step_count + 1):
        if step in recorded:
            timestamp = min(step * timestep, duration)
            for peer in peers:
                peer_id = peer["peer_id"]
                x, y = current[peer_id]
                rows.append({"timestamp": f"{timestamp:.6f}", "peer_id": peer_id, "x": f"{x:.6f}", "y": f"{y:.6f}"})
        if model != "random_walk" or step >= last_step:
            # Nothing after the final recorded sample can be observed, so stop walking.
            break
        targets = _current_targets(pattern, entity_positions, step * timestep) if coupling_enabled else {}
        for peer in peers:
            peer_id = peer["peer_id"]
            x, y = current[peer_id]
            target = targets.get(peer_id)
            if target is not None and bias_rng.random() < bias_probability:
                # Head straight at the centroid of whoever currently calls this
                # peer "dominant" instead of drawing a uniform-random heading.
                tx, ty = target
                angle = math.atan2(ty - y, tx - x)
            else:
                angle = movement_rng.uniform(0, 2 * math.pi)
            distance = speed * timestep
            # Reflection avoids artificial mass at clipped world boundaries.
            nx, ny = x + distance * math.cos(angle), y + distance * math.sin(angle)
            if nx < 0 or nx > width:
                nx = min(width, max(0.0, -nx if nx < 0 else 2 * width - nx))
            if ny < 0 or ny > height:
                ny = min(height, max(0.0, -ny if ny < 0 else 2 * height - ny))
            current[peer_id] = (nx, ny)
    return peers, rows
