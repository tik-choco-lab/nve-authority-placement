from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nve_dataset.util import rng_for, weighted_choice


@dataclass(frozen=True)
class ScenarioPattern:
    name: str
    peer_ids: list[str]
    dominant_by_entity: dict[str, str]
    phase_duration: float | None = None
    dominant_ratio: float | None = None
    burst_intervals: tuple[tuple[float, float], ...] = ()

    def dominant_at(self, entity_id: str, timestamp: float) -> str | None:
        if self.name == "uniform":
            return None
        base_index = self.peer_ids.index(self.dominant_by_entity[entity_id])
        if self.name == "shifting":
            assert self.phase_duration is not None
            phase = int(timestamp // self.phase_duration)
            return self.peer_ids[(base_index + phase) % len(self.peer_ids)]
        if self.name == "burst":
            in_burst = any(start <= timestamp < end for start, end in self.burst_intervals)
            return self.dominant_by_entity[entity_id] if in_burst else None
        return self.dominant_by_entity[entity_id]

    def choose_peer(self, entity_id: str, timestamp: float, rng: Any) -> str:
        dominant = self.dominant_at(entity_id, timestamp)
        if dominant is None or len(self.peer_ids) == 1:
            return rng.choice(self.peer_ids)
        assert self.dominant_ratio is not None
        others = [peer for peer in self.peer_ids if peer != dominant]
        probabilities = [self.dominant_ratio] + [
            (1.0 - self.dominant_ratio) / len(others)
        ] * len(others)
        return weighted_choice(rng, [dominant, *others], probabilities)

    def metadata(self, duration: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dominant_peer_by_entity": self.dominant_by_entity,
        }
        if self.dominant_ratio is not None:
            result["dominant_peer_ratio"] = self.dominant_ratio
        if self.phase_duration is not None:
            result["phase_duration"] = self.phase_duration
            result["phases"] = [
                {
                    "start": start,
                    "end": min(start + self.phase_duration, duration),
                    "dominant_peer_by_entity": {
                        entity_id: self.dominant_at(entity_id, start)
                        for entity_id in self.dominant_by_entity
                    },
                }
                for start in _frange(0.0, duration, self.phase_duration)
            ]
        if self.burst_intervals:
            result["burst_intervals"] = [
                {"start": start, "end": end} for start, end in self.burst_intervals
            ]
        return result


def _frange(start: float, stop: float, step: float) -> list[float]:
    result = []
    value = start
    while value < stop:
        result.append(value)
        value += step
    return result


def build_pattern(config: dict[str, Any], peer_ids: list[str], entity_ids: list[str]) -> ScenarioPattern:
    scenario = config["scenario"]
    rng = rng_for(config["seed"], f"scenario-assignment-{scenario}")
    shuffled = peer_ids.copy()
    rng.shuffle(shuffled)
    dominant = {entity_id: shuffled[index % len(shuffled)] for index, entity_id in enumerate(entity_ids)}
    interaction = config["interaction"]
    # OPT-IN, OFF by default (`interaction.shared_dominant_peer`). The round
    # robin above gives every entity a DIFFERENT dominant peer whenever there
    # are at least as many peers as entities, which is the case in every
    # dataset this project generates. Authority-placement load balance is then
    # measured on a workload that cannot express the adversarial case for it:
    # several entities engaging the SAME popular player at once, which is what
    # an interaction-following policy would pile onto one peer. Turning this on
    # collapses the assignment onto a single peer so that case can be measured;
    # with the key absent, `dominant` is untouched and generation stays
    # byte-identical (see tests/test_shared_dominant.py).
    if interaction.get("shared_dominant_peer", False) and entity_ids:
        dominant = {entity_id: dominant[entity_ids[0]] for entity_id in entity_ids}
    if scenario == "uniform":
        return ScenarioPattern(scenario, peer_ids, dominant)
    section = interaction[scenario]
    ratio = float(section["dominant_peer_ratio"])
    if scenario == "shifting":
        return ScenarioPattern(scenario, peer_ids, dominant, float(section["phase_duration"]), ratio)
    if scenario == "burst":
        intervals = tuple(
            (float(item["start"]), float(item["start"] + item["duration"]))
            for item in section["intervals"]
        )
        return ScenarioPattern(scenario, peer_ids, dominant, dominant_ratio=ratio, burst_intervals=intervals)
    return ScenarioPattern(scenario, peer_ids, dominant, dominant_ratio=ratio)
