"""Replay one interaction trace under one authority placement policy.

Causal model, identical for every policy:

    1. A request leaves its peer and arrives at whichever peer currently holds
       authority. The sender cannot know about a migration that has not happened
       yet, so the request is always served by ``authority_before``.
    2. Only afterwards does the policy observe the request and possibly migrate.

An offline oracle still gains its advantage inside this model: it may place the
authority correctly *before* the request arrives, because its plan was computed
with knowledge of the whole sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nve_dataset.util import write_csv, write_json
from nve_policy.dataset import Dataset, Interaction

RESULT_COLUMNS = [
    "timestamp", "request_id", "entity_id", "request_peer",
    "authority_before", "authority_after", "interaction_latency_ms",
    "migration_occurred", "migration_bytes", "migration_network_latency_ms",
    "migration_duration_ms", "network_traffic_bytes",
]

# Written only for policies that expose a `decisions` log (currently
# engagement_aware) - see write_decisions below. Not every policy has reason codes
# worth reporting, so this is opt-in rather than a column every policy must
# populate.
DECISION_COLUMNS = [
    "timestamp", "request_id", "entity_id", "request_peer",
    "authority_before", "authority_after", "reason", "target_peer",
    "engagement", "dwell_s", "holder_distance",
]


@dataclass
class Evaluation:
    """Evaluation-side assumptions. Not dataset facts - see docs/POLICY_EVALUATION.md."""
    request_bytes: int
    response_bytes: int
    migration_bandwidth_mbps: float | None = None

    def migration_duration_ms(self, network_latency_ms: float, size_bytes: int) -> float | None:
        if self.migration_bandwidth_mbps is None:
            return None
        serialization_ms = size_bytes * 8 / (self.migration_bandwidth_mbps * 1e6) * 1e3
        return network_latency_ms + serialization_ms


@dataclass
class EntityState:
    authority: int
    since: float = 0.0
    holding_times: list[float] = field(default_factory=list)


def replay(dataset: Dataset, policy: Any, evaluation: Evaluation) -> dict[str, Any]:
    state = {
        entity_id: EntityState(entity.initial_authority)
        for entity_id, entity in dataset.entities.items()
    }
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    migrations = 0
    migration_bytes_total = 0
    traffic_total = 0
    peer_count = len(dataset.peers)
    # Processing load, LB-Spiral's second objective alongside stretch: the
    # execution work a peer is actually made to do, not just the network cost
    # of reaching it. served_per_peer counts requests (authority_before is who
    # serves - see the causal model above); authority_time_per_peer accumulates
    # wall-clock seconds of authority held, mirroring holding_times below.
    served_per_peer = [0] * peer_count
    authority_time_per_peer = [0.0] * peer_count

    for request in dataset.interactions:
        entity = dataset.entities[request.entity_id]
        current = state[request.entity_id]
        before = current.authority
        served_per_peer[before] += 1

        latency = dataset.rtt[request.peer, before] + dataset.processing_delay_ms
        latencies.append(latency)
        traffic = evaluation.request_bytes + evaluation.response_bytes

        after = policy.on_request(request, before)
        if after == before:
            row_migration = {"migration_occurred": 0, "migration_bytes": "",
                             "migration_network_latency_ms": "", "migration_duration_ms": ""}
        else:
            migrations += 1
            migration_bytes_total += entity.state_size_bytes
            traffic += entity.state_size_bytes
            network_latency = dataset.rtt[before, after]
            duration = evaluation.migration_duration_ms(network_latency, entity.state_size_bytes)
            current.holding_times.append(request.timestamp - current.since)
            # Credited to the outgoing holder (`before`): it held authority for
            # [since, timestamp) regardless of who takes over next.
            authority_time_per_peer[before] += request.timestamp - current.since
            current.since = request.timestamp
            current.authority = after
            row_migration = {
                "migration_occurred": 1,
                "migration_bytes": entity.state_size_bytes,
                "migration_network_latency_ms": f"{network_latency:.6f}",
                "migration_duration_ms": "" if duration is None else f"{duration:.6f}",
            }
        traffic_total += traffic
        rows.append({
            "timestamp": f"{request.timestamp:.6f}",
            "request_id": request.request_id,
            "entity_id": request.entity_id,
            "request_peer": dataset.peers[request.peer],
            "authority_before": dataset.peers[before],
            "authority_after": dataset.peers[after],
            "interaction_latency_ms": f"{latency:.6f}",
            "network_traffic_bytes": traffic,
            **row_migration,
        })

    for current in state.values():
        current.holding_times.append(dataset.duration - current.since)
        # Final holder of each entity is credited for the remainder of the run.
        # Summed across entities this can exceed dataset.duration - a peer may
        # hold several entities at once, and each one's time counts - which is
        # correct: authority_time_per_peer measures load, not a single clock.
        authority_time_per_peer[current.authority] += dataset.duration - current.since
    return {
        "rows": rows,
        "latencies": latencies,
        "migrations": migrations,
        "migration_bytes_total": migration_bytes_total,
        "traffic_total": traffic_total,
        "holding_times": [value for current in state.values() for value in current.holding_times],
        "served_per_peer": served_per_peer,
        "authority_time_per_peer": authority_time_per_peer,
    }


def write_results(root: Path, policy_id: str, rows: list[dict[str, Any]]) -> Path:
    path = root / "output" / policy_id / "results.csv"
    write_csv(path, RESULT_COLUMNS, rows)
    return path


def write_decisions(root: Path, policy_id: str, rows: list[dict[str, Any]]) -> Path:
    path = root / "output" / policy_id / "decisions.csv"
    write_csv(path, DECISION_COLUMNS, rows)
    return path


def write_metrics(root: Path, policy_id: str, metrics: dict[str, Any]) -> Path:
    path = root / "output" / policy_id / "metrics.json"
    write_json(path, metrics)
    return path
