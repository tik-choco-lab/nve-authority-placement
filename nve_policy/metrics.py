from __future__ import annotations

import statistics
from typing import Any

from nve_dataset.analysis.statistics import _percentile
from nve_policy.dataset import Dataset


def summarize(dataset: Dataset, policy_id: str, policy_spec: dict[str, Any],
              outcome: dict[str, Any]) -> dict[str, Any]:
    latencies = outcome["latencies"]
    holding = outcome["holding_times"]
    duration = dataset.duration
    requests = len(latencies)
    migrations = outcome["migrations"]
    served = outcome["served_per_peer"]
    authority_time = outcome["authority_time_per_peer"]
    peer_count = len(served)
    # Mean is taken over ALL peers, not just the ones that served a request:
    # an idle peer is a real peer whose capacity is going unused, and excluding
    # it from the mean would understate the imbalance we are trying to detect
    # (LB-Spiral's processing-load objective is about load ACROSS the peer set,
    # not conditioned on who happened to get picked).
    mean_requests = requests / peer_count if peer_count else 0.0
    max_requests = max(served) if served else 0
    mean_authority_time = sum(authority_time) / peer_count if peer_count else 0.0
    max_authority_time = max(authority_time) if authority_time else 0.0
    return {
        "dataset_id": dataset.manifest["dataset_id"],
        "scenario": dataset.manifest["scenario"],
        "seed": dataset.manifest["seed"],
        "policy_id": policy_id,
        "policy_spec": policy_spec,
        "total_requests": requests,
        "interaction_latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "migration_count": migrations,
        "migration_rate_per_second": migrations / duration,
        "migration_rate_per_request": migrations / requests if requests else 0.0,
        "migration_bytes_total": outcome["migration_bytes_total"],
        "authority_holding_time_s": {
            "mean": statistics.fmean(holding) if holding else 0.0,
            "median": statistics.median(holding) if holding else 0.0,
            "min": min(holding) if holding else 0.0,
        },
        "total_network_traffic_bytes": outcome["traffic_total"],
        "processing_load": {
            "max_requests_per_peer": max_requests,
            "mean_requests_per_peer": mean_requests,
            "request_imbalance": max_requests / mean_requests if mean_requests else 0.0,
            "active_peer_count": sum(1 for count in served if count > 0),
            "max_authority_time_s": max_authority_time,
            "authority_time_imbalance": (
                max_authority_time / mean_authority_time if mean_authority_time else 0.0
            ),
        },
    }


COMPARISON_COLUMNS = [
    "dataset_id", "scenario", "seed", "policy_id",
    "total_requests", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "max_latency_ms",
    "migration_count", "migration_rate_per_second", "migration_rate_per_request",
    "migration_bytes_total", "mean_holding_time_s", "total_network_traffic_bytes",
    "max_requests_per_peer", "request_imbalance",
]


def comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    latency = metrics["interaction_latency_ms"]
    processing_load = metrics["processing_load"]
    return {
        "dataset_id": metrics["dataset_id"],
        "scenario": metrics["scenario"],
        "seed": metrics["seed"],
        "policy_id": metrics["policy_id"],
        "total_requests": metrics["total_requests"],
        "mean_latency_ms": f"{latency['mean']:.6f}",
        "median_latency_ms": f"{latency['median']:.6f}",
        "p95_latency_ms": f"{latency['p95']:.6f}",
        "max_latency_ms": f"{latency['max']:.6f}",
        "migration_count": metrics["migration_count"],
        "migration_rate_per_second": f"{metrics['migration_rate_per_second']:.6f}",
        "migration_rate_per_request": f"{metrics['migration_rate_per_request']:.6f}",
        "migration_bytes_total": metrics["migration_bytes_total"],
        "mean_holding_time_s": f"{metrics['authority_holding_time_s']['mean']:.6f}",
        "total_network_traffic_bytes": metrics["total_network_traffic_bytes"],
        "max_requests_per_peer": processing_load["max_requests_per_peer"],
        "request_imbalance": f"{processing_load['request_imbalance']:.6f}",
    }
