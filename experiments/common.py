"""Machinery shared by the experiments in this directory.

`switching_sweep.py` (target switching *speed*, via `interaction.shifting.
phase_duration`) and `ambiguity_sweep.py` (target switching *ambiguity*, via
`interaction.shifting.dominant_peer_ratio`) compare the exact same 18-policy
set on datasets generated from the same `shifting` scenario, and both need to
verify the same theta_G=0/theta_D=0 <-> naive_target_aware equivalence and
aggregate the same metrics over seeds. That overlap lives here so it is
defined once. Everything that differs between the two sweeps - which config
key is swept, dataset directory layout, summary CSV schema, the extra
`migration_floor` column - stays in the sweep script itself.
"""
from __future__ import annotations

import statistics
import sys
from typing import Any

from nve_policy.engine import Evaluation

# relevant_range effectively infinite: world diagonal is sqrt(1000^2+1000^2) =
# ~1414, so 1e9 never triggers RELEASE_OUT_OF_RANGE / KEEP_TARGET_OUT_OF_RANGE.
RELEVANT_RANGE = 1.0e9

# theta_D sweep shared by both engagement_threshold arms in build_policy_specs.
DWELL_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

EVALUATION = Evaluation(request_bytes=128, response_bytes=128, migration_bandwidth_mbps=None)

DEMAND_WINDOW = {"type": "count", "size": 10}


def build_policy_specs() -> list[dict[str, Any]]:
    """The comparison set for these experiments (kept out of policies.yaml on
    purpose: that file is the baseline suite and must keep validating/running
    unchanged)."""
    specs: list[dict[str, Any]] = [
        {"id": "static", "type": "static"},
        {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
        {
            "id": "naive_target_aware", "type": "interaction_aware",
            "window": DEMAND_WINDOW, "use_interaction_weight": False, "min_holding_time_s": 0.0,
        },
    ]
    for engagement_label, engagement_threshold in (("g00", 0.0), ("g05", 0.5)):
        for dwell in DWELL_VALUES:
            specs.append({
                "id": f"engagement_{engagement_label}_d{dwell}",
                "type": "engagement_aware",
                "target_rule": "demand_argmax",
                "engagement_threshold": engagement_threshold,
                "dwell_time_s": dwell,
                "relevant_range": RELEVANT_RANGE,
                "window": DEMAND_WINDOW,
                "use_interaction_weight": False,
            })
    specs.append({"id": "oracle_a1", "type": "oracle", "migration_penalty_weight": 1.0})
    return specs


def check_equivalence(outcomes: dict[str, dict[str, Any]], context: str) -> None:
    """engagement_g00_d0.0 (theta_G=0, theta_D=0, theta_R=inf) must be EXACTLY
    naive_target_aware: both migrate to the argmax of the same count-10 window
    with zero hysteresis. This is asserted on every generated dataset, not
    just once, because a divergence that only shows up at some
    (swept knob, seed) combination would itself be evidence of an off-by-one
    in one of the two policies. `context` names that combination (e.g.
    "phase_duration=10.0 seed=1") purely for the error message.
    """
    left, right = outcomes["engagement_g00_d0.0"], outcomes["naive_target_aware"]
    left_seq = [row["authority_after"] for row in left["rows"]]
    right_seq = [row["authority_after"] for row in right["rows"]]
    if left_seq != right_seq:
        for index, (a, b) in enumerate(zip(left_seq, right_seq)):
            if a != b:
                print(f"EQUIVALENCE CHECK FAILED at {context} "
                      f"row {index}: engagement_g00_d0.0={a!r} naive_target_aware={b!r}", file=sys.stderr)
                break
        raise AssertionError(
            f"engagement_g00_d0.0 != naive_target_aware at {context}: "
            "authority_after sequences diverge (see stderr for the first differing row)"
        )
    assert left["migrations"] == right["migrations"]
    assert sum(left["latencies"]) == sum(right["latencies"])


def aggregate_over_seeds(summary_rows: list[dict[str, Any]], group_column: str) -> list[dict[str, Any]]:
    """Mean/std of the three metrics over the seed dimension, grouped by
    (group_column, policy_id). `group_column` is the name of the swept knob's
    key in each summary row (e.g. "phase_duration" or "dominant_peer_ratio").
    Any other column present in a summary row (e.g. a migration_floor a
    caller wants to carry through) is untouched here - callers that need one
    add it to the returned rows themselves.
    """
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row[group_column], row["policy_id"])
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for (group_value, policy_id), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        mean_latencies = [float(row["mean_latency_ms"]) for row in rows]
        p95_latencies = [float(row["p95_latency_ms"]) for row in rows]
        migrations = [float(row["migration_count"]) for row in rows]
        max_requests = [float(row["max_requests_per_peer"]) for row in rows]
        imbalances = [float(row["request_imbalance"]) for row in rows]
        aggregate_rows.append({
            group_column: group_value,
            "policy_id": policy_id,
            "mean_latency_ms_mean": f"{statistics.fmean(mean_latencies):.6f}",
            "mean_latency_ms_std": f"{statistics.stdev(mean_latencies):.6f}",
            "p95_latency_ms_mean": f"{statistics.fmean(p95_latencies):.6f}",
            "p95_latency_ms_std": f"{statistics.stdev(p95_latencies):.6f}",
            "migration_count_mean": f"{statistics.fmean(migrations):.6f}",
            "migration_count_std": f"{statistics.stdev(migrations):.6f}",
            "max_requests_per_peer_mean": f"{statistics.fmean(max_requests):.6f}",
            "max_requests_per_peer_std": f"{statistics.stdev(max_requests):.6f}",
            "request_imbalance_mean": f"{statistics.fmean(imbalances):.6f}",
            "request_imbalance_std": f"{statistics.stdev(imbalances):.6f}",
        })
    return aggregate_rows
