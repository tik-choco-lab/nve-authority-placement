"""Decompose latency into local service and conditional remote RTT.

mean_latency = (1-s) * E[RTT | served remotely] + processing_delay,
where s is the fraction of requests served by their requester. Self RTT is
zero. Peer RTT is generated independently of position; spatial proximity
therefore does not imply network proximity in these workloads.

Both s and the conditional remote mean are measured from replay rows. Policy
latency differences can reflect either term, so a single global mean RTT is
not substituted for the conditional mean. Reuses the switching and ambiguity
sweep workloads and policy set; missing inputs are generated from config and seed.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments import ambiguity_sweep, switching_sweep
from experiments.common import EVALUATION, build_policy_specs, check_equivalence
from nve_dataset.config import load_config
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = [
    "total_requests", "self_service_count", "self_service_rate",
    "mean_rtt_remote_ms", "mean_rtt_all_ms",
    "measured_mean_latency_ms", "predicted_mean_latency_ms",
    "abs_error_ms", "rel_error",
]

# Subset printed to stdout: the baselines, the two extreme dwell settings of
# each engagement_aware arm (dwell dominates the CSV with 14 rows per (knob, seed)
# and most of that detail belongs in the CSV, not the terminal), and Oracle.
PRINT_POLICIES = [
    "static", "nearest", "naive_target_aware",
    "engagement_g00_d0.0", "engagement_g05_d0.0", "engagement_g05_d2.0", "engagement_g05_d10.0",
    "oracle_a1",
]


def verify_rtt_diagonal(dataset: Dataset, context: str) -> None:
    """The whole self-service model rests on self-RTT being exactly zero
    (docs/POLICY_EVALUATION.md section 3 states RTT is a dataset fact; section
    4's Oracle DP relies on the same thing). Checked per-dataset, not once,
    because this experiment's conclusions are worthless if it silently held
    for some datasets and not others.
    """
    diagonal = np.diag(dataset.rtt)
    if not np.allclose(diagonal, 0.0):
        bad = np.flatnonzero(~np.isclose(diagonal, 0.0))
        print(
            f"FATAL: dataset.rtt diagonal is NOT all zero at {context}. "
            f"Nonzero self-RTT at peer indices {bad.tolist()[:10]} "
            f"(values {diagonal[bad][:10].tolist()}). "
            "The self-service model this experiment tests assumes self-RTT=0; "
            "that assumption does not hold for this dataset. Stopping.",
            file=sys.stderr,
        )
        sys.exit(1)


def self_service_stats(dataset: Dataset, outcome: dict[str, Any]) -> dict[str, float]:
    """Latency decomposition: mean_latency = (1-s) * E[RTT | remote] + processing_delay.

    Recomputed straight from replay's per-request rows and latencies (`rows[i]`
    corresponds to `latencies[i]`, both produced by the same loop in
    nve_policy/engine.py::replay), not from dataset.rtt indices directly, so
    this stays a read of the engine's own output rather than a re-derivation
    that could silently diverge from what replay actually charged.
    """
    rows = outcome["rows"]
    latencies = outcome["latencies"]
    processing_delay = dataset.processing_delay_ms
    total = len(rows)

    rtt_terms = [latency - processing_delay for latency in latencies]
    self_mask = [row["authority_before"] == row["request_peer"] for row in rows]

    self_rtts = [rtt for rtt, is_self in zip(rtt_terms, self_mask) if is_self]
    if self_rtts and max(abs(value) for value in self_rtts) > 1e-6:
        worst = max(abs(value) for value in self_rtts)
        raise AssertionError(
            f"self-service request has nonzero RTT term (max |RTT|={worst:.6f} ms); "
            "the zero-diagonal assumption failed at replay time, not just in dataset.rtt"
        )

    remote_rtts = [rtt for rtt, is_self in zip(rtt_terms, self_mask) if not is_self]
    self_count = sum(self_mask)
    rate = self_count / total if total else 0.0
    mean_rtt_remote = statistics.fmean(remote_rtts) if remote_rtts else 0.0
    mean_rtt_all = statistics.fmean(rtt_terms) if rtt_terms else 0.0
    measured_mean_latency = statistics.fmean(latencies) if latencies else 0.0
    predicted_mean_latency = (1.0 - rate) * mean_rtt_remote + processing_delay
    abs_error = abs(predicted_mean_latency - measured_mean_latency)
    rel_error = abs_error / measured_mean_latency if measured_mean_latency else 0.0

    return {
        "total_requests": total,
        "self_service_count": self_count,
        "self_service_rate": rate,
        "mean_rtt_remote_ms": mean_rtt_remote,
        "mean_rtt_all_ms": mean_rtt_all,
        "measured_mean_latency_ms": measured_mean_latency,
        "predicted_mean_latency_ms": predicted_mean_latency,
        "abs_error_ms": abs_error,
        "rel_error": rel_error,
    }


def aggregate_over_seeds(summary_rows: list[dict[str, Any]], group_column: str) -> list[dict[str, Any]]:
    """Same mean/std-over-seeds convention as experiments/common.py's
    aggregate_over_seeds, generalized over METRIC_COLUMNS instead of a fixed
    metric set, since this script's summary schema is specific to the
    self-service model and doesn't fit the shared helper.
    """
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row[group_column], row["policy_id"])
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for (group_value, policy_id), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, Any] = {group_column: group_value, "policy_id": policy_id}
        for column in METRIC_COLUMNS:
            values = [float(row[column]) for row in rows]
            out[f"{column}_mean"] = f"{statistics.fmean(values):.6f}"
            out[f"{column}_std"] = f"{statistics.stdev(values):.6f}"
        aggregate_rows.append(out)
    return aggregate_rows


def run_sweep(
    sweep_name: str,
    group_column: str,
    knob_values: list[float],
    seeds: list[int],
    ensure_dataset: Any,
    base_config: dict[str, Any],
) -> list[dict[str, Any]]:
    specs = build_policy_specs()
    summary_rows: list[dict[str, Any]] = []

    for knob in knob_values:
        for seed in seeds:
            root = ensure_dataset(base_config, knob, seed)
            dataset = Dataset(root)
            context = f"{sweep_name} {group_column}={knob} seed={seed}"
            verify_rtt_diagonal(dataset, context)

            outcomes: dict[str, dict[str, Any]] = {}
            for spec in specs:
                policy = build_policy(dataset, spec)
                outcome = replay(dataset, policy, EVALUATION)
                outcomes[spec["id"]] = outcome
                stats = self_service_stats(dataset, outcome)
                summary_rows.append({
                    group_column: knob,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": stats["total_requests"],
                    "self_service_count": stats["self_service_count"],
                    "self_service_rate": f"{stats['self_service_rate']:.6f}",
                    "mean_rtt_remote_ms": f"{stats['mean_rtt_remote_ms']:.6f}",
                    "mean_rtt_all_ms": f"{stats['mean_rtt_all_ms']:.6f}",
                    "measured_mean_latency_ms": f"{stats['measured_mean_latency_ms']:.6f}",
                    "predicted_mean_latency_ms": f"{stats['predicted_mean_latency_ms']:.6f}",
                    "abs_error_ms": f"{stats['abs_error_ms']:.6f}",
                    "rel_error": f"{stats['rel_error']:.6f}",
                })
            check_equivalence(outcomes, context)
            print(f"[ok]    {context}: {len(specs)} policies replayed, self-service model evaluated")

    return summary_rows


def print_table(aggregate_rows: list[dict[str, Any]], group_column: str, knob_values: list[float]) -> None:
    by_key = {(row[group_column], row["policy_id"]): row for row in aggregate_rows}
    header = (
        f"{group_column:>18}  {'policy_id':>16}  {'s (self-svc)':>13}  "
        f"{'E[RTT|remote]':>14}  {'measured_ms':>12}  {'predicted_ms':>13}  "
        f"{'abs_err_ms':>11}  {'rel_err':>8}"
    )
    print(header)
    print("-" * len(header))
    for knob in knob_values:
        for policy_id in PRINT_POLICIES:
            row = by_key.get((knob, policy_id))
            if row is None:
                continue
            print(
                f"{knob:>18.4g}  {policy_id:>16}  "
                f"{float(row['self_service_rate_mean']):>13.4f}  "
                f"{float(row['mean_rtt_remote_ms_mean']):>14.4f}  "
                f"{float(row['measured_mean_latency_ms_mean']):>12.4f}  "
                f"{float(row['predicted_mean_latency_ms_mean']):>13.4f}  "
                f"{float(row['abs_error_ms_mean']):>11.4f}  "
                f"{float(row['rel_error_mean']):>8.4f}"
            )
        print()


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")

    switching_summary = run_sweep(
        "switching", "phase_duration", switching_sweep.PHASE_DURATIONS, switching_sweep.SEEDS,
        switching_sweep.ensure_dataset, base_config,
    )
    switching_summary_columns = [
        "phase_duration", "seed", "policy_id",
        "total_requests", "self_service_count", "self_service_rate",
        "mean_rtt_remote_ms", "mean_rtt_all_ms",
        "measured_mean_latency_ms", "predicted_mean_latency_ms",
        "abs_error_ms", "rel_error",
    ]
    switching_summary_path = switching_sweep.SWITCHING_ROOT / "self_service_summary.csv"
    write_csv(switching_summary_path, switching_summary_columns, switching_summary)
    print(f"[write] {switching_summary_path.relative_to(ROOT)} ({len(switching_summary)} rows)")

    switching_aggregate = aggregate_over_seeds(switching_summary, "phase_duration")
    switching_aggregate_columns = ["phase_duration", "policy_id"] + [
        f"{column}_{suffix}" for column in METRIC_COLUMNS for suffix in ("mean", "std")
    ]
    switching_aggregate_path = switching_sweep.SWITCHING_ROOT / "self_service_aggregate.csv"
    write_csv(switching_aggregate_path, switching_aggregate_columns, switching_aggregate)
    print(f"[write] {switching_aggregate_path.relative_to(ROOT)} ({len(switching_aggregate)} rows)")

    ambiguity_summary = run_sweep(
        "ambiguity", "dominant_peer_ratio", ambiguity_sweep.RATIOS, ambiguity_sweep.SEEDS,
        ambiguity_sweep.ensure_dataset, base_config,
    )
    ambiguity_summary_columns = [
        "dominant_peer_ratio", "seed", "policy_id",
        "total_requests", "self_service_count", "self_service_rate",
        "mean_rtt_remote_ms", "mean_rtt_all_ms",
        "measured_mean_latency_ms", "predicted_mean_latency_ms",
        "abs_error_ms", "rel_error",
    ]
    ambiguity_summary_path = ambiguity_sweep.AMBIGUITY_ROOT / "self_service_summary.csv"
    write_csv(ambiguity_summary_path, ambiguity_summary_columns, ambiguity_summary)
    print(f"[write] {ambiguity_summary_path.relative_to(ROOT)} ({len(ambiguity_summary)} rows)")

    ambiguity_aggregate = aggregate_over_seeds(ambiguity_summary, "dominant_peer_ratio")
    ambiguity_aggregate_columns = ["dominant_peer_ratio", "policy_id"] + [
        f"{column}_{suffix}" for column in METRIC_COLUMNS for suffix in ("mean", "std")
    ]
    ambiguity_aggregate_path = ambiguity_sweep.AMBIGUITY_ROOT / "self_service_aggregate.csv"
    write_csv(ambiguity_aggregate_path, ambiguity_aggregate_columns, ambiguity_aggregate)
    print(f"[write] {ambiguity_aggregate_path.relative_to(ROOT)} ({len(ambiguity_aggregate)} rows)")

    print(f"\nRTT diagonal check: PASSED (self-RTT == 0) on every dataset touched above.\n")

    print("=" * 100)
    print("Self-service model: mean_latency = (1 - s) * E[RTT | served remotely] + processing_delay")
    print("=" * 100)
    print("\n--- switching sweep (datasets/switching/, knob = phase_duration) ---\n")
    print_table(switching_aggregate, "phase_duration", switching_sweep.PHASE_DURATIONS)
    print("\n--- ambiguity sweep (datasets/ambiguity/, knob = dominant_peer_ratio) ---\n")
    print_table(ambiguity_aggregate, "dominant_peer_ratio", ambiguity_sweep.RATIOS)


if __name__ == "__main__":
    main()
