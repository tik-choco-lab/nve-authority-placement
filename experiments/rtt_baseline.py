"""Evaluate RTT-weighted online placement alongside demand-argmax and static/proximity baselines.

Both online selectors use the same demand history; the RTT-weighted selector
also uses the RTT matrix. Argmax-specific results must not be generalized to
all interaction-following policies.

EVALUATION. Replays the already-generated `datasets/ambiguity/ratio_*`
traces (no dataset generation) under the same combined objective
`experiments/combined_objective.py` prices the paper's policies with
(sum of served RTT + alpha * sum of migration RTT, alpha=1, the Oracle's own
alpha), for the paper's Static/Nearest/proposed set plus LowestRttPolicy at
several window sizes. Differences are seed-paired bootstraps, as everywhere
else in this directory.

WINDOW SIZES. The demand window is count-based, so it sets how many recent
requests the RTT sum is taken over. w=10 matches the paper's estimator
exactly (`experiments/common.py::DEMAND_WINDOW`); w=20/50/100 are included
because the RTT-weighted rule wants a longer window than the argmax rule
does -- it is averaging a cost over interacting peers rather than picking a
single winner, so a short window makes it chase whoever spoke last. w=1 is
deliberately excluded: it degenerates to "the requesting peer", migrating on
every request (see LowestRttPolicy's docstring).
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import dataset_root, RATIOS
from experiments.combined_objective import total_migration_rtt_ms
from experiments.common import DEMAND_WINDOW, EVALUATION, RELEVANT_RANGE
from experiments.load_significance import paired_bootstrap
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "rtt_baseline"

SEEDS = list(range(1, 31))
ALPHA = 1.0
RTT_WINDOWS = [10, 20, 50, 100]
PROPOSED_DWELLS = [0.0, 0.25, 1.0]


def build_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"id": "static", "type": "static"},
        {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
    ]
    # The paper's own policies, at the two dwell values Table I reports as
    # cheapest in some regime (0 above the diagonal, 0.25s inside the band)
    # plus the 1s the two-axis reading suggested.
    for dwell in PROPOSED_DWELLS:
        specs.append({
            "id": f"proposed_d{dwell}", "type": "engagement_aware",
            "target_rule": "demand_argmax",
            "engagement_threshold": 0.0, "dwell_time_s": dwell,
            "relevant_range": RELEVANT_RANGE, "window": DEMAND_WINDOW,
            "use_interaction_weight": False,
        })
    for size in RTT_WINDOWS:
        specs.append({
            "id": f"lowest_rtt_w{size}", "type": "lowest_rtt",
            "min_holding_time_s": 0.0, "window": {"type": "count", "size": size},
            "use_interaction_weight": False,
        })
    # Included so the stretch factors the paper quotes (cheapest online cost
    # over the offline optimum) can be derived here rather than by joining two
    # scripts' CSVs: the cheapest online policy is now sometimes an RTT-weighted
    # one, which combined_objective.py's policy set does not contain.
    specs.append({"id": "oracle_a1", "type": "oracle", "migration_penalty_weight": 1.0})
    return specs


def self_service_fraction(rows: list[dict[str, Any]]) -> float:
    """Fraction of requests served by the requesting peer itself, the `s` of
    the paper's latency decomposition (Section V-E). Recomputed here rather
    than imported from self_service.py so this script stands alone."""
    hits = sum(1 for row in rows if row["request_peer"] == row["authority_before"])
    return hits / len(rows)


def mean_remote_rtt_ms(rows: list[dict[str, Any]], processing_delay_ms: float) -> float:
    """E[RTT | remote], the other term of that decomposition. Taken from the
    served latency of the requests that were NOT self-served, minus the
    processing delay every request pays, so it needs no second lookup into the
    RTT matrix. Returns 0.0 if a policy somehow served everything locally."""
    remote = [float(row["interaction_latency_ms"]) - processing_delay_ms
              for row in rows if row["request_peer"] != row["authority_before"]]
    return statistics.fmean(remote) if remote else 0.0


SUMMARY_COLUMNS = [
    "dominant_peer_ratio", "seed", "policy_id", "total_requests",
    "mean_latency_ms", "migration_count", "self_service_fraction", "mean_remote_rtt_ms",
    "total_served_ms", "total_migration_rtt_ms", "combined_ms_alpha1",
]
AGGREGATE_COLUMNS = [
    "dominant_peer_ratio", "policy_id", "n_seeds",
    "mean_latency_ms_mean", "migration_count_mean", "self_service_fraction_mean",
    "mean_remote_rtt_ms_mean", "combined_s_alpha1_mean", "combined_s_alpha1_std",
]
COMPARISON_COLUMNS = [
    "dominant_peer_ratio", "policy_a", "policy_b", "n_seeds",
    "mean_diff_combined_s", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]
STRETCH_COLUMNS = [
    "dominant_peer_ratio", "cheapest_online_policy", "cheapest_online_combined_s",
    "oracle_combined_s", "stretch",
]


def verdict(low: float, high: float) -> str:
    if high < 0:
        return "a_cheaper"
    if low > 0:
        return "a_worse"
    return "not_significant"


def main() -> None:
    specs = build_specs()
    summary_rows: list[dict[str, Any]] = []
    combined_by_seed: dict[tuple[float, str], list[float]] = {}

    for ratio in RATIOS:
        for seed in SEEDS:
            root = dataset_root(ratio, seed)
            if not (root / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"{root} has no manifest.json - this script replays existing traces only; "
                    "run experiments/ambiguity_sweep.py first")
            dataset = Dataset(root)
            for spec in specs:
                outcome = replay(dataset, build_policy(dataset, spec), EVALUATION)
                served = sum(outcome["latencies"])
                migration_rtt = total_migration_rtt_ms(outcome["rows"])
                combined = served + ALPHA * migration_rtt
                combined_by_seed.setdefault((ratio, spec["id"]), []).append(combined / 1000.0)
                summary_rows.append({
                    "dominant_peer_ratio": ratio,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": len(outcome["latencies"]),
                    "mean_latency_ms": f"{statistics.fmean(outcome['latencies']):.6f}",
                    "migration_count": outcome["migrations"],
                    "self_service_fraction": f"{self_service_fraction(outcome['rows']):.6f}",
                    "mean_remote_rtt_ms": f"{mean_remote_rtt_ms(outcome['rows'], dataset.processing_delay_ms):.6f}",
                    "total_served_ms": f"{served:.6f}",
                    "total_migration_rtt_ms": f"{migration_rtt:.6f}",
                    "combined_ms_alpha1": f"{combined:.6f}",
                })
        print(f"[ok]    dominant_peer_ratio={ratio}: {len(SEEDS)} seeds x {len(specs)} policies")

    aggregate_rows = []
    for ratio in RATIOS:
        for spec in specs:
            rows = [row for row in summary_rows
                    if row["dominant_peer_ratio"] == ratio and row["policy_id"] == spec["id"]]
            combined = combined_by_seed[(ratio, spec["id"])]
            aggregate_rows.append({
                "dominant_peer_ratio": ratio,
                "policy_id": spec["id"],
                "n_seeds": len(rows),
                "mean_latency_ms_mean": f"{statistics.fmean(float(r['mean_latency_ms']) for r in rows):.6f}",
                "migration_count_mean": f"{statistics.fmean(float(r['migration_count']) for r in rows):.6f}",
                "self_service_fraction_mean": f"{statistics.fmean(float(r['self_service_fraction']) for r in rows):.6f}",
                "mean_remote_rtt_ms_mean": f"{statistics.fmean(float(r['mean_remote_rtt_ms']) for r in rows):.6f}",
                "combined_s_alpha1_mean": f"{statistics.fmean(combined):.6f}",
                "combined_s_alpha1_std": f"{statistics.stdev(combined):.6f}",
            })

    comparison_rows = []
    for ratio in RATIOS:
        best_rtt = min((f"lowest_rtt_w{size}" for size in RTT_WINDOWS),
                       key=lambda pid: statistics.fmean(combined_by_seed[(ratio, pid)]))
        best_baseline = min(("static", "nearest"),
                            key=lambda pid: statistics.fmean(combined_by_seed[(ratio, pid)]))
        best_proposed = min((f"proposed_d{d}" for d in PROPOSED_DWELLS),
                            key=lambda pid: statistics.fmean(combined_by_seed[(ratio, pid)]))
        for policy_b in (best_baseline, best_proposed):
            mean_diff, _std, low, high = paired_bootstrap(
                combined_by_seed[(ratio, best_rtt)], combined_by_seed[(ratio, policy_b)])
            comparison_rows.append({
                "dominant_peer_ratio": ratio,
                "policy_a": best_rtt,
                "policy_b": policy_b,
                "n_seeds": len(SEEDS),
                "mean_diff_combined_s": f"{mean_diff:.6f}",
                "bootstrap_ci95_low": f"{low:.6f}",
                "bootstrap_ci95_high": f"{high:.6f}",
                "verdict": verdict(low, high),
            })

    stretch_rows = []
    online_ids = [spec["id"] for spec in specs if spec["id"] != "oracle_a1"]
    for ratio in RATIOS:
        cheapest = min(online_ids, key=lambda pid: statistics.fmean(combined_by_seed[(ratio, pid)]))
        online = statistics.fmean(combined_by_seed[(ratio, cheapest)])
        oracle = statistics.fmean(combined_by_seed[(ratio, "oracle_a1")])
        stretch_rows.append({
            "dominant_peer_ratio": ratio,
            "cheapest_online_policy": cheapest,
            "cheapest_online_combined_s": f"{online:.6f}",
            "oracle_combined_s": f"{oracle:.6f}",
            "stretch": f"{online / oracle:.6f}",
        })

    write_csv(OUTPUT_ROOT / "stretch.csv", STRETCH_COLUMNS, stretch_rows)
    write_csv(OUTPUT_ROOT / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(OUTPUT_ROOT / "aggregate.csv", AGGREGATE_COLUMNS, aggregate_rows)
    write_csv(OUTPUT_ROOT / "comparisons.csv", COMPARISON_COLUMNS, comparison_rows)

    print("\ncombined cost at alpha=1, seconds per 300s run (mean over 30 seeds)")
    ids = [spec["id"] for spec in specs]
    print(f"{'ratio':<8}" + "".join(f"{pid:>17}" for pid in ids))
    for ratio in RATIOS:
        print(f"{ratio:<8}" + "".join(
            f"{statistics.fmean(combined_by_seed[(ratio, pid)]):>17.1f}" for pid in ids))
    print("\nbest RTT-weighted policy vs. best baseline / best proposed (paired bootstrap)")
    for row in comparison_rows:
        print(f"  ratio={row['dominant_peer_ratio']:<6} {row['policy_a']:<16} - {row['policy_b']:<16} "
              f"{float(row['mean_diff_combined_s']):+8.1f}s "
              f"CI[{float(row['bootstrap_ci95_low']):+7.1f}, {float(row['bootstrap_ci95_high']):+7.1f}] "
              f"{row['verdict']}")
    print("\nstretch: cheapest online cost over the offline optimum")
    for row in stretch_rows:
        print(f"  ratio={row['dominant_peer_ratio']:<6} {row['cheapest_online_policy']:<16} "
              f"{float(row['cheapest_online_combined_s']):6.1f}s / {float(row['oracle_combined_s']):6.1f}s "
              f"= {float(row['stretch']):.2f}")
    print(f"\n[write] {OUTPUT_ROOT.relative_to(ROOT)}/{{summary,aggregate,comparisons,stretch}}.csv")


if __name__ == "__main__":
    main()
