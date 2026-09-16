"""How much of the "boundary near a dominant peer ratio of 0.30" is a
property of the policy, and how much of the workload it was measured on?

The paper's boundary is measured at one request rate (2 requests/s/entity,
`config.yaml`) with one demand estimator (the last 10 requests,
`experiments/common.py::DEMAND_WINDOW`). Both are free parameters that were
never swept, and the paper's own Threats section flags the second one. This
script sweeps them.

WHY THE RATE SHOULD MATTER, A PRIORI. Under the combined objective a
migration costs one round trip, a fixed price independent of how many
requests follow it, while the latency it saves accrues per request served
locally afterwards. Doubling the request rate therefore doubles the return
on a migration without changing its price. A count-based demand window
compounds this: 10 requests span 5 s of wall clock at 2 requests/s but only
1.25 s at 8, so the estimator also reacts faster at a higher rate. Neither
effect is visible in a sweep that holds the rate fixed, and both push the
boundary in the same direction.

TWO SWEEPS, TWO CSVs.

1. `rate_aggregate.csv` / `rate_comparisons.csv`: request rate in
   {0.5, 2, 8} requests/s/entity crossed with the dominant peer ratios that
   bracket the paper's boundary. New datasets ARE generated here (under
   `datasets/boundary/rate_*`), since the rate is a generator parameter;
   everything else matches `ambiguity_sweep.py` exactly (300 s, 20 peers,
   5 entities, phase_duration 10 s), so the rate=2 arm is the paper's own
   configuration re-generated under a different directory name rather than a
   different experiment.

2. `window_aggregate.csv`: demand window size in {5, 10, 20, 30} crossed
   with theta_D, replayed on the EXISTING `datasets/ambiguity` traces (the
   window is a policy parameter, so no regeneration is needed). This asks
   whether the boundary is an artifact of the count-10 estimator, and how
   the cost level at a fixed ratio compares across window sizes.

The reported quantity is the same combined objective as
`experiments/combined_objective.py` (served RTT + alpha * migration RTT at
alpha = 1), so the numbers are directly comparable to Table I of the paper,
and "does interaction-following beat the better baseline here" is decided by
a seed-paired bootstrap rather than by comparing two means.
"""
from __future__ import annotations

import copy
import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import dataset_root as ambiguity_dataset_root
from experiments.combined_objective import total_migration_rtt_ms
from experiments.common import EVALUATION, RELEVANT_RANGE
from experiments.load_significance import paired_bootstrap
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "boundary"

# Matched to ambiguity_sweep.py so the rate=2 arm reproduces the paper's setup.
DURATION = 300.0
PEER_COUNT = 20
ENTITY_COUNT = 5
PHASE_DURATION = 10.0
SEEDS = list(range(1, 31))

RATES = [0.5, 2.0, 8.0]
RATIOS = [0.85, 0.40, 0.30, 0.25]
WINDOW_SIZES = [5, 10, 20, 30]
DWELLS = [0.0, 0.25, 1.0, 2.0]
ALPHA = 1.0

BASELINES: list[dict[str, Any]] = [
    {"id": "static", "type": "static"},
    {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
]


def follower(window_size: int, dwell: float) -> dict[str, Any]:
    return {
        "id": f"follow_w{window_size}_d{dwell}", "type": "engagement_aware",
        "target_rule": "demand_argmax",
        "engagement_threshold": 0.0, "dwell_time_s": dwell,
        "relevant_range": RELEVANT_RANGE,
        "window": {"type": "count", "size": window_size},
        "use_interaction_weight": False,
    }


def rate_dataset_root(rate: float, ratio: float, seed: int) -> Path:
    return OUTPUT_ROOT / f"rate_{rate:g}" / f"ratio_{round(ratio * 100):03d}" / f"seed_{seed:03d}"


def ensure_rate_dataset(base_config: dict[str, Any], rate: float, ratio: float, seed: int) -> Path:
    root = rate_dataset_root(rate, ratio, seed)
    if (root / "manifest.json").is_file():
        return root
    config = copy.deepcopy(base_config)
    config["simulation"]["duration"] = DURATION
    config["peers"]["count"] = PEER_COUNT
    config["entities"]["count"] = ENTITY_COUNT
    config["interaction"]["rate_per_entity"] = rate
    config["interaction"]["shifting"]["phase_duration"] = PHASE_DURATION
    config["interaction"]["shifting"]["dominant_peer_ratio"] = ratio
    generate_dataset(config_for_run(config, "shifting", seed), output_dir=root, analyze=False)
    return root


def combined_seconds(dataset: Dataset, spec: dict[str, Any]) -> tuple[float, float, int]:
    outcome = replay(dataset, build_policy(dataset, spec), EVALUATION)
    served = sum(outcome["latencies"])
    combined = served + ALPHA * total_migration_rtt_ms(outcome["rows"])
    return combined / 1000.0, statistics.fmean(outcome["latencies"]), outcome["migrations"]


RATE_AGGREGATE_COLUMNS = [
    "rate_per_entity", "dominant_peer_ratio", "policy_id", "n_seeds",
    "combined_s_alpha1_mean", "combined_s_alpha1_std",
    "mean_latency_ms_mean", "migration_count_mean",
]
RATE_COMPARISON_COLUMNS = [
    "rate_per_entity", "dominant_peer_ratio", "best_follower", "best_baseline", "n_seeds",
    "mean_diff_combined_s", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]
WINDOW_AGGREGATE_COLUMNS = [
    "dominant_peer_ratio", "window_size", "dwell_time_s", "n_seeds",
    "combined_s_alpha1_mean", "combined_s_alpha1_std", "beats_best_baseline",
]


def verdict(low: float, high: float) -> str:
    if high < 0:
        return "follower_wins"
    if low > 0:
        return "follower_loses"
    return "not_significant"


def rate_sweep(base_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sweep 1: request rate. Only the count-10 window is replayed here - the
    window is sweep 2's variable, and crossing both would confound them."""
    specs = BASELINES + [follower(10, dwell) for dwell in DWELLS]
    combined: dict[tuple[float, float, str], list[float]] = {}
    latency: dict[tuple[float, float, str], list[float]] = {}
    migrations: dict[tuple[float, float, str], list[float]] = {}
    for rate in RATES:
        for ratio in RATIOS:
            for seed in SEEDS:
                dataset = Dataset(ensure_rate_dataset(base_config, rate, ratio, seed))
                for spec in specs:
                    cost, mean_latency, migration_count = combined_seconds(dataset, spec)
                    combined.setdefault((rate, ratio, spec["id"]), []).append(cost)
                    latency.setdefault((rate, ratio, spec["id"]), []).append(mean_latency)
                    migrations.setdefault((rate, ratio, spec["id"]), []).append(migration_count)
            print(f"[ok]    rate={rate} ratio={ratio}: {len(SEEDS)} seeds x {len(specs)} policies")

    aggregate_rows = []
    comparison_rows = []
    for rate in RATES:
        for ratio in RATIOS:
            for spec in specs:
                key = (rate, ratio, spec["id"])
                aggregate_rows.append({
                    "rate_per_entity": rate,
                    "dominant_peer_ratio": ratio,
                    "policy_id": spec["id"],
                    "n_seeds": len(SEEDS),
                    "combined_s_alpha1_mean": f"{statistics.fmean(combined[key]):.6f}",
                    "combined_s_alpha1_std": f"{statistics.stdev(combined[key]):.6f}",
                    "mean_latency_ms_mean": f"{statistics.fmean(latency[key]):.6f}",
                    "migration_count_mean": f"{statistics.fmean(migrations[key]):.6f}",
                })
            best_follower = min((f"follow_w10_d{d}" for d in DWELLS),
                                key=lambda pid: statistics.fmean(combined[(rate, ratio, pid)]))
            best_baseline = min((spec["id"] for spec in BASELINES),
                                key=lambda pid: statistics.fmean(combined[(rate, ratio, pid)]))
            mean_diff, _std, low, high = paired_bootstrap(
                combined[(rate, ratio, best_follower)], combined[(rate, ratio, best_baseline)])
            comparison_rows.append({
                "rate_per_entity": rate,
                "dominant_peer_ratio": ratio,
                "best_follower": best_follower,
                "best_baseline": best_baseline,
                "n_seeds": len(SEEDS),
                "mean_diff_combined_s": f"{mean_diff:.6f}",
                "bootstrap_ci95_low": f"{low:.6f}",
                "bootstrap_ci95_high": f"{high:.6f}",
                "verdict": verdict(low, high),
            })
    return aggregate_rows, comparison_rows


def window_sweep() -> list[dict[str, Any]]:
    """Sweep 2: demand window size, on the paper's own traces (rate = 2)."""
    specs = BASELINES + [follower(size, dwell) for size in WINDOW_SIZES for dwell in DWELLS]
    combined: dict[tuple[float, str], list[float]] = {}
    for ratio in RATIOS:
        for seed in SEEDS:
            root = ambiguity_dataset_root(ratio, seed)
            if not (root / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"{root} has no manifest.json - run experiments/ambiguity_sweep.py first")
            dataset = Dataset(root)
            for spec in specs:
                cost, _latency, _migrations = combined_seconds(dataset, spec)
                combined.setdefault((ratio, spec["id"]), []).append(cost)
        print(f"[ok]    window sweep ratio={ratio}: {len(SEEDS)} seeds x {len(specs)} policies")

    rows = []
    for ratio in RATIOS:
        best_baseline = min((spec["id"] for spec in BASELINES),
                            key=lambda pid: statistics.fmean(combined[(ratio, pid)]))
        for size in WINDOW_SIZES:
            for dwell in DWELLS:
                key = (ratio, f"follow_w{size}_d{dwell}")
                _mean, _std, low, high = paired_bootstrap(
                    combined[key], combined[(ratio, best_baseline)])
                rows.append({
                    "dominant_peer_ratio": ratio,
                    "window_size": size,
                    "dwell_time_s": dwell,
                    "n_seeds": len(SEEDS),
                    "combined_s_alpha1_mean": f"{statistics.fmean(combined[key]):.6f}",
                    "combined_s_alpha1_std": f"{statistics.stdev(combined[key]):.6f}",
                    "beats_best_baseline": verdict(low, high),
                })
    return rows


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")

    rate_aggregate, rate_comparisons = rate_sweep(base_config)
    write_csv(OUTPUT_ROOT / "rate_aggregate.csv", RATE_AGGREGATE_COLUMNS, rate_aggregate)
    write_csv(OUTPUT_ROOT / "rate_comparisons.csv", RATE_COMPARISON_COLUMNS, rate_comparisons)

    window_rows = window_sweep()
    write_csv(OUTPUT_ROOT / "window_aggregate.csv", WINDOW_AGGREGATE_COLUMNS, window_rows)

    print("\nSWEEP 1: best interaction-following policy vs. best baseline, combined cost at alpha=1")
    print("(negative = following is cheaper; window fixed at 10 requests)")
    for row in rate_comparisons:
        print(f"  rate={row['rate_per_entity']:<5} ratio={row['dominant_peer_ratio']:<6} "
              f"{row['best_follower']:<18} vs {row['best_baseline']:<8} "
              f"{float(row['mean_diff_combined_s']):+9.1f}s "
              f"CI[{float(row['bootstrap_ci95_low']):+8.1f}, {float(row['bootstrap_ci95_high']):+8.1f}] "
              f"{row['verdict']}")

    print("\nSWEEP 2: combined cost by demand window size (seconds, rate = 2 as in the paper)")
    for ratio in RATIOS:
        print(f"  -- ratio {ratio}")
        header = "     window" + "".join(f"   d={d:<6}" for d in DWELLS)
        print(header)
        for size in WINDOW_SIZES:
            cells = []
            for dwell in DWELLS:
                row = next(r for r in window_rows
                           if r["dominant_peer_ratio"] == ratio and r["window_size"] == size
                           and r["dwell_time_s"] == dwell)
                mark = "*" if row["beats_best_baseline"] == "follower_wins" else " "
                cells.append(f"{float(row['combined_s_alpha1_mean']):9.1f}{mark}")
            print(f"     {size:<6}" + "".join(cells))
    print("\n     (* = significantly cheaper than the better of Static/Nearest, paired bootstrap)")
    print(f"\n[write] {OUTPUT_ROOT.relative_to(ROOT)}/{{rate_aggregate,rate_comparisons,window_aggregate}}.csv")


if __name__ == "__main__":
    main()
