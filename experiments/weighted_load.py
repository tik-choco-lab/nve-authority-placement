"""Processing load under the two placement rules of the paper's Section 4.

WHY THIS EXISTS. `load_significance.py` measures request imbalance for the
baselines and for the demand-argmax rule, which is what the paper reported
while argmax-following was the only proposed rule. Once weighted placement
(argmin over engagement-weighted round trip) became the paper's general rule,
its load went unmeasured -- and it is not a formality: minimising round trip to
the interaction window keeps naming the same network-central peers, so the rule
that is cheapest on the combined latency-plus-migration objective turns out to
be the WORST on load. That is the two-objective tension LB-Spiral bounds
jointly (Rai et al.), showing up in our measurements rather than in a bound,
and the paper's Section 5 quotes these numbers, so they need a script.

WHAT IT MEASURES. `imbalance` = maximum per-peer served requests over the mean
across all peers, idle capacity included (nve_policy/metrics.py), replayed on
the EXISTING datasets/ambiguity traces at the two ratios the paper quotes.
Policies are the ones Table I calls out, so the load column lines up with the
cost column row for row:

    ratio 0.85: static, nearest, argmax w=10 d=0, argmax w=5 d=0 (cheapest),
                weighted w=5 d=0 (cheapest)
    ratio 0.25: static, nearest, argmax w=10 d=0, argmax w=5 d=5 (cheapest),
                weighted w=100 d=1 (cheapest)

Differences against the baselines are seed-paired bootstraps, reusing
`load_significance.paired_bootstrap` rather than a second implementation.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import dataset_root
from experiments.common import EVALUATION, RELEVANT_RANGE
from experiments.load_significance import paired_bootstrap
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.metrics import summarize
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "weighted_load"
SEEDS = list(range(1, 31))
BASELINES = ["static", "nearest"]

SUMMARY_COLUMNS = ["dominant_peer_ratio", "policy_id", "n_seeds",
                   "request_imbalance_mean", "request_imbalance_std"]
COMPARISON_COLUMNS = ["dominant_peer_ratio", "policy_a", "policy_b", "n_seeds",
                      "mean_diff_imbalance", "bootstrap_ci95_low", "bootstrap_ci95_high",
                      "verdict"]


def follower(rule: str, window: int, dwell: float) -> dict[str, Any]:
    return {
        "id": f"{rule}_w{window}_d{dwell:g}", "type": "engagement_aware", "target_rule": rule,
        "engagement_threshold": 0.0, "dwell_time_s": dwell, "relevant_range": RELEVANT_RANGE,
        "window": {"type": "count", "size": window}, "use_interaction_weight": False,
    }


def specs_for(ratio: float) -> list[dict[str, Any]]:
    common = [{"id": "static", "type": "static"},
              {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
              follower("demand_argmax", 10, 0.0)]
    if ratio == 0.85:
        return common + [follower("demand_argmax", 5, 0.0), follower("rtt_weighted", 5, 0.0)]
    return common + [follower("demand_argmax", 5, 5.0), follower("rtt_weighted", 100, 1.0)]


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "a_more_imbalanced"
    if high < 0:
        return "a_less_imbalanced"
    return "not_significant"


def main() -> None:
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for ratio in (0.85, 0.25):
        specs = specs_for(ratio)
        per_policy: dict[str, list[float]] = {spec["id"]: [] for spec in specs}
        for seed in SEEDS:
            dataset = Dataset(dataset_root(ratio, seed))
            for spec in specs:
                outcome = replay(dataset, build_policy(dataset, spec), EVALUATION)
                metrics = summarize(dataset, spec["id"], spec, outcome)
                per_policy[spec["id"]].append(metrics["processing_load"]["request_imbalance"])

        for spec in specs:
            values = per_policy[spec["id"]]
            summary_rows.append({
                "dominant_peer_ratio": f"{ratio:g}", "policy_id": spec["id"],
                "n_seeds": len(values),
                "request_imbalance_mean": f"{statistics.mean(values):.6f}",
                "request_imbalance_std": f"{statistics.stdev(values):.6f}",
            })

        for spec in specs:
            if spec["id"] in BASELINES:
                continue
            for baseline in BASELINES:
                mean_diff, _, low, high = paired_bootstrap(per_policy[spec["id"]],
                                                           per_policy[baseline])
                comparison_rows.append({
                    "dominant_peer_ratio": f"{ratio:g}", "policy_a": spec["id"],
                    "policy_b": baseline, "n_seeds": len(SEEDS),
                    "mean_diff_imbalance": f"{mean_diff:.6f}",
                    "bootstrap_ci95_low": f"{low:.6f}", "bootstrap_ci95_high": f"{high:.6f}",
                    "verdict": verdict(low, high),
                })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "imbalance.csv"
    comparison_path = OUTPUT_ROOT / "comparisons.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    write_csv(comparison_path, COMPARISON_COLUMNS, comparison_rows)
    print(f"[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")
    print(f"[write] {comparison_path.relative_to(ROOT)} ({len(comparison_rows)} rows)")

    print("\nRequest imbalance (mean over 30 seeds; 1 is even, 20 is worst):")
    for row in summary_rows:
        print(f"  ratio {row['dominant_peer_ratio']:<5} {row['policy_id']:<24} "
              f"{float(row['request_imbalance_mean']):6.2f}")
    print("\nSeed-paired differences against the baselines:")
    for row in comparison_rows:
        print(f"  ratio {row['dominant_peer_ratio']:<5} {row['policy_a']:<24} "
              f"- {row['policy_b']:<8} {float(row['mean_diff_imbalance']):+6.2f} "
              f"[{float(row['bootstrap_ci95_low']):+6.2f},{float(row['bootstrap_ci95_high']):+6.2f}] "
              f"{row['verdict']}")


if __name__ == "__main__":
    main()
