"""Does interaction-following still balance processing load when several
entities engage the SAME player?

WHY THIS EXISTS. Section V-F of the paper argues that the expected load
concern is reversed: following interaction spreads authority because moving
targets carry load with them, and it backs this on a `concentrated` workload
(10 peers, 3 entities, fixed dominant peers, 30 seeds) where interaction-
following reaches an imbalance of 3.55 against Static's 4.11 and Nearest's
4.60.

That workload cannot express the case the concern is actually about.
`nve_dataset/scenario/patterns.py::build_pattern` assigns dominant peers by
round robin over a shuffled peer list, so with at least as many peers as
entities - true of every dataset here - each entity gets a DIFFERENT dominant
peer, and "authority follows demand" spreads three entities over three peers
by construction. The adversarial case is the popular player: several
entities engaging one peer at once, where following demand piles all of them
onto it while Static and Nearest are unaffected. The paper hedges this
("stronger player concentration may reverse the result") without measuring
it.

WHAT IT DOES. Reruns that exact configuration twice - once as published
(distinct dominant peers) and once with `interaction.shared_dominant_peer`
enabled, the opt-in generator flag that collapses every entity's dominant
peer onto one - and compares processing load. Two load measures are
reported, because the paper's is a whole-run average and the concern is
partly about transient pile-up:

    request_imbalance      max requests served by any peer / mean over all
                           peers, over the whole run (the paper's measure,
                           `nve_policy/metrics.py`); 1 is even, PEER_COUNT
                           is one peer serving everything

    peak_window_imbalance  the same ratio computed inside each 10 s window
                           and maximised over windows, so a policy that is
                           even on average but concentrates in bursts cannot
                           hide behind the run-long mean

Differences are seed-paired bootstraps over 30 seeds, matching
`experiments/load_significance.py`, whose numbers this script reproduces in
its `distinct` arm.
"""
from __future__ import annotations

import copy
import statistics
from pathlib import Path
from typing import Any

from experiments.common import EVALUATION, RELEVANT_RANGE
from experiments.load_significance import paired_bootstrap
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "shared_dominant"

# config.yaml's defaults, which are also load_significance.py's configuration
# and the one behind the paper's Section V-F numbers.
SCENARIO = "concentrated"
DURATION = 60.0
PEER_COUNT = 10
ENTITY_COUNT = 3
SEEDS = list(range(1, 31))
WINDOW_SECONDS = 10.0

POLICY_SPECS: list[dict[str, Any]] = [
    {"id": "static", "type": "static"},
    {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
]
for _dwell in (0.0, 1.0):
    POLICY_SPECS.append({
        "id": f"follow_d{_dwell}", "type": "engagement_aware",
        "target_rule": "demand_argmax",
        "engagement_threshold": 0.0, "dwell_time_s": _dwell,
        "relevant_range": RELEVANT_RANGE, "window": {"type": "count", "size": 10},
        "use_interaction_weight": False,
    })

SUMMARY_COLUMNS = [
    "assignment", "seed", "policy_id", "total_requests",
    "mean_latency_ms", "migration_count", "request_imbalance", "peak_window_imbalance",
]
AGGREGATE_COLUMNS = [
    "assignment", "policy_id", "n_seeds",
    "request_imbalance_mean", "request_imbalance_std",
    "peak_window_imbalance_mean", "peak_window_imbalance_std",
    "mean_latency_ms_mean",
]
COMPARISON_COLUMNS = [
    "assignment", "metric", "policy_a", "policy_b", "n_seeds",
    "mean_diff", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]


def dataset_root(shared: bool, seed: int) -> Path:
    return OUTPUT_ROOT / ("shared" if shared else "distinct") / f"seed_{seed:03d}"


def ensure_dataset(base_config: dict[str, Any], shared: bool, seed: int) -> Path:
    root = dataset_root(shared, seed)
    if (root / "manifest.json").is_file():
        return root
    config = copy.deepcopy(base_config)
    config["simulation"]["duration"] = DURATION
    config["peers"]["count"] = PEER_COUNT
    config["entities"]["count"] = ENTITY_COUNT
    if shared:
        config["interaction"]["shared_dominant_peer"] = True
    generate_dataset(config_for_run(config, SCENARIO, seed), output_dir=root, analyze=False)
    return root


def peak_window_imbalance(rows: list[dict[str, Any]], peer_index: dict[str, int]) -> float:
    """Worst per-window imbalance, windows of WINDOW_SECONDS. Counted on
    `authority_before`, the peer that actually served the request - the same
    column `served_per_peer` in nve_policy/engine.py counts."""
    windows: dict[int, list[int]] = {}
    for row in rows:
        index = int(float(row["timestamp"]) // WINDOW_SECONDS)
        served = windows.setdefault(index, [0] * len(peer_index))
        served[peer_index[row["authority_before"]]] += 1
    worst = 0.0
    for served in windows.values():
        mean = sum(served) / len(served)
        if mean:
            worst = max(worst, max(served) / mean)
    return worst


def verdict(low: float, high: float) -> str:
    if high < 0:
        return "follower_better"
    if low > 0:
        return "follower_worse"
    return "not_significant"


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")
    summary_rows: list[dict[str, Any]] = []
    by_seed: dict[tuple[str, str, str], list[float]] = {}

    for shared in (False, True):
        assignment = "shared" if shared else "distinct"
        for seed in SEEDS:
            dataset = Dataset(ensure_dataset(base_config, shared, seed))
            peer_index = {peer: index for index, peer in enumerate(dataset.peers)}
            for spec in POLICY_SPECS:
                outcome = replay(dataset, build_policy(dataset, spec), EVALUATION)
                served = outcome["served_per_peer"]
                imbalance = max(served) / (sum(served) / len(served))
                peak = peak_window_imbalance(outcome["rows"], peer_index)
                latency = statistics.fmean(outcome["latencies"])
                by_seed.setdefault((assignment, spec["id"], "request_imbalance"), []).append(imbalance)
                by_seed.setdefault((assignment, spec["id"], "peak_window_imbalance"), []).append(peak)
                by_seed.setdefault((assignment, spec["id"], "mean_latency_ms"), []).append(latency)
                summary_rows.append({
                    "assignment": assignment,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": len(outcome["latencies"]),
                    "mean_latency_ms": f"{latency:.6f}",
                    "migration_count": outcome["migrations"],
                    "request_imbalance": f"{imbalance:.6f}",
                    "peak_window_imbalance": f"{peak:.6f}",
                })
        print(f"[ok]    {assignment} dominant peers: {len(SEEDS)} seeds x {len(POLICY_SPECS)} policies")

    aggregate_rows = []
    for assignment in ("distinct", "shared"):
        for spec in POLICY_SPECS:
            imbalance = by_seed[(assignment, spec["id"], "request_imbalance")]
            peak = by_seed[(assignment, spec["id"], "peak_window_imbalance")]
            aggregate_rows.append({
                "assignment": assignment,
                "policy_id": spec["id"],
                "n_seeds": len(SEEDS),
                "request_imbalance_mean": f"{statistics.fmean(imbalance):.6f}",
                "request_imbalance_std": f"{statistics.stdev(imbalance):.6f}",
                "peak_window_imbalance_mean": f"{statistics.fmean(peak):.6f}",
                "peak_window_imbalance_std": f"{statistics.stdev(peak):.6f}",
                "mean_latency_ms_mean": f"{statistics.fmean(by_seed[(assignment, spec['id'], 'mean_latency_ms')]):.6f}",
            })

    comparison_rows = []
    for assignment in ("distinct", "shared"):
        for metric in ("request_imbalance", "peak_window_imbalance"):
            for follower in ("follow_d0.0", "follow_d1.0"):
                for baseline in ("static", "nearest"):
                    mean_diff, _std, low, high = paired_bootstrap(
                        by_seed[(assignment, follower, metric)],
                        by_seed[(assignment, baseline, metric)])
                    comparison_rows.append({
                        "assignment": assignment,
                        "metric": metric,
                        "policy_a": follower,
                        "policy_b": baseline,
                        "n_seeds": len(SEEDS),
                        "mean_diff": f"{mean_diff:.6f}",
                        "bootstrap_ci95_low": f"{low:.6f}",
                        "bootstrap_ci95_high": f"{high:.6f}",
                        "verdict": verdict(low, high),
                    })

    write_csv(OUTPUT_ROOT / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(OUTPUT_ROOT / "aggregate.csv", AGGREGATE_COLUMNS, aggregate_rows)
    write_csv(OUTPUT_ROOT / "comparisons.csv", COMPARISON_COLUMNS, comparison_rows)

    print(f"\nprocessing load, {PEER_COUNT} peers / {ENTITY_COUNT} entities / {len(SEEDS)} seeds")
    print(f"{'assignment':<11}{'policy':<13}{'run imbalance':>15}{'peak 10s':>11}{'lat ms':>9}")
    for row in aggregate_rows:
        print(f"{row['assignment']:<11}{row['policy_id']:<13}"
              f"{float(row['request_imbalance_mean']):>15.2f}"
              f"{float(row['peak_window_imbalance_mean']):>11.2f}"
              f"{float(row['mean_latency_ms_mean']):>9.1f}")
    print("\nfollower minus baseline (paired bootstrap; negative = follower spreads load better)")
    for row in comparison_rows:
        print(f"  {row['assignment']:<9}{row['metric']:<23}{row['policy_a']:<12} - {row['policy_b']:<8}"
              f"{float(row['mean_diff']):+7.2f} "
              f"CI[{float(row['bootstrap_ci95_low']):+6.2f}, {float(row['bootstrap_ci95_high']):+6.2f}] "
              f"{row['verdict']}")
    print(f"\n[write] {OUTPUT_ROOT.relative_to(ROOT)}/{{summary,aggregate,comparisons}}.csv")


if __name__ == "__main__":
    main()
