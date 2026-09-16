"""Compare demand-argmax and RTT-weighted selection on the same window/dwell grid.

The dwell timer tracks the selected rule's target identity. Results for one
selector do not establish the effect of dwell for the other.

GRID. `ambiguity_sweep.py` and
`combined_objective.py` only ever swept theta_D on `target_rule:
demand_argmax` at a single window size (count=10,
`experiments/common.py::DEMAND_WINDOW`). `boundary_sensitivity.py` swept the
window (5/10/20/30) but, again, only for the argmax rule. This script crosses
selector (argmax vs. RTT-weighted) x theta_D (the same seven-point grid used
everywhere else in this directory) x window size, extended to
{5, 10, 20, 30, 100} -- the 100 arm matters here specifically because
`rtt_baseline.py` found the RTT-weighted rule wants a longer window than the
argmax rule (it averages a cost over interacting peers rather than picking a
single winner, so a short window makes it chase whoever spoke last) -- over
every `dominant_peer_ratio` directory `ambiguity_sweep.py` has already
generated and all 30 seeds. No new datasets are generated: the selector,
theta_D and window are all policy parameters, not generator parameters, so
the existing `datasets/ambiguity/ratio_*/seed_*` traces are replayed as-is.

theta_G (`engagement_threshold`) is pinned at 0 and theta_R (`relevant_range`)
beyond the field diagonal (`experiments.common.RELEVANT_RANGE = 1e9`,
world is 1000x1000 so the diagonal is ~1414), matching every other theta_D
sweep in this directory (`ambiguity_sweep.py`, `combined_objective.py`,
`rtt_baseline.py`) and the paper's own configuration: the shifting scenario
picks each entity's dominant peer independently of peer coordinates
(`nve_dataset/scenario/patterns.py`), so a finite theta_R would inject
release/candidacy noise unrelated to the dwell question this script asks.
`use_interaction_weight` is False throughout, again matching every other
sweep here (the dataset only has one request type, `config.yaml`, so the
weight flag is a no-op at these settings, but it is still a required spec
key and stated explicitly explicitly).

OBJECTIVE. The same combined objective as `combined_objective.py` --
sum(interaction_latency_ms) + alpha * sum(migration_network_latency_ms),
which is `nve_policy.policies.oracle.OraclePolicy`'s own DP objective at
alpha=1, `_verify_oracle_matches_engine_objective` there having already
checked once that the Oracle's DP and the engine's replay charge the
identical served/transition quantities -- reported here in seconds to match
Table I of the paper (`rtt_baseline.py` and `boundary_sensitivity.py` both
report the same objective in seconds for the same reason). Static and
Nearest are replayed on the same traces as the two baselines every other
sweep in this directory compares against.

OUTPUTS.

    datasets/weighted_dwell/aggregate.csv
        One row per (dominant_peer_ratio, policy_id): mean/std combined cost
        over 30 seeds, mean latency, mean migration count. selector, theta_d
        and window_size columns are blank for the two baselines (static,
        nearest), which carry neither.

    datasets/weighted_dwell/comparisons.csv
        Seed-paired bootstrap comparisons (`experiments/load_significance.py
        ::paired_bootstrap`, 10,000 resamples, reused rather than
        re-implemented, exactly as `combined_objective.py` /
        `rtt_baseline.py` / `boundary_sensitivity.py` all do), two kinds:

        (a) "rtt_weighted_vs_best_baseline": every (window, theta_D)
            rtt_weighted configuration against the better of static/nearest
            at that ratio (chosen by mean combined cost, per ratio).
        (b) "best_rtt_weighted_vs_best_demand_argmax": the single cheapest
            rtt_weighted configuration against the single cheapest
            demand_argmax configuration, per ratio, both selected by mean
            combined cost over the same (window, theta_D) grid.
        (c) "demand_argmax_vs_best_baseline": every (window, theta_D)
            demand_argmax configuration against the better baseline, per
            ratio - the same style of check as (a), for the argmax rule,
            over the full 5-window grid (this script's grid is wider than
            `boundary_sensitivity.py`'s w in {5,10,20,30}: it adds w=100).
        (d) "best_demand_argmax_vs_best_baseline": the single cheapest
            demand_argmax configuration against the better baseline, per
            ratio - the number the paper's boundary sentence ("no
            argmax-following configuration beats a static or proximity
            baseline below ratio ~0.30") needs, re-measured over the full
            grid rather than the w=10 headline table.

No raw per-request output is written : only these two aggregated
CSVs are produced, plus a stdout report answering the four questions this
script was commissioned to answer (does theta_D>0 ever help the RTT-weighted
rule; does a 0.25s optimum bottom out at the grid edge; which window is
cheapest per ratio and does the claimed short/long inversion survive a
5-point grid; and the exact combined-cost numbers at ratios 0.85/0.40/0.25).
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import dataset_root, RATIOS, SEEDS
from experiments.combined_objective import total_migration_rtt_ms
from experiments.common import EVALUATION, RELEVANT_RANGE
from experiments.load_significance import paired_bootstrap
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "weighted_dwell"

ALPHA = 1.0
SELECTORS = ["demand_argmax", "rtt_weighted"]
DWELL_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
WINDOW_SIZES = [5, 10, 20, 30, 100]
ENGAGEMENT_THRESHOLD = 0.0  # theta_G = 0, matching the paper's configuration

BASELINE_SPECS: list[dict[str, Any]] = [
    {"id": "static", "type": "static"},
    {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
]


def follower_id(selector: str, window_size: int, dwell: float) -> str:
    return f"{selector}_w{window_size}_d{dwell:g}"


def build_follower_specs() -> list[dict[str, Any]]:
    specs = []
    for selector in SELECTORS:
        for window_size in WINDOW_SIZES:
            for dwell in DWELL_VALUES:
                specs.append({
                    "id": follower_id(selector, window_size, dwell),
                    "type": "engagement_aware",
                    "target_rule": selector,
                    "engagement_threshold": ENGAGEMENT_THRESHOLD,
                    "dwell_time_s": dwell,
                    "relevant_range": RELEVANT_RANGE,
                    "window": {"type": "count", "size": window_size},
                    "use_interaction_weight": False,
                })
    return specs


ALL_SPECS = BASELINE_SPECS + build_follower_specs()

AGGREGATE_COLUMNS = [
    "dominant_peer_ratio", "policy_id", "selector", "theta_d", "window_size", "n_seeds",
    "combined_s_alpha1_mean", "combined_s_alpha1_std",
    "mean_latency_ms_mean", "migration_count_mean",
]
COMPARISON_COLUMNS = [
    "comparison_type", "dominant_peer_ratio", "policy_a", "policy_b", "n_seeds",
    "mean_diff_combined_s", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]


def combined_s(total_served_ms: float, total_mig_rtt_ms: float, alpha: float) -> float:
    return (total_served_ms + alpha * total_mig_rtt_ms) / 1000.0


def verdict(low: float, high: float) -> str:
    if high < 0:
        return "a_cheaper"
    if low > 0:
        return "a_worse"
    return "not_significant"


def main() -> None:
    print(f"[run] {len(ALL_SPECS)} policy specs ({len(BASELINE_SPECS)} baselines + "
          f"{len(SELECTORS)} selectors x {len(WINDOW_SIZES)} windows x {len(DWELL_VALUES)} theta_D) "
          f"x {len(RATIOS)} ratios x {len(SEEDS)} seeds, on existing datasets/ambiguity traces")

    # (ratio, policy_id) -> list of per-seed values, ordered by seed (paired_bootstrap needs this)
    combined_by_key: dict[tuple[float, str], list[float]] = {}
    latency_by_key: dict[tuple[float, str], list[float]] = {}
    migrations_by_key: dict[tuple[float, str], list[float]] = {}

    for ratio in RATIOS:
        for seed in SEEDS:
            root = dataset_root(ratio, seed)
            if not (root / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"{root} has no manifest.json - this script replays existing traces only; "
                    "run experiments/ambiguity_sweep.py first")
            dataset = Dataset(root)
            for spec in ALL_SPECS:
                outcome = replay(dataset, build_policy(dataset, spec), EVALUATION)
                served = sum(outcome["latencies"])
                mig_rtt = total_migration_rtt_ms(outcome["rows"])
                key = (ratio, spec["id"])
                combined_by_key.setdefault(key, []).append(combined_s(served, mig_rtt, ALPHA))
                latency_by_key.setdefault(key, []).append(statistics.fmean(outcome["latencies"]))
                migrations_by_key.setdefault(key, []).append(float(outcome["migrations"]))
        print(f"[ok]    dominant_peer_ratio={ratio}: {len(SEEDS)} seeds x {len(ALL_SPECS)} policies replayed")

    # ------------------------------------------------------------------
    # aggregate.csv
    # ------------------------------------------------------------------
    spec_by_id = {spec["id"]: spec for spec in ALL_SPECS}
    aggregate_rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for spec in ALL_SPECS:
            key = (ratio, spec["id"])
            combined = combined_by_key[key]
            latency = latency_by_key[key]
            migrations = migrations_by_key[key]
            is_follower = spec["type"] == "engagement_aware"
            aggregate_rows.append({
                "dominant_peer_ratio": ratio,
                "policy_id": spec["id"],
                "selector": spec["target_rule"] if is_follower else "",
                "theta_d": spec["dwell_time_s"] if is_follower else "",
                "window_size": spec["window"]["size"] if is_follower else "",
                "n_seeds": len(combined),
                "combined_s_alpha1_mean": f"{statistics.fmean(combined):.6f}",
                "combined_s_alpha1_std": f"{statistics.stdev(combined):.6f}",
                "mean_latency_ms_mean": f"{statistics.fmean(latency):.6f}",
                "migration_count_mean": f"{statistics.fmean(migrations):.6f}",
            })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_path = OUTPUT_ROOT / "aggregate.csv"
    write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregate_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(aggregate_rows)} rows)")

    # ------------------------------------------------------------------
    # comparisons.csv
    # ------------------------------------------------------------------
    def mean_combined(ratio: float, policy_id: str) -> float:
        return statistics.fmean(combined_by_key[(ratio, policy_id)])

    def configs_for(selector: str) -> list[str]:
        return [follower_id(selector, w, d) for w in WINDOW_SIZES for d in DWELL_VALUES]

    comparison_rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        best_baseline = min((spec["id"] for spec in BASELINE_SPECS),
                            key=lambda pid: mean_combined(ratio, pid))
        # (a) every rtt_weighted configuration vs. the better baseline
        for pid in configs_for("rtt_weighted"):
            mean_diff, _std, lo, hi = paired_bootstrap(
                combined_by_key[(ratio, pid)], combined_by_key[(ratio, best_baseline)])
            comparison_rows.append({
                "comparison_type": "rtt_weighted_vs_best_baseline",
                "dominant_peer_ratio": ratio,
                "policy_a": pid,
                "policy_b": best_baseline,
                "n_seeds": len(SEEDS),
                "mean_diff_combined_s": f"{mean_diff:.6f}",
                "bootstrap_ci95_low": f"{lo:.6f}",
                "bootstrap_ci95_high": f"{hi:.6f}",
                "verdict": verdict(lo, hi),
            })
        # (b) best rtt_weighted vs. best demand_argmax, per ratio
        best_rtt = min(configs_for("rtt_weighted"), key=lambda pid: mean_combined(ratio, pid))
        best_argmax = min(configs_for("demand_argmax"), key=lambda pid: mean_combined(ratio, pid))
        mean_diff, _std, lo, hi = paired_bootstrap(
            combined_by_key[(ratio, best_rtt)], combined_by_key[(ratio, best_argmax)])
        comparison_rows.append({
            "comparison_type": "best_rtt_weighted_vs_best_demand_argmax",
            "dominant_peer_ratio": ratio,
            "policy_a": best_rtt,
            "policy_b": best_argmax,
            "n_seeds": len(SEEDS),
            "mean_diff_combined_s": f"{mean_diff:.6f}",
            "bootstrap_ci95_low": f"{lo:.6f}",
            "bootstrap_ci95_high": f"{hi:.6f}",
            "verdict": verdict(lo, hi),
        })
        # (c) every demand_argmax configuration vs. the better baseline. Added
        # alongside (a) so the paper's argmax-vs-baseline boundary claim can be
        # re-checked over the SAME (window, theta_D) grid this script already
        # replayed for the RTT-weighted arm, rather than the narrower w in
        # {5,10,20,30} grid `boundary_sensitivity.py` used - this script's grid
        # additionally includes w=100.
        for pid in configs_for("demand_argmax"):
            mean_diff, _std, lo, hi = paired_bootstrap(
                combined_by_key[(ratio, pid)], combined_by_key[(ratio, best_baseline)])
            comparison_rows.append({
                "comparison_type": "demand_argmax_vs_best_baseline",
                "dominant_peer_ratio": ratio,
                "policy_a": pid,
                "policy_b": best_baseline,
                "n_seeds": len(SEEDS),
                "mean_diff_combined_s": f"{mean_diff:.6f}",
                "bootstrap_ci95_low": f"{lo:.6f}",
                "bootstrap_ci95_high": f"{hi:.6f}",
                "verdict": verdict(lo, hi),
            })
        # (d) best demand_argmax configuration (over the FULL grid, selected by
        # mean combined cost exactly like best_argmax above) vs. the better
        # baseline, per ratio - the single number the paper's boundary
        # sentence ("no argmax-following configuration beats a static or
        # proximity baseline") actually needs.
        mean_diff, _std, lo, hi = paired_bootstrap(
            combined_by_key[(ratio, best_argmax)], combined_by_key[(ratio, best_baseline)])
        comparison_rows.append({
            "comparison_type": "best_demand_argmax_vs_best_baseline",
            "dominant_peer_ratio": ratio,
            "policy_a": best_argmax,
            "policy_b": best_baseline,
            "n_seeds": len(SEEDS),
            "mean_diff_combined_s": f"{mean_diff:.6f}",
            "bootstrap_ci95_low": f"{lo:.6f}",
            "bootstrap_ci95_high": f"{hi:.6f}",
            "verdict": verdict(lo, hi),
        })

    comparisons_path = OUTPUT_ROOT / "comparisons.csv"
    write_csv(comparisons_path, COMPARISON_COLUMNS, comparison_rows)
    print(f"[write] {comparisons_path.relative_to(ROOT)} ({len(comparison_rows)} rows)")

    # ------------------------------------------------------------------
    # stdout report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Q(a): does theta_D > 0 ever win on the combined objective for rtt_weighted?")
    print("=" * 78)
    for ratio in RATIOS:
        best_per_window = {}
        for w in WINDOW_SIZES:
            best_d = min(DWELL_VALUES, key=lambda d: mean_combined(ratio, follower_id("rtt_weighted", w, d)))
            best_per_window[w] = best_d
        overall_best_pid = min(configs_for("rtt_weighted"), key=lambda pid: mean_combined(ratio, pid))
        print(f"  ratio={ratio:.2f}: best (window,theta_D) per window = "
              f"{ {w: best_per_window[w] for w in WINDOW_SIZES} }; "
              f"global best rtt_weighted config = {overall_best_pid} "
              f"({mean_combined(ratio, overall_best_pid):.3f}s)")

    print("\n" + "=" * 78)
    print("Q(c): cheapest window size for rtt_weighted, per ratio (theta_D best per window)")
    print("=" * 78)
    print(f"{'ratio':>6}  " + "".join(f"w={w:<9}" for w in WINDOW_SIZES))
    for ratio in RATIOS:
        cells = []
        for w in WINDOW_SIZES:
            best_d = min(DWELL_VALUES, key=lambda d: mean_combined(ratio, follower_id("rtt_weighted", w, d)))
            cells.append(f"{mean_combined(ratio, follower_id('rtt_weighted', w, best_d)):>9.2f}")
        print(f"{ratio:>6.2f}  " + "  ".join(cells))

    print("\n" + "=" * 78)
    print("Q(d): exact combined-cost numbers, best rtt_weighted config, ratios 0.85/0.40/0.25")
    print("=" * 78)
    for ratio in (0.85, 0.40, 0.25):
        best_pid = min(configs_for("rtt_weighted"), key=lambda pid: mean_combined(ratio, pid))
        vals = combined_by_key[(ratio, best_pid)]
        print(f"  ratio={ratio:.2f}: {best_pid:<22} mean={statistics.fmean(vals):.6f}s  "
              f"std={statistics.stdev(vals):.6f}s  n={len(vals)}")

    print("\n" + "=" * 78)
    print("GAP CHECK: does the best demand_argmax configuration (full w in "
          f"{WINDOW_SIZES} grid) beat the better baseline, per ratio?")
    print("=" * 78)
    closest_ratio, closest_abs_mid = None, None
    for ratio in RATIOS:
        best_baseline = min((spec["id"] for spec in BASELINE_SPECS),
                            key=lambda pid: mean_combined(ratio, pid))
        best_argmax = min(configs_for("demand_argmax"), key=lambda pid: mean_combined(ratio, pid))
        mean_diff, _std, lo, hi = paired_bootstrap(
            combined_by_key[(ratio, best_argmax)], combined_by_key[(ratio, best_baseline)])
        v = verdict(lo, hi)
        n_beat = sum(1 for pid in configs_for("demand_argmax")
                     if mean_combined(ratio, pid) < mean_combined(ratio, best_baseline))
        mid = abs((lo + hi) / 2.0)
        if closest_abs_mid is None or mid < closest_abs_mid:
            closest_abs_mid, closest_ratio = mid, ratio
        print(f"  ratio={ratio:.2f}: best_argmax={best_argmax:<22} ({mean_combined(ratio, best_argmax):>7.3f}s)  "
              f"best_baseline={best_baseline:<8} ({mean_combined(ratio, best_baseline):>7.3f}s)  "
              f"diff={mean_diff:+7.3f}s CI[{lo:+7.3f},{hi:+7.3f}]  {v}  "
              f"({n_beat}/{len(configs_for('demand_argmax'))} argmax configs beat best_baseline on point estimate)")
    print(f"\n  Closest-to-zero verdict is at ratio={closest_ratio:.2f}.")

    print(f"\n[done] {len(RATIOS) * len(SEEDS) * len(ALL_SPECS)} (ratio, seed, policy) replays reduced. "
          "See datasets/weighted_dwell/{aggregate,comparisons}.csv for full numbers.")


if __name__ == "__main__":
    main()
