"""Estimate seed-level uncertainty in processing load and dwell-time effects.

The concentrated workload uses 10 peers, 3 entities, duration 60 seconds,
dominant_peer_ratio=0.75 and 30 seeds. Policies replay the same trace for each
seed. Paired bootstrap intervals use 10,000 resamples and a fixed statistical
seed; zero excluded from the interval is treated as a significant difference.
The additional paired t statistic uses a normal approximation for its p value,
not a Student-t distribution lookup. mean_ci95 uses Student-t critical values.

The ambiguity analysis computes migration-reduction and latency-increase
percentages per seed before averaging. It evaluates ratio=0.25 with 30 seeds
using ambiguity_sweep's generation functions. Configurations, output columns,
and statistical implementations are defined below.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import (
    dataset_root as ambiguity_dataset_root,
    ensure_dataset as ambiguity_ensure_dataset,
)
from experiments.common import EVALUATION
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.metrics import summarize
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "load_significance"
SCENARIO = "concentrated"
SEEDS = list(range(1, 31))  # 30 seeds, vs. the 3 seeds behind the original claim

POLICY_SPECS: list[dict[str, Any]] = [
    {"id": "static", "type": "static"},
    {"id": "nearest_h0", "type": "nearest", "min_holding_time_s": 0.0},
    {
        "id": "interaction_aware_w10", "type": "interaction_aware",
        "window": {"type": "count", "size": 10}, "use_interaction_weight": False,
        "min_holding_time_s": 0.0,
    },
    {
        "id": "interaction_aware_w50", "type": "interaction_aware",
        "window": {"type": "count", "size": 50}, "use_interaction_weight": False,
        "min_holding_time_s": 0.0,
    },
]

SUMMARY_COLUMNS = [
    "seed", "policy_id", "total_requests", "mean_latency_ms", "migration_count",
    "max_requests_per_peer", "request_imbalance",
]
STATS_COLUMNS = [
    "policy_id", "n_seeds", "mean", "std", "ci95_low", "ci95_high",
]
COMPARISON_COLUMNS = [
    "comparison", "n_seeds", "mean_diff", "std_diff",
    "bootstrap_ci95_low", "bootstrap_ci95_high", "bootstrap_significant",
    "paired_t_stat", "paired_t_p_normal_approx",
]


# --- Part 2: ambiguity-sweep headline-number CIs (ratio=0.25 arm) ---
AMBIGUITY_SUMMARY_CSV = ROOT / "datasets" / "ambiguity" / "sweep_summary.csv"
AMBIGUITY_EXTENDED_CSV = ROOT / "datasets" / "ambiguity" / "ratio_025_seeds_06_30.csv"
HEADLINE_RATIO = 0.25
ORIGINAL_SEEDS = [1, 2, 3, 4, 5]
EXTENDED_SEEDS = list(range(6, 31))  # extends the ratio=0.25 arm from 5 to 30 seeds
DWELL_SPECS = {
    "engagement_g00_d0.0": {
        "id": "engagement_g00_d0.0", "type": "engagement_aware", "target_rule": "demand_argmax",
        "engagement_threshold": 0.0,
        "dwell_time_s": 0.0, "relevant_range": 1.0e9,
        "window": {"type": "count", "size": 10}, "use_interaction_weight": False,
    },
    "engagement_g00_d1.0": {
        "id": "engagement_g00_d1.0", "type": "engagement_aware", "target_rule": "demand_argmax",
        "engagement_threshold": 0.0,
        "dwell_time_s": 1.0, "relevant_range": 1.0e9,
        "window": {"type": "count", "size": 10}, "use_interaction_weight": False,
    },
    "engagement_g00_d2.0": {
        "id": "engagement_g00_d2.0", "type": "engagement_aware", "target_rule": "demand_argmax",
        "engagement_threshold": 0.0,
        "dwell_time_s": 2.0, "relevant_range": 1.0e9,
        "window": {"type": "count", "size": 10}, "use_interaction_weight": False,
    },
}
HEADLINE_EXTENDED_COLUMNS = [
    "dominant_peer_ratio", "seed", "policy_id",
    "total_requests", "mean_latency_ms", "migration_count",
]
HEADLINE_RATIO_COLUMNS = [
    "n_seeds", "quantity",
    "per_seed_mean", "per_seed_std", "per_seed_ci95_low", "per_seed_ci95_high",
    "aggregate_ratio_of_means",
]


def dataset_root(seed: int) -> Path:
    return OUTPUT_ROOT / SCENARIO / f"seed_{seed:03d}"


def ensure_dataset(base_config: dict[str, Any], seed: int) -> Path:
    root = dataset_root(seed)
    manifest_path = root / "manifest.json"
    if root.exists() and manifest_path.is_file():
        try:
            manifest_path.read_text(encoding="utf-8")
            print(f"[skip]  {root.relative_to(ROOT)} already exists, manifest readable")
            return root
        except OSError:
            pass
    run_config = config_for_run(base_config, SCENARIO, seed)
    print(f"[gen]   {root.relative_to(ROOT)} (scenario={SCENARIO}, seed={seed})")
    generate_dataset(run_config, output_dir=root, analyze=False)
    return root


# Two-sided 95% critical values of Student's t, indexed by degrees of freedom
# (n-1). scipy is not a dependency (pyproject.toml), so this is a plain table
# instead of scipy.stats.t.ppf. This matters: with n=5 (df=4) the correct
# critical value is 2.776 vs. the large-sample normal approximation of 1.96 --
# a 42% wider half-width. Using 1.96 at n=5 was tried first here and produced
# a materially narrower, WRONG interval for the ratio=0.25 headline-number
# check in `headline_analysis` (see its docstring): it made a CI that should
# straddle zero appear to exclude it. Values for df=1..30 (Fisher's table);
# df>30 falls back to the normal limit, where the two agree to <3%.
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical(df: int) -> float:
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96 if df > 30 else T_CRIT_95[max(1, df)]


def mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Returns (mean, std, ci_low, ci_high): a proper Student's t interval
    on the mean (see T_CRIT_95 above for why this is a table lookup rather
    than scipy.stats.t.ppf)."""
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n)
    t = t_critical(n - 1) if n > 1 else 0.0
    return mean, std, mean - t * se, mean + t * se


def paired_bootstrap(a: list[float], b: list[float], n_resamples: int = 10_000,
                      seed: int = 12345) -> tuple[float, float, float, float]:
    """Paired bootstrap over seeds for mean(a - b). `a` and `b` must be
    aligned by seed (same order). Returns (mean_diff, std_diff, ci_low, ci_high)
    of the *observed* difference and its bootstrap 95% CI.
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    resample_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.fmean(sample))
    resample_means.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples) - 1
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    return mean_diff, std_diff, resample_means[lo_idx], resample_means[hi_idx]


def paired_t_stat(a: list[float], b: list[float]) -> tuple[float, float]:
    """Classic paired t-statistic plus a normal-approximation two-sided
    p-value (erf-based), NOT a t-distribution p-value (would need scipy).
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = statistics.fmean(diffs)
    std_d = statistics.stdev(diffs) if n > 1 else 0.0
    if std_d == 0:
        return math.inf if mean_d != 0 else 0.0, 0.0 if mean_d != 0 else 1.0
    t = mean_d / (std_d / math.sqrt(n))
    p_two_sided = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p_two_sided


def read_ambiguity_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def generate_extended_ambiguity_seeds(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Seeds 6-30 of the ratio=0.25 arm, generated with `ambiguity_sweep.py`'s
    own dataset-generation code (same DURATION/PEER_COUNT/ENTITY_COUNT/
    PHASE_DURATION, only the seed differs) so the extension is drawn from
    exactly the same population as the original 5. Only the three dwell
    policies needed for the headline numbers are replayed, not the full
    18-policy comparison set, since that is all this question needs.
    """
    rows: list[dict[str, Any]] = []
    for seed in EXTENDED_SEEDS:
        root = ambiguity_ensure_dataset(base_config, HEADLINE_RATIO, seed)
        _ = ambiguity_dataset_root  # (imported for symmetry / documentation; root already resolved above)
        dataset = Dataset(root)
        for policy_id, spec in DWELL_SPECS.items():
            policy = build_policy(dataset, spec)
            outcome = replay(dataset, policy, EVALUATION)
            metrics = summarize(dataset, policy_id, spec, outcome)
            rows.append({
                "dominant_peer_ratio": HEADLINE_RATIO,
                "seed": seed,
                "policy_id": policy_id,
                "total_requests": metrics["total_requests"],
                "mean_latency_ms": f"{metrics['interaction_latency_ms']['mean']:.6f}",
                "migration_count": metrics["migration_count"],
            })
        print(f"[ok]    ambiguity ratio={HEADLINE_RATIO} seed={seed}: "
              f"{len(DWELL_SPECS)} dwell policies replayed")
    return rows


def per_seed_ratio_stats(by_seed_base: dict[str, dict[str, str]],
                          by_seed_other: dict[str, dict[str, str]],
                          field: str, mode: str) -> tuple[list[float], float, float, float, float]:
    """mode='reduction': (base-other)/base per seed (e.g. migration drop).
    mode='increase': (other-base)/base per seed (e.g. latency rise).
    Returns (per_seed_values, mean, std, ci_low, ci_high)."""
    values = []
    for seed in sorted(by_seed_base, key=int):
        base = float(by_seed_base[seed][field])
        other = float(by_seed_other[seed][field])
        values.append((base - other) / base if mode == "reduction" else (other - base) / base)
    mean, std, lo, hi = mean_ci95(values)
    return values, mean, std, lo, hi


def headline_analysis() -> None:
    print("\n" + "=" * 70)
    print("PART 2: paper's headline dwell-timer numbers (ambiguity ratio=0.25)")
    print("=" * 70)

    # ORIGINAL_SEEDS is a filter, not a comment. `ambiguity_sweep.py` has since
    # been extended from 5 seeds to 30, so the summary it writes already holds
    # every seed this file's "30 seeds" arm needs. Without the filter the
    # "5 seeds" arm silently becomes a second 30-seed arm and the two rows
    # report the same number under different labels.
    original_rows = [
        row for row in read_ambiguity_rows(AMBIGUITY_SUMMARY_CSV)
        if row["dominant_peer_ratio"] == str(HEADLINE_RATIO)
        and row["policy_id"] in DWELL_SPECS
        and int(row["seed"]) in set(ORIGINAL_SEEDS)
    ]

    base_config = load_config(ROOT / "config.yaml")
    if AMBIGUITY_EXTENDED_CSV.is_file():
        print(f"[skip]  {AMBIGUITY_EXTENDED_CSV.relative_to(ROOT)} already exists, reusing it")
        extended_rows = read_ambiguity_rows(AMBIGUITY_EXTENDED_CSV)
    else:
        extended_rows = generate_extended_ambiguity_seeds(base_config)
        write_csv(AMBIGUITY_EXTENDED_CSV, HEADLINE_EXTENDED_COLUMNS, extended_rows)
        print(f"[write] {AMBIGUITY_EXTENDED_CSV.relative_to(ROOT)} ({len(extended_rows)} rows)")

    all_rows = original_rows + extended_rows

    def by_seed(rows: list[dict[str, str]], policy_id: str) -> dict[str, dict[str, str]]:
        return {row["seed"]: row for row in rows if row["policy_id"] == policy_id}

    d0_5 = by_seed(original_rows, "engagement_g00_d0.0")
    d1_5 = by_seed(original_rows, "engagement_g00_d1.0")
    d2_5 = by_seed(original_rows, "engagement_g00_d2.0")
    d0_30 = by_seed(all_rows, "engagement_g00_d0.0")
    d1_30 = by_seed(all_rows, "engagement_g00_d1.0")
    d2_30 = by_seed(all_rows, "engagement_g00_d2.0")

    def aggregate_ratio_of_means(base: dict[str, dict[str, str]], other: dict[str, dict[str, str]],
                                  field: str, mode: str) -> float:
        base_mean = statistics.fmean(float(row[field]) for row in base.values())
        other_mean = statistics.fmean(float(row[field]) for row in other.values())
        return (base_mean - other_mean) / base_mean if mode == "reduction" else (other_mean - base_mean) / base_mean

    claims = [
        ("theta_D 0->1 migration reduction", d0_5, d1_5, d0_30, d1_30, "migration_count", "reduction"),
        ("theta_D 0->1 latency increase", d0_5, d1_5, d0_30, d1_30, "mean_latency_ms", "increase"),
        ("theta_D 0->2 migration reduction", d0_5, d2_5, d0_30, d2_30, "migration_count", "reduction"),
        ("theta_D 0->2 latency increase", d0_5, d2_5, d0_30, d2_30, "mean_latency_ms", "increase"),
    ]

    result_rows = []
    print(f"\n{'quantity':<34}{'n':>4}{'per-seed mean':>16}{'95% CI':>24}{'agg-of-means':>14}")
    for label, base5, other5, base30, other30, field, mode in claims:
        _, mean5, _, lo5, hi5 = per_seed_ratio_stats(base5, other5, field, mode)
        agg5 = aggregate_ratio_of_means(base5, other5, field, mode)
        print(f"  [5 seeds]  {label:<32}{5:>4}{mean5*100:>15.2f}%"
              f"  [{lo5*100:+.2f}%, {hi5*100:+.2f}%]{'':>1}{agg5*100:>10.2f}%")
        result_rows.append({
            "n_seeds": 5, "quantity": label,
            "per_seed_mean": f"{mean5:.6f}", "per_seed_std": "",
            "per_seed_ci95_low": f"{lo5:.6f}", "per_seed_ci95_high": f"{hi5:.6f}",
            "aggregate_ratio_of_means": f"{agg5:.6f}",
        })
        values30, mean30, std30, lo30, hi30 = per_seed_ratio_stats(base30, other30, field, mode)
        agg30 = aggregate_ratio_of_means(base30, other30, field, mode)
        print(f"  [30 seeds] {label:<32}{30:>4}{mean30*100:>15.2f}%"
              f"  [{lo30*100:+.2f}%, {hi30*100:+.2f}%]{'':>1}{agg30*100:>10.2f}%")
        result_rows.append({
            "n_seeds": 30, "quantity": label,
            "per_seed_mean": f"{mean30:.6f}", "per_seed_std": f"{std30:.6f}",
            "per_seed_ci95_low": f"{lo30:.6f}", "per_seed_ci95_high": f"{hi30:.6f}",
            "aggregate_ratio_of_means": f"{agg30:.6f}",
        })

    ratio_stats_path = OUTPUT_ROOT / "ambiguity_headline_ratio_stats.csv"
    write_csv(ratio_stats_path, HEADLINE_RATIO_COLUMNS, result_rows)
    print(f"\n[write] {ratio_stats_path.relative_to(ROOT)} ({len(result_rows)} rows)")


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")

    summary_rows: list[dict[str, Any]] = []
    per_policy_imbalance: dict[str, list[float]] = {spec["id"]: [] for spec in POLICY_SPECS}

    for seed in SEEDS:
        root = ensure_dataset(base_config, seed)
        dataset = Dataset(root)
        for spec in POLICY_SPECS:
            policy = build_policy(dataset, spec)
            outcome = replay(dataset, policy, EVALUATION)
            metrics = summarize(dataset, spec["id"], spec, outcome)
            imbalance = metrics["processing_load"]["request_imbalance"]
            summary_rows.append({
                "seed": seed,
                "policy_id": spec["id"],
                "total_requests": metrics["total_requests"],
                "mean_latency_ms": f"{metrics['interaction_latency_ms']['mean']:.6f}",
                "migration_count": metrics["migration_count"],
                "max_requests_per_peer": metrics["processing_load"]["max_requests_per_peer"],
                "request_imbalance": f"{imbalance:.6f}",
            })
            per_policy_imbalance[spec["id"]].append(imbalance)
        print(f"[ok]    seed={seed}: {len(POLICY_SPECS)} policies replayed")

    summary_path = OUTPUT_ROOT / "sweep_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"\n[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    # --- per-policy mean / std / 95% CI on request_imbalance ---
    stats_rows = []
    for spec in POLICY_SPECS:
        values = per_policy_imbalance[spec["id"]]
        mean, std, lo, hi = mean_ci95(values)
        stats_rows.append({
            "policy_id": spec["id"], "n_seeds": len(values),
            "mean": f"{mean:.6f}", "std": f"{std:.6f}",
            "ci95_low": f"{lo:.6f}", "ci95_high": f"{hi:.6f}",
        })
    stats_path = OUTPUT_ROOT / "request_imbalance_stats.csv"
    write_csv(stats_path, STATS_COLUMNS, stats_rows)
    print(f"[write] {stats_path.relative_to(ROOT)} ({len(stats_rows)} rows)")

    print(f"\n{'policy_id':<24}{'n':>4}{'mean':>10}{'std':>10}{'95% CI':>24}")
    for row in stats_rows:
        ci = f"[{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}]"
        print(f"{row['policy_id']:<24}{row['n_seeds']:>4}{float(row['mean']):>10.4f}"
              f"{float(row['std']):>10.4f}{ci:>24}")

    # --- paired comparisons: interaction_aware_w10 vs static, vs nearest_h0 ---
    comparisons = [
        ("interaction_aware_w10", "static"),
        ("interaction_aware_w10", "nearest_h0"),
    ]
    comparison_rows = []
    print("\nPaired comparisons on request_imbalance (paired by seed):")
    for name_a, name_b in comparisons:
        a = per_policy_imbalance[name_a]
        b = per_policy_imbalance[name_b]
        mean_diff, std_diff, boot_lo, boot_hi = paired_bootstrap(a, b)
        t_stat, p_norm = paired_t_stat(a, b)
        significant = not (boot_lo <= 0.0 <= boot_hi)
        comparison_rows.append({
            "comparison": f"{name_a} - {name_b}",
            "n_seeds": len(a),
            "mean_diff": f"{mean_diff:.6f}",
            "std_diff": f"{std_diff:.6f}",
            "bootstrap_ci95_low": f"{boot_lo:.6f}",
            "bootstrap_ci95_high": f"{boot_hi:.6f}",
            "bootstrap_significant": significant,
            "paired_t_stat": f"{t_stat:.6f}",
            "paired_t_p_normal_approx": f"{p_norm:.6f}",
        })
        verdict = "SIGNIFICANT (CI excludes 0)" if significant else "NOT significant (CI includes 0)"
        print(f"  {name_a} - {name_b}: mean_diff={mean_diff:+.4f}  "
              f"bootstrap 95% CI=[{boot_lo:+.4f}, {boot_hi:+.4f}]  "
              f"paired t={t_stat:+.3f} (p~{p_norm:.3f}, normal approx)  -> {verdict}")

    comparison_path = OUTPUT_ROOT / "paired_comparisons.csv"
    write_csv(comparison_path, COMPARISON_COLUMNS, comparison_rows)
    print(f"\n[write] {comparison_path.relative_to(ROOT)} ({len(comparison_rows)} rows)")

    print(
        "\nVerdict: the paper's 3.44/3.41/3.38 ordering was read off a single "
        "seed (seed_001 of a 3-seed dataset). Over 30 seeds of the identical "
        "configuration, see the CIs and paired comparisons above for whether "
        "the ordering is distinguishable from noise."
    )


if __name__ == "__main__":
    main()
    headline_analysis()
