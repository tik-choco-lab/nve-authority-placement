"""Summarize ambiguity-sweep results and sensitivity to the seed subset.

Reads sweep_summary.csv and sweep_aggregate.csv without replaying policies.
Writes tables for ratio=0.25, per-seed percentage changes with Student-t
intervals, and migration counts relative to the target-change reference for
five versus thirty seeds. These are descriptive analyses, not a claim that
old manuscript quotations match the current paper.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.load_significance import mean_ci95
from nve_dataset.util import write_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "ambiguity"
SUMMARY_CSV = OUTPUT_ROOT / "sweep_summary.csv"
AGGREGATE_CSV = OUTPUT_ROOT / "sweep_aggregate.csv"

HEADLINE_RATIO = 0.25
FIVE_SEED_SUBSET = {1, 2, 3, 4, 5}
MIGRATION_FLOOR = 150.0  # (duration/phase_duration)*entity_count = (300/10)*5, constant across the sweep


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    summary = read_rows(SUMMARY_CSV)
    aggregate = read_rows(AGGREGATE_CSV)
    n_seeds_total = len({row["seed"] for row in summary})
    print(f"[info] sweep_summary.csv: {len(summary)} rows, {n_seeds_total} distinct seeds, "
          f"ratios={sorted({row['dominant_peer_ratio'] for row in summary})}")

    agg_by_key = {(row["dominant_peer_ratio"], row["policy_id"]): row for row in aggregate}

    def summary_subset(ratio: float, policy_id: str, seeds: set[int] | None = None) -> list[dict[str, str]]:
        rows = [r for r in summary if r["dominant_peer_ratio"] == str(ratio) and r["policy_id"] == policy_id]
        if seeds is not None:
            rows = [r for r in rows if int(r["seed"]) in seeds]
        return rows

    def field_values(rows: list[dict[str, str]], field: str) -> list[float]:
        return [float(r[field]) for r in rows]

    def mean_of(ratio: float, policy_id: str, field: str, seeds: set[int] | None = None) -> float:
        rows = summary_subset(ratio, policy_id, seeds)
        return statistics.fmean(field_values(rows, field))

    # =====================================================================
    # TABLE 1: theta_D in {0,1,2,5} for the proposed policy, plus
    # Static / Nearest / Oracle, at ratio=0.25, 30 seeds -- straight from
    # the canonical aggregate CSV ambiguity_sweep.py just wrote.
    # =====================================================================
    table1_specs = [
        ("Proposed, theta_D=0", "engagement_g00_d0.0"),
        ("Proposed, theta_D=1", "engagement_g00_d1.0"),
        ("Proposed, theta_D=2", "engagement_g00_d2.0"),
        ("Proposed, theta_D=5", "engagement_g00_d5.0"),
        ("Static", "static"),
        ("Nearest", "nearest"),
        ("Oracle (alpha=1)", "oracle_a1"),
    ]
    print("\n" + "=" * 78)
    print(f"TABLE 1: ratio={HEADLINE_RATIO}, n_seeds=30 (from sweep_aggregate.csv)")
    print("=" * 78)
    print(f"{'policy':<24}{'migration_count':>18}{'mean_latency_ms':>18}{'p95_latency_ms':>18}")
    table1_rows = []
    for label, policy_id in table1_specs:
        row = agg_by_key[(str(HEADLINE_RATIO), policy_id)]
        mig = float(row["migration_count_mean"])
        lat = float(row["mean_latency_ms_mean"])
        p95 = float(row["p95_latency_ms_mean"])
        print(f"{label:<24}{mig:>18.1f}{lat:>18.2f}{p95:>18.2f}")
        table1_rows.append({
            "policy": label, "policy_id": policy_id, "n_seeds": 30,
            "migration_count_mean": f"{mig:.4f}", "mean_latency_ms_mean": f"{lat:.4f}",
            "p95_latency_ms_mean": f"{p95:.4f}",
        })
    write_csv(OUTPUT_ROOT / "theta_d_table_ratio025_30seed.csv",
              ["policy", "policy_id", "n_seeds", "migration_count_mean", "mean_latency_ms_mean", "p95_latency_ms_mean"],
              table1_rows)
    print(f"[write] datasets/ambiguity/theta_d_table_ratio025_30seed.csv ({len(table1_rows)} rows)")

    # =====================================================================
    # TABLE 2: per-seed migration reduction / latency increase, theta_D=1
    # and theta_D=2 vs theta_D=0, at ratio=0.25 -- 5-seed-subset AND
    # 30-seed-full, both from this same summary file, with 95% CI.
    # =====================================================================
    def per_seed_pct(base_rows: list[dict[str, str]], other_rows: list[dict[str, str]],
                      field: str, mode: str) -> list[float]:
        base_by_seed = {int(r["seed"]): float(r[field]) for r in base_rows}
        other_by_seed = {int(r["seed"]): float(r[field]) for r in other_rows}
        values = []
        for seed in sorted(base_by_seed):
            b, o = base_by_seed[seed], other_by_seed[seed]
            values.append((b - o) / b if mode == "reduction" else (o - b) / b)
        return values

    print("\n" + "=" * 78)
    print(f"TABLE 2: theta_D reduction/increase vs theta_D=0, ratio={HEADLINE_RATIO}, per-seed then aggregated")
    print("=" * 78)
    print(f"{'quantity':<38}{'n':>4}{'mean':>10}{'95% CI':>22}")
    table2_rows = []
    claims = [
        ("theta_D=1 migration reduction", "engagement_g00_d1.0", "migration_count", "reduction"),
        ("theta_D=1 latency increase", "engagement_g00_d1.0", "mean_latency_ms", "increase"),
        ("theta_D=2 migration reduction", "engagement_g00_d2.0", "migration_count", "reduction"),
        ("theta_D=2 latency increase", "engagement_g00_d2.0", "mean_latency_ms", "increase"),
    ]
    base30 = summary_subset(HEADLINE_RATIO, "engagement_g00_d0.0")
    base5 = summary_subset(HEADLINE_RATIO, "engagement_g00_d0.0", FIVE_SEED_SUBSET)
    for label, other_id, field, mode in claims:
        other30 = summary_subset(HEADLINE_RATIO, other_id)
        other5 = summary_subset(HEADLINE_RATIO, other_id, FIVE_SEED_SUBSET)
        for tag, base_rows, other_rows, n in (("5-seed subset", base5, other5, 5), ("30-seed full", base30, other30, 30)):
            values = per_seed_pct(base_rows, other_rows, field, mode)
            mean, std, lo, hi = mean_ci95(values)
            print(f"  [{tag:<13}] {label:<24}{n:>4}{mean*100:>9.2f}%  [{lo*100:+.2f}%, {hi*100:+.2f}%]")
            table2_rows.append({
                "quantity": label, "n_seeds": n, "mean_pct": f"{mean*100:.4f}",
                "std_pct": f"{std*100:.4f}", "ci95_low_pct": f"{lo*100:.4f}", "ci95_high_pct": f"{hi*100:.4f}",
            })
    write_csv(OUTPUT_ROOT / "headline_pct_ci_ratio025.csv",
              ["quantity", "n_seeds", "mean_pct", "std_pct", "ci95_low_pct", "ci95_high_pct"],
              table2_rows)
    print(f"[write] datasets/ambiguity/headline_pct_ci_ratio025.csv ({len(table2_rows)} rows)")

    # =====================================================================
    # TABLE 3: migration floor gap (naive_target_aware / engagement_g00_d0.0
    # migration_count - 150), all 7 ratios, 5-seed-subset vs 30-seed-full.
    # =====================================================================
    print("\n" + "=" * 78)
    print("TABLE 3: migration floor gap (naive_target_aware - 150), all ratios")
    print("=" * 78)
    print(f"{'ratio':>7}{'5-seed gap':>14}{'30-seed gap':>14}{'change':>10}")
    ratios = sorted({float(r["dominant_peer_ratio"]) for r in summary}, reverse=True)
    table3_rows = []
    for ratio in ratios:
        gap5 = mean_of(ratio, "naive_target_aware", "migration_count", FIVE_SEED_SUBSET) - MIGRATION_FLOOR
        gap30 = mean_of(ratio, "naive_target_aware", "migration_count") - MIGRATION_FLOOR
        change = "n/a" if gap5 == 0 else f"{(gap30 - gap5) / gap5 * 100:+.1f}%"
        print(f"{ratio:>7.2f}{gap5:>14.1f}{gap30:>14.1f}{change:>10}")
        table3_rows.append({
            "dominant_peer_ratio": ratio, "gap_5seed": f"{gap5:.4f}", "gap_30seed": f"{gap30:.4f}",
            "relative_change": change,
        })
    write_csv(OUTPUT_ROOT / "migration_floor_gap_5v30.csv",
              ["dominant_peer_ratio", "gap_5seed", "gap_30seed", "relative_change"], table3_rows)
    print(f"[write] datasets/ambiguity/migration_floor_gap_5v30.csv ({len(table3_rows)} rows)")


if __name__ == "__main__":
    main()
