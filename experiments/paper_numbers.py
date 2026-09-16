"""Report seed-paired margins and stretch for the combined_objective policy set.

Reads saved summary/aggregate CSVs without generating traces or replaying policies.
The ratio=0.40 comparisons use alpha=1 and paired bootstrap confidence intervals.
Stretch uses only the online policies in combined_objective, not the full
window/dwell grid in weighted_dwell; it is not Table I's full-grid stretch.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from experiments.load_significance import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "datasets" / "combined_objective" / "summary.csv"

# The operating point Section V-D selects: inside the band where a dwell time is
# the cheapest online policy, priced at the alpha the Oracle itself uses.
BAND_RATIO = 0.40
ALPHA_KEY = "combined_ms_alpha1"
ORACLE_ID = "oracle_a1"

# Section V-D names these three ratios: above the boundary, inside the band, below it.
STRETCH_RATIOS = [0.85, 0.40, 0.25]

PROPOSED = "engagement_g00_d0.25"  # theta_D = 0.25 s, theta_G = 0
LABELS = {
    "engagement_g00_d0.0": "theta_D=0",
    "engagement_g00_d0.25": "theta_D=0.25s",
    "engagement_g00_d1.0": "theta_D=1s",
    "static": "Static",
    "nearest": "Nearest",
}


def load_ambiguity_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["sweep"] == "ambiguity"]


def series(rows: list[dict[str, str]], ratio: float, policy_id: str,
           column: str) -> list[float]:
    """One value per seed, ordered by seed, so two series are paired positionally."""
    picked = [r for r in rows
              if float(r["knob_value"]) == ratio and r["policy_id"] == policy_id]
    picked.sort(key=lambda r: int(r["seed"]))
    return [float(r[column]) for r in picked]


def report_margins(rows: list[dict[str, str]]) -> None:
    print("=" * 92)
    print(f"SECTION V-D: seed-paired margins at dominant_peer_ratio={BAND_RATIO}, alpha=1")
    print("=" * 92)

    proposed = series(rows, BAND_RATIO, PROPOSED, ALPHA_KEY)
    print(f"{len(proposed)} seeds; combined objective in SECONDS "
          f"(negative margin = {LABELS[PROPOSED]} is cheaper)\n")
    print(f"{'comparison':<34}{'margin (s)':>12}  {'95% CI (s)':>20}  verdict")

    for other in ["engagement_g00_d0.0", "engagement_g00_d1.0", "static", "nearest"]:
        b = series(rows, BAND_RATIO, other, ALPHA_KEY)
        mean_diff, _, lo, hi = paired_bootstrap(proposed, b)
        verdict = "separable" if hi < 0 or lo > 0 else "inside the noise"
        print(f"{LABELS[PROPOSED] + ' vs ' + LABELS[other]:<34}"
              f"{mean_diff / 1000:>12.2f}  [{lo / 1000:>8.2f},{hi / 1000:>8.2f}]  {verdict}")

    # The baselines against each other: the paper says the two are not separable,
    # which is why it reports a single "better baseline" margin rather than two.
    a = series(rows, BAND_RATIO, "static", ALPHA_KEY)
    b = series(rows, BAND_RATIO, "nearest", ALPHA_KEY)
    mean_diff, _, lo, hi = paired_bootstrap(a, b)
    verdict = "separable" if hi < 0 or lo > 0 else "inside the noise"
    print(f"{'Static vs Nearest':<34}{mean_diff / 1000:>12.2f}  "
          f"[{lo / 1000:>8.2f},{hi / 1000:>8.2f}]  {verdict}")

    # Where the margin comes from: fewer migrations bought with a little latency.
    # Computed per seed and then averaged, not as a ratio of the two means.
    print()
    base = "engagement_g00_d0.0"
    for column, name in [("migration_count", "migrations removed"),
                         ("total_served_ms", "interaction latency added")]:
        p = series(rows, BAND_RATIO, PROPOSED, column)
        z = series(rows, BAND_RATIO, base, column)
        pct = [100.0 * (x - y) / y for x, y in zip(p, z)]
        mean_pct, _, lo, hi = paired_bootstrap(pct, [0.0] * len(pct))
        print(f"{name:<34}{mean_pct:>11.1f}%  [{lo:>8.1f},{hi:>8.1f}]  "
              f"(vs {LABELS[base]}, per seed)")
    print()


def report_stretch(rows: list[dict[str, str]]) -> None:
    print("=" * 92)
    print("SECTION V-D: stretch = cheapest online policy / offline optimum, alpha=1")
    print("=" * 92)
    print(f"{'ratio':>6}  {'cheapest online policy':<34}{'online (s)':>12}"
          f"{'oracle (s)':>12}{'stretch':>10}")

    for ratio in STRETCH_RATIOS:
        cell = [r for r in rows if float(r["knob_value"]) == ratio]
        # Sorted, not a set: `naive_target_aware` and `engagement_g00_d0.0` are
        # provably the same policy and so tie to the last decimal. Picking the
        # argmin out of set iteration order would print a different name on
        # every run, because the tie is exact and set order is not stable.
        online_ids = sorted({r["policy_id"] for r in cell
                             if not r["policy_id"].startswith("oracle_")})
        means = {pid: statistics.fmean(series(rows, ratio, pid, ALPHA_KEY))
                 for pid in online_ids}
        best_cost = min(means.values())
        tied = [pid for pid in online_ids if means[pid] == best_cost]
        label = tied[0] if len(tied) == 1 else f"{tied[0]} (+{len(tied) - 1} tied)"
        oracle = statistics.fmean(series(rows, ratio, ORACLE_ID, ALPHA_KEY))
        print(f"{ratio:>6.2f}  {label:<34}{best_cost / 1000:>12.1f}"
              f"{oracle / 1000:>12.1f}{best_cost / oracle:>10.2f}")
    print()


def main() -> None:
    rows = load_ambiguity_rows()
    report_margins(rows)
    report_stretch(rows)


if __name__ == "__main__":
    main()
