"""Compare dimensionless benefit/cost measures across request rates.

Only the 2 req/s sweep samples both signs of the cost margin. The 0.5 and
8 req/s sweeps alone do not establish a shared crossover. The axis uses
measured outcomes and is an explanatory summary, not independent prediction.

THE AXIS. A migration costs one round trip -- a fixed price, independent of
how many requests follow it -- while the latency saved by interaction-
following accrues per request the new holder then serves locally
afterwards. Per phase (the interval at which an entity's target changes,
`PHASE_DURATION = 10s` in `boundary_sensitivity.py`), interaction-following
pays for itself iff

    rho_eff = (s - s0) * lambda * w / (alpha * m)  >  1

where `lambda` is the request rate per entity, `w` the phase duration,
`alpha = 1`, `s` the best follower's self-served request fraction, `s0` the
better baseline's self-served fraction, and `m` the follower's migrations per
entity per phase. There is also a coarser, design-time version that uses the
generator's dominant peer ratio `r` as a stand-in for the measured
self-service share instead of measuring it:

    rho = r * lambda * w / alpha

A preliminary check inferred `s` from mean latency via the self-service
model's own inversion (`experiments/self_service.py`'s
`mean_latency = (1-s)*E[RTT|remote] + delta`, solved for `s` from a single
assumed `E[RTT|remote]=105ms`). That inversion is one algebraic step removed
from a measured `s`, and folds any policy-to-policy difference in
`E[RTT|remote]` into an error in the inferred `s`. This script instead
measures `s`, `s0`, and `m` directly from the replayed traces --
`self_service_stats` in `experiments/self_service.py` already computes the
self-service rate from `outcome["rows"]`/`outcome["latencies"]` (the
self-RTT=0 boundary condition, not an inference), so it is reused rather
than reimplemented; migrations per entity per phase is a plain count off
`outcome["migrations"]`.

NO NEW DATASETS. This reads the exact traces `boundary_sensitivity.py`
already generated under `datasets/boundary/rate_{0.5,2,8}/ratio_*/seed_*`,
and picks "the best interaction-following configuration" and "the better
baseline" the same way that script's `rate_sweep` does (min mean combined
cost at alpha=1 over the same candidate sets: `follow_w10_d{0,0.25,1,2}` for
the follower, `{static, nearest}` for the baseline) -- recomputed here from
scratch via the identical argmin, then cross-checked against
`datasets/boundary/rate_comparisons.csv` so a silent drift between the two
selections would fail loudly instead of quietly misaligning the rows. The
seed-paired mean margin and its bootstrap CI are NOT recomputed: they are
read straight out of `rate_comparisons.csv`, which already computed them
with `experiments/load_significance.py::paired_bootstrap` on these same
traces.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.boundary_sensitivity import (
    ALPHA,
    BASELINES,
    DWELLS,
    PHASE_DURATION,
    RATES,
    RATIOS,
    SEEDS,
    follower,
    rate_dataset_root,
)
from experiments.combined_objective import total_migration_rtt_ms
from experiments.common import EVALUATION
from experiments.self_service import self_service_stats, verify_rtt_diagonal
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "boundary"
COMPARISONS_CSV = OUTPUT_ROOT / "rate_comparisons.csv"
COLLAPSE_CSV = OUTPUT_ROOT / "collapse.csv"

FOLLOWER_IDS = [f"follow_w10_d{d}" for d in DWELLS]
BASELINE_IDS = [spec["id"] for spec in BASELINES]
SPECS = BASELINES + [follower(10, d) for d in DWELLS]

COLLAPSE_COLUMNS = [
    "rate_per_entity", "dominant_peer_ratio",
    "rho", "rho_eff",
    "s", "s0", "s0_static", "s0_nearest", "m",
    "best_follower", "best_baseline",
    "mean_diff_combined_s", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]


def load_comparisons() -> dict[tuple[float, float], dict[str, str]]:
    with COMPARISONS_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {(float(row["rate_per_entity"]), float(row["dominant_peer_ratio"])): row for row in rows}


def measure(rate: float, ratio: float) -> dict[str, Any]:
    """Replays every seed's trace under every candidate spec and reduces to
    the per-(rate, ratio) quantities the collapse axis needs: mean combined
    cost per spec (to pick the best follower/baseline, exactly as
    `rate_sweep` does), mean self-service rate per spec (`s`, `s0`), and mean
    migration count for the chosen follower (`m`, before dividing by
    entity_count * n_phases -- done by the caller once dataset geometry is
    known).
    """
    combined: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}
    self_service_rate: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}
    migrations: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}
    total_requests: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}
    mean_rtt_remote: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}
    total_migration_rtt: dict[str, list[float]] = {spec["id"]: [] for spec in SPECS}

    entity_count = None
    n_phases = None
    for seed in SEEDS:
        root = rate_dataset_root(rate, ratio, seed)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"{root} has no manifest.json - this script only replays traces "
                f"experiments/boundary_sensitivity.py already generated; run that first")
        dataset = Dataset(root)
        verify_rtt_diagonal(dataset, f"rate={rate} ratio={ratio} seed={seed}")
        if entity_count is None:
            entity_count = len(dataset.entities)
            n_phases = dataset.duration / PHASE_DURATION

        for spec in SPECS:
            policy = build_policy(dataset, spec)
            outcome = replay(dataset, policy, EVALUATION)
            served = sum(outcome["latencies"])
            mig_rtt = total_migration_rtt_ms(outcome["rows"])
            combined[spec["id"]].append((served + ALPHA * mig_rtt) / 1000.0)
            migrations[spec["id"]].append(outcome["migrations"])
            total_migration_rtt[spec["id"]].append(mig_rtt)

            stats = self_service_stats(dataset, outcome)
            self_service_rate[spec["id"]].append(stats["self_service_rate"])
            total_requests[spec["id"]].append(stats["total_requests"])
            mean_rtt_remote[spec["id"]].append(stats["mean_rtt_remote_ms"])

    return {
        "combined": combined,
        "self_service_rate": self_service_rate,
        "migrations": migrations,
        "total_requests": total_requests,
        "mean_rtt_remote": mean_rtt_remote,
        "total_migration_rtt": total_migration_rtt,
        "entity_count": entity_count,
        "n_phases": n_phases,
    }


def build_rows() -> tuple[list[dict[str, Any]], dict[tuple[float, float], dict[str, Any]]]:
    comparisons = load_comparisons()
    rows: list[dict[str, Any]] = []
    detail: dict[tuple[float, float], dict[str, Any]] = {}

    for rate in RATES:
        for ratio in RATIOS:
            m = measure(rate, ratio)
            combined = m["combined"]

            best_follower = min(FOLLOWER_IDS, key=lambda pid: statistics.fmean(combined[pid]))
            best_baseline = min(BASELINE_IDS, key=lambda pid: statistics.fmean(combined[pid]))

            comparison_row = comparisons[(rate, ratio)]
            if comparison_row["best_follower"] != best_follower or comparison_row["best_baseline"] != best_baseline:
                raise AssertionError(
                    f"selection mismatch at rate={rate} ratio={ratio}: recomputed "
                    f"best_follower={best_follower!r} best_baseline={best_baseline!r} vs. "
                    f"rate_comparisons.csv best_follower={comparison_row['best_follower']!r} "
                    f"best_baseline={comparison_row['best_baseline']!r}")

            s = statistics.fmean(m["self_service_rate"][best_follower])
            s0 = statistics.fmean(m["self_service_rate"][best_baseline])
            s0_static = statistics.fmean(m["self_service_rate"]["static"])
            s0_nearest = statistics.fmean(m["self_service_rate"]["nearest"])
            migrations_per_run = statistics.fmean(m["migrations"][best_follower])
            m_per_entity_phase = migrations_per_run / (m["entity_count"] * m["n_phases"])

            rho = ratio * rate * PHASE_DURATION / ALPHA
            rho_eff = ((s - s0) * rate * PHASE_DURATION / (ALPHA * m_per_entity_phase)
                       if m_per_entity_phase > 0 else float("inf"))

            row = {
                "rate_per_entity": rate,
                "dominant_peer_ratio": ratio,
                "rho": f"{rho:.6f}",
                "rho_eff": f"{rho_eff:.6f}",
                "s": f"{s:.6f}",
                "s0": f"{s0:.6f}",
                "s0_static": f"{s0_static:.6f}",
                "s0_nearest": f"{s0_nearest:.6f}",
                "m": f"{m_per_entity_phase:.6f}",
                "best_follower": best_follower,
                "best_baseline": best_baseline,
                "mean_diff_combined_s": comparison_row["mean_diff_combined_s"],
                "bootstrap_ci95_low": comparison_row["bootstrap_ci95_low"],
                "bootstrap_ci95_high": comparison_row["bootstrap_ci95_high"],
                "verdict": comparison_row["verdict"],
            }
            rows.append(row)
            detail[(rate, ratio)] = m
            print(f"[ok]    rate={rate:<4} ratio={ratio:<5} best_follower={best_follower:<16} "
                  f"best_baseline={best_baseline:<8} s={s:.4f} s0={s0:.4f} m={m_per_entity_phase:.4f} "
                  f"rho={rho:.3f} rho_eff={rho_eff:.3f} verdict={row['verdict']}")
    return rows, detail


def crossing(rows: list[dict[str, Any]], key: str) -> None:
    """Sorts by `key` and reports whether the verdict is monotone in it, plus
    a linear interpolation (in log10(key)) of where the seed-paired margin
    crosses zero between the two bracketing samples. Mirrors the crossover
    interpolation in `experiments/combined_objective.py`'s section 3, just
    with the swept axis being `key` (rho or rho_eff) instead of the
    dominant peer ratio directly.
    """
    import math

    ordered = sorted(rows, key=lambda r: float(r[key]))
    # Rank 0 = follower_wins, 2 = follower_loses: a larger key should make
    # winning MORE likely, so the rank sequence should be non-increasing as
    # key increases (i.e. non-decreasing as key decreases, which is how
    # `ordered` is sorted here).
    verdict_rank = {"follower_wins": 0, "not_significant": 1, "follower_loses": 2}
    ranks = [verdict_rank[r["verdict"]] for r in ordered]
    monotone = all(a >= b for a, b in zip(ranks, ranks[1:]))
    print(f"\n  Sorted by {key} (ascending):")
    for r in ordered:
        print(f"    rate={r['rate_per_entity']:<4} ratio={r['dominant_peer_ratio']:<5} "
              f"{key}={float(r[key]):>10.4f}  margin={float(r['mean_diff_combined_s']):>+12.3f}s  "
              f"CI[{float(r['bootstrap_ci95_low']):>+10.3f}, {float(r['bootstrap_ci95_high']):>+10.3f}]  "
              f"{r['verdict']}")
    print(f"  => verdict is {'MONOTONE' if monotone else 'NOT monotone'} in {key} "
          f"(follower_loses -> not_significant -> follower_wins as {key} increases)")

    diffs = [float(r["mean_diff_combined_s"]) for r in ordered]
    keys = [float(r[key]) for r in ordered]
    crossed = False
    for (k1, d1), (k2, d2) in zip(zip(keys, diffs), zip(keys[1:], diffs[1:])):
        if d1 == 0 or (d1 < 0) != (d2 < 0):
            if d1 == 0:
                x_cross = k1
            else:
                log1, log2 = math.log10(k1), math.log10(k2)
                frac = -d1 / (d2 - d1) if d2 != d1 else 0.0
                x_cross = 10 ** (log1 + frac * (log2 - log1))
            print(f"  Crossing (margin = 0, log-linear interpolation between {key}={k1:.4f} "
                  f"and {key}={k2:.4f}): {key} ~= {x_cross:.4f}")
            crossed = True
    if not crossed:
        print(f"  No sign change of the margin across the sampled {key} grid.")


def sanity_check(detail: dict[tuple[float, float], dict[str, Any]], rows: list[dict[str, Any]],
                  rate: float, ratio: float) -> None:
    """Task 4: does (s - s0) * N_requests * mean_RTT - alpha * M * mean_migration_RTT
    (the combined-cost identity the rho_eff axis is built from) actually predict the
    measured combined-cost margin, or is rho_eff only capturing an ordinal trend?

    N_requests, mean_RTT (E[RTT|remote], follower), M and mean_migration_RTT (follower)
    are all seed-means of directly measured quantities -- no flat-RTT assumption, no
    per-policy invariance of E[RTT|remote] asserted, just plugged in as measured. The
    baseline's own migration cost (nearest is not migration-free) is reported separately
    so a shortfall in the two-term formula can be attributed to it rather than left a
    mystery.
    """
    row = next(r for r in rows if r["rate_per_entity"] == rate and r["dominant_peer_ratio"] == ratio)
    m = detail[(rate, ratio)]
    best_follower = row["best_follower"]
    best_baseline = row["best_baseline"]

    s = float(row["s"])
    s0 = float(row["s0"])
    n_requests = statistics.fmean(m["total_requests"][best_follower])
    mean_rtt = statistics.fmean(m["mean_rtt_remote"][best_follower])
    migrations_follower = statistics.fmean(m["migrations"][best_follower])
    total_mig_rtt_follower = statistics.fmean(m["total_migration_rtt"][best_follower])
    mean_migration_rtt_follower = (total_mig_rtt_follower / migrations_follower
                                    if migrations_follower > 0 else 0.0)
    migrations_baseline = statistics.fmean(m["migrations"][best_baseline])
    total_mig_rtt_baseline = statistics.fmean(m["total_migration_rtt"][best_baseline])

    predicted_two_term = (s - s0) * n_requests * mean_rtt - ALPHA * migrations_follower * mean_migration_rtt_follower
    predicted_with_baseline_migration = predicted_two_term + ALPHA * total_mig_rtt_baseline

    measured_margin_ms = -float(row["mean_diff_combined_s"]) * 1000.0  # baseline - follower, ms

    print("\n" + "=" * 78)
    print(f"SANITY CHECK: rate={rate}, ratio={ratio} "
          f"(best_follower={best_follower}, best_baseline={best_baseline})")
    print("=" * 78)
    print(f"  s={s:.6f}  s0={s0:.6f}  N_requests={n_requests:.1f}  mean_RTT|remote(follower)={mean_rtt:.4f} ms")
    print(f"  M(follower)={migrations_follower:.1f}  mean_migration_RTT(follower)={mean_migration_rtt_follower:.4f} ms")
    print(f"  M(baseline={best_baseline})={migrations_baseline:.1f}  "
          f"total_migration_RTT(baseline)={total_mig_rtt_baseline:.4f} ms")
    print(f"\n  measured margin (baseline - follower, i.e. follower's saving): {measured_margin_ms:+.3f} ms")
    print(f"  predicted, two-term formula (s-s0)*N*mean_RTT - alpha*M*mean_migration_RTT: "
          f"{predicted_two_term:+.3f} ms")
    if measured_margin_ms != 0:
        rel_err_two_term = abs(predicted_two_term - measured_margin_ms) / abs(measured_margin_ms)
        print(f"    relative error: {rel_err_two_term:.4%}")
    print(f"  predicted, with baseline's own migration cost added back: "
          f"{predicted_with_baseline_migration:+.3f} ms")
    if measured_margin_ms != 0:
        rel_err_full = abs(predicted_with_baseline_migration - measured_margin_ms) / abs(measured_margin_ms)
        print(f"    relative error: {rel_err_full:.4%}")


def main() -> None:
    rows, detail = build_rows()
    write_csv(COLLAPSE_CSV, COLLAPSE_COLUMNS, rows)
    print(f"\n[write] {COLLAPSE_CSV.relative_to(ROOT)} ({len(rows)} rows)")

    print("\n" + "=" * 78)
    print("COARSE AXIS: rho = r * lambda * w / alpha")
    print("=" * 78)
    crossing(rows, "rho")

    print("\n" + "=" * 78)
    print("MEASURED AXIS: rho_eff = (s - s0) * lambda * w / (alpha * m)")
    print("=" * 78)
    crossing(rows, "rho_eff")

    # The point closest to where the preliminary latency-inversion estimate put the
    # crossing (rate=2, ratio=0.30, verdict=not_significant) -- see module docstring.
    sanity_check(detail, rows, rate=2.0, ratio=0.30)


if __name__ == "__main__":
    main()
