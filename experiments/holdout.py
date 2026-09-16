"""Held-out confirmation of the configurations the paper selects.

`weighted_dwell.py` and `boundary_sensitivity.py` both pick a cheapest
(window, theta_D) per ratio on seeds 1-30 and then report a paired bootstrap
CI for that same pick on those same 30 seeds. Those intervals are conditional
on a selection made from the data they are computed on, so they are
exploratory: they do not cover the selection's own uncertainty, and a
difference that is small relative to seed noise can be an artifact of taking a
minimum over 35 (or 4) candidates.

This script separates the two roles. It takes the selection as given -
recomputed here from the committed aggregates over seeds 1-30, and asserted to
match what those aggregates already record - and re-evaluates only the
SELECTED configurations on 30 seeds that took no part in choosing them. The
selection seeds stay exploratory; the numbers this script emits are
confirmatory.

Both parts generate their own traces on first run (`ensure_dataset` /
`ensure_rate_dataset` skip seeds already on disk), so nothing under
datasets/ambiguity or datasets/boundary that the paper's existing numbers rest
on is touched.

    python -m experiments.holdout

Outputs
  datasets/holdout/ratio_confirmation.csv - Table I and Sec. V-D margins
  datasets/holdout/rate_confirmation.csv  - Sec. V-D rate sweep, rho_eff
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import RATIOS as RATIO_SWEEP_RATIOS, ensure_dataset
from experiments.boundary_collapse import BASELINE_IDS as RATE_BASELINE_IDS, FOLLOWER_IDS, SPECS as RATE_SPECS
from experiments.boundary_sensitivity import (
    ALPHA,
    PHASE_DURATION,
    RATES,
    RATIOS as RATE_SWEEP_RATIOS,
    ensure_rate_dataset,
)
from experiments.combined_objective import total_migration_rtt_ms
from experiments.common import EVALUATION
from experiments.load_significance import paired_bootstrap
from experiments.self_service import self_service_stats, verify_rtt_diagonal
from experiments.weighted_dwell import ALL_SPECS as RATIO_SPECS, combined_s, verdict
from nve_dataset.config import load_config
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "holdout"

# Seeds 1-30 chose the configurations; seeds 31-60 evaluate them. The split is
# by seed rather than by trace segment because a seed indexes an independent
# generated request sequence (see ambiguity_sweep.ensure_dataset).
SELECTION_SEEDS = list(range(1, 31))
HOLDOUT_SEEDS = list(range(31, 61))

WEIGHTED_DWELL_AGGREGATE = ROOT / "datasets" / "weighted_dwell" / "aggregate.csv"
COLLAPSE_CSV = ROOT / "datasets" / "boundary" / "collapse.csv"

RATIO_COLUMNS = [
    "dominant_peer_ratio", "comparison", "policy_a", "policy_b",
    "selection_n_seeds", "selection_mean_diff_combined_s",
    "holdout_n_seeds", "holdout_mean_diff_combined_s",
    "holdout_ci95_low", "holdout_ci95_high", "holdout_verdict",
    "holdout_mean_combined_a", "holdout_mean_combined_b",
]
RATE_COLUMNS = [
    "rate_per_entity", "dominant_peer_ratio", "best_follower", "best_baseline",
    "selection_mean_diff_combined_s", "selection_rho_eff",
    "holdout_n_seeds", "holdout_mean_diff_combined_s",
    "holdout_ci95_low", "holdout_ci95_high", "holdout_verdict",
    "holdout_s", "holdout_s0", "holdout_m", "holdout_rho_eff",
]

RATIO_SPEC_BY_ID = {spec["id"]: spec for spec in RATIO_SPECS}
RATE_SPEC_BY_ID = {spec["id"]: spec for spec in RATE_SPECS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# ----------------------------------------------------------------------
# Part A: the ratio sweep behind Table I and Sec. V-D
# ----------------------------------------------------------------------
def ratio_selection() -> dict[float, dict[str, str]]:
    """Recomputes, per ratio, the three policy ids the paper selects on seeds
    1-30: the cheaper baseline, the cheapest demand_argmax cell and the
    cheapest rtt_weighted cell."""
    rows = read_rows(WEIGHTED_DWELL_AGGREGATE)
    mean_by: dict[tuple[float, str], float] = {}
    arm_members: dict[tuple[float, str], list[str]] = {}
    for row in rows:
        ratio = float(row["dominant_peer_ratio"])
        pid = row["policy_id"]
        mean_by[(ratio, pid)] = float(row["combined_s_alpha1_mean"])
        arm = row["selector"] or "baseline"
        arm_members.setdefault((ratio, arm), []).append(pid)
        if int(row["n_seeds"]) != len(SELECTION_SEEDS):
            raise AssertionError(
                f"{WEIGHTED_DWELL_AGGREGATE} row for ratio={ratio} {pid} has "
                f"n_seeds={row['n_seeds']}, expected {len(SELECTION_SEEDS)}")

    selection: dict[float, dict[str, str]] = {}
    for ratio in RATIO_SWEEP_RATIOS:
        pick = {}
        for arm, label in (("baseline", "baseline"),
                           ("demand_argmax", "argmax"),
                           ("rtt_weighted", "weighted")):
            members = arm_members[(ratio, arm)]
            pick[label] = min(members, key=lambda pid: mean_by[(ratio, pid)])
        pick["selection_means"] = {v: mean_by[(ratio, v)]
                                   for k, v in pick.items() if k != "selection_means"}
        selection[ratio] = pick
    return selection


def replay_ratio_holdout(base_config: dict[str, Any], ratio: float,
                         policy_ids: list[str]) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {pid: [] for pid in policy_ids}
    for seed in HOLDOUT_SEEDS:
        root = ensure_dataset(base_config, ratio, seed)
        dataset = Dataset(root)
        for pid in policy_ids:
            outcome = replay(dataset, build_policy(dataset, RATIO_SPEC_BY_ID[pid]), EVALUATION)
            combined[pid].append(combined_s(sum(outcome["latencies"]),
                                            total_migration_rtt_ms(outcome["rows"]), ALPHA))
    return combined


def ratio_part() -> list[dict[str, Any]]:
    base_config = load_config(ROOT / "config.yaml")
    selection = ratio_selection()
    out_rows: list[dict[str, Any]] = []

    for ratio in RATIO_SWEEP_RATIOS:
        pick = selection[ratio]
        ids = [pick["baseline"], pick["argmax"], pick["weighted"]]
        combined = replay_ratio_holdout(base_config, ratio, ids)
        sel_mean = pick["selection_means"]

        for label, a, b in (("argmax_vs_best_baseline", pick["argmax"], pick["baseline"]),
                            ("weighted_vs_best_baseline", pick["weighted"], pick["baseline"]),
                            ("weighted_vs_argmax", pick["weighted"], pick["argmax"])):
            mean_diff, _std, lo, hi = paired_bootstrap(combined[a], combined[b])
            out_rows.append({
                "dominant_peer_ratio": ratio,
                "comparison": label,
                "policy_a": a,
                "policy_b": b,
                "selection_n_seeds": len(SELECTION_SEEDS),
                "selection_mean_diff_combined_s": f"{sel_mean[a] - sel_mean[b]:.6f}",
                "holdout_n_seeds": len(HOLDOUT_SEEDS),
                "holdout_mean_diff_combined_s": f"{mean_diff:.6f}",
                "holdout_ci95_low": f"{lo:.6f}",
                "holdout_ci95_high": f"{hi:.6f}",
                "holdout_verdict": verdict(lo, hi),
                "holdout_mean_combined_a": f"{statistics.fmean(combined[a]):.6f}",
                "holdout_mean_combined_b": f"{statistics.fmean(combined[b]):.6f}",
            })
        print(f"[ok]    ratio={ratio:<5} baseline={pick['baseline']:<8} "
              f"argmax={pick['argmax']:<22} weighted={pick['weighted']}")
    return out_rows


# ----------------------------------------------------------------------
# Part B: the rate sweep behind the rho_eff collapse
# ----------------------------------------------------------------------
def rate_part() -> list[dict[str, Any]]:
    base_config = load_config(ROOT / "config.yaml")
    selection = {(float(r["rate_per_entity"]), float(r["dominant_peer_ratio"])): r
                 for r in read_rows(COLLAPSE_CSV)}
    out_rows: list[dict[str, Any]] = []

    for rate in RATES:
        for ratio in RATE_SWEEP_RATIOS:
            sel = selection[(rate, ratio)]
            follower_id = sel["best_follower"]
            baseline_id = sel["best_baseline"]
            if follower_id not in FOLLOWER_IDS or baseline_id not in RATE_BASELINE_IDS:
                raise AssertionError(f"unexpected selection at rate={rate} ratio={ratio}: {sel}")

            ids = [follower_id, baseline_id]
            combined: dict[str, list[float]] = {pid: [] for pid in ids}
            self_service: dict[str, list[float]] = {pid: [] for pid in ids}
            migrations: list[float] = []
            entity_count = n_phases = None

            for seed in HOLDOUT_SEEDS:
                root = ensure_rate_dataset(base_config, rate, ratio, seed)
                dataset = Dataset(root)
                verify_rtt_diagonal(dataset, f"holdout rate={rate} ratio={ratio} seed={seed}")
                if entity_count is None:
                    entity_count = len(dataset.entities)
                    n_phases = dataset.duration / PHASE_DURATION
                for pid in ids:
                    outcome = replay(dataset, build_policy(dataset, RATE_SPEC_BY_ID[pid]), EVALUATION)
                    combined[pid].append(combined_s(sum(outcome["latencies"]),
                                                    total_migration_rtt_ms(outcome["rows"]), ALPHA))
                    self_service[pid].append(self_service_stats(dataset, outcome)["self_service_rate"])
                    if pid == follower_id:
                        migrations.append(float(outcome["migrations"]))

            mean_diff, _std, lo, hi = paired_bootstrap(combined[follower_id], combined[baseline_id])
            s = statistics.fmean(self_service[follower_id])
            s0 = statistics.fmean(self_service[baseline_id])
            m_per_entity_phase = statistics.fmean(migrations) / (entity_count * n_phases)
            rho_eff = ((s - s0) * rate * PHASE_DURATION / (ALPHA * m_per_entity_phase)
                       if m_per_entity_phase > 0 else float("inf"))

            out_rows.append({
                "rate_per_entity": rate,
                "dominant_peer_ratio": ratio,
                "best_follower": follower_id,
                "best_baseline": baseline_id,
                "selection_mean_diff_combined_s": sel["mean_diff_combined_s"],
                "selection_rho_eff": sel["rho_eff"],
                "holdout_n_seeds": len(HOLDOUT_SEEDS),
                "holdout_mean_diff_combined_s": f"{mean_diff:.6f}",
                "holdout_ci95_low": f"{lo:.6f}",
                "holdout_ci95_high": f"{hi:.6f}",
                "holdout_verdict": verdict(lo, hi),
                "holdout_s": f"{s:.6f}",
                "holdout_s0": f"{s0:.6f}",
                "holdout_m": f"{m_per_entity_phase:.6f}",
                "holdout_rho_eff": f"{rho_eff:.6f}",
            })
            print(f"[ok]    rate={rate:<4} ratio={ratio:<5} follower={follower_id:<16} "
                  f"margin={mean_diff:+9.3f}s CI[{lo:+.3f},{hi:+.3f}] "
                  f"rho_eff={rho_eff:.3f} ({verdict(lo, hi)})")
    return out_rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[run] selection seeds {SELECTION_SEEDS[0]}-{SELECTION_SEEDS[-1]}, "
          f"holdout seeds {HOLDOUT_SEEDS[0]}-{HOLDOUT_SEEDS[-1]}")

    print("\n[part A] ratio sweep (Table I, Sec. V-D)")
    ratio_rows = ratio_part()
    write_csv(OUTPUT_ROOT / "ratio_confirmation.csv", RATIO_COLUMNS, ratio_rows)

    print("\n[part B] rate sweep (Sec. V-D, Fig. 4)")
    rate_rows = rate_part()
    write_csv(OUTPUT_ROOT / "rate_confirmation.csv", RATE_COLUMNS, rate_rows)

    print(f"\n[done] wrote {OUTPUT_ROOT.relative_to(ROOT)}/ratio_confirmation.csv "
          f"and rate_confirmation.csv")


if __name__ == "__main__":
    main()
