"""Compare online policies and the offline oracle under one combined cost.

Cost = sum(interaction_latency_ms) + alpha * sum(migration_network_latency_ms).
Every policy serves each request before deciding whether to migrate. Actual
per-hop RTT is used, not a constant migration-RTT approximation. The oracle
objective is checked against the replay engine by
_verify_oracle_matches_engine_objective.

Replays datasets/ambiguity/ratio_*/seed_* and datasets/switching/phase_*/seed_*;
run ambiguity_sweep and switching_sweep first. No datasets are generated here.
Outputs include per-seed costs, aggregates, and paired policy comparisons.
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path
from typing import Any

from experiments.ambiguity_sweep import (
    dataset_root as ambiguity_dataset_root,
    RATIOS,
)
from experiments.ambiguity_sweep import SEEDS as AMBIGUITY_SEEDS
from experiments.common import build_policy_specs, check_equivalence, EVALUATION
from experiments.load_significance import paired_bootstrap
from experiments.switching_sweep import (
    dataset_root as switching_dataset_root,
    PHASE_DURATIONS,
)
from experiments.switching_sweep import SEEDS as SWITCHING_SEEDS
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "combined_objective"

ALPHAS = [0.5, 1.0, 2.0]

# The operating point this script was written to price: theta_G ("engagement
# threshold") = 0, theta_D ("dwell") = 1s, which is what the paper proposed on
# the two-axis reading before this sweep priced the axes against each other. In
# this codebase's ids that is engagement_g00_d1.0 (see experiments/common.py
# build_policy_specs: the g00 label is engagement_threshold=0.0). "naive"
# (theta_D=0) is engagement_g00_d0.0, exactly naive_target_aware (asserted equal
# below by check_equivalence, as in the other two sweeps). The paper's final
# selection, theta_D=0.25s, comes out of the sweep this script runs; it is not
# an input to it.
PROPOSED_ID = "engagement_g00_d1.0"
NAIVE_ID = "naive_target_aware"
BASELINE_IDS = ["static", "nearest"]

# "Interaction-following" policies for the crossover question: everything that
# chases recent demand, i.e. naive_target_aware plus the whole engagement_aware
# family (both engagement_threshold arms, all dwell values). Deliberately excludes
# the oracle (not an online policy) and the baselines themselves.
INTERACTION_FOLLOWING_PREFIXES = ("engagement_g00_", "engagement_g05_")


def alpha_oracle_id(alpha: float) -> str:
    return f"oracle_a{alpha:g}"


def build_specs_with_all_oracles() -> list[dict[str, Any]]:
    """build_policy_specs() only ships oracle_a1 (alpha=1.0), because that is
    the alpha the paper states (the evaluation settings). But an oracle solved for
    alpha=1 is only *guaranteed* optimal under the alpha=1 objective - it is
    not necessarily optimal under alpha=0.5 or alpha=2 objectives, since the
    DP's optimal plan itself depends on alpha. To honestly check "does the
    oracle beat every online policy under the combined objective" at all
    three alphas this script reports, we need an oracle actually solved at
    each of those alphas, not just relabelled. This adds oracle_a0.5 and
    oracle_a2 alongside the existing oracle_a1; nothing about OraclePolicy or
    any online policy is modified.
    """
    specs = build_policy_specs()
    have = {spec["id"] for spec in specs}
    for alpha in ALPHAS:
        oid = alpha_oracle_id(alpha)
        if oid not in have:
            specs.append({"id": oid, "type": "oracle", "migration_penalty_weight": alpha})
    return specs


def _verify_oracle_matches_engine_objective() -> None:
    """Read-only static check (not a replay) that oracle.py's DP charges
    exactly what engine.py's replay charges, so the two are commensurable
    before we build an experiment on that assumption. See the module
    docstring for the argument; this just makes it fail loudly if the source
    ever drifts instead of silently trusting the docstring.
    """
    oracle_src = (ROOT / "nve_policy" / "policies" / "oracle.py").read_text(encoding="utf-8")
    engine_src = (ROOT / "nve_policy" / "engine.py").read_text(encoding="utf-8")
    problems = []
    if "serve = rtt[[r.peer for r in requests], :] + self.dataset.processing_delay_ms" not in oracle_src:
        problems.append("oracle.py's serve term no longer looks like RTT(request.peer, a) + processing_delay_ms")
    if "transition = self.alpha * rtt" not in oracle_src:
        problems.append("oracle.py's transition term no longer looks like alpha * RTT[b, a]")
    if "latency = dataset.rtt[request.peer, before] + dataset.processing_delay_ms" not in engine_src:
        problems.append("engine.py's interaction_latency_ms no longer looks like RTT(request.peer, before) + processing_delay_ms")
    if "network_latency = dataset.rtt[before, after]" not in engine_src:
        problems.append("engine.py's migration_network_latency_ms no longer looks like RTT(before, after)")
    if problems:
        print("OBJECTIVE MISMATCH: the Oracle's DP and the engine's replay no longer "
              "charge the same quantities. The Oracle and the online policies would "
              "NOT be commensurable under a combined objective. Details:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise AssertionError("Oracle objective and engine replay charges have diverged; see stderr")
    print("[check] oracle.py's DP and engine.py's replay charge the identical serve/transition "
          "quantities (interaction_latency_ms and migration_network_latency_ms) -- "
          "Oracle and online policies ARE commensurable under sum(interaction_latency_ms) "
          "+ alpha * sum(migration_network_latency_ms).")


def total_migration_rtt_ms(rows: list[dict[str, Any]]) -> float:
    """Sum of migration_network_latency_ms over rows where a migration
    occurred. Not provided by engine.replay's outcome dict (which stops at a
    migration *count*, see nve_policy/metrics.py), so computed here directly
    from the same in-memory rows replay() already produced - no re-derivation
    of RTTs, just summing a column replay already computed per-row.
    """
    total = 0.0
    for row in rows:
        if row["migration_occurred"] == 1:
            total += float(row["migration_network_latency_ms"])
    return total


def combined_ms(total_served_ms: float, total_mig_rtt_ms: float, alpha: float) -> float:
    return total_served_ms + alpha * total_mig_rtt_ms


SUMMARY_COLUMNS = [
    "sweep", "knob_name", "knob_value", "seed", "policy_id",
    "total_requests", "migration_count",
    "total_served_ms", "total_migration_rtt_ms", "mean_migration_rtt_ms",
    "migration_bytes_total", "mean_migration_bytes_per_migration",
    "combined_ms_alpha0.5", "combined_ms_alpha1", "combined_ms_alpha2",
]
AGGREGATE_COLUMNS = [
    "sweep", "knob_name", "knob_value", "policy_id", "n_seeds",
    "total_served_ms_mean", "total_migration_rtt_ms_mean", "mean_migration_rtt_ms_mean",
    "migration_count_mean", "migration_bytes_total_mean", "mean_migration_bytes_per_migration_mean",
    "combined_ms_alpha0.5_mean", "combined_ms_alpha0.5_std",
    "combined_ms_alpha1_mean", "combined_ms_alpha1_std",
    "combined_ms_alpha2_mean", "combined_ms_alpha2_std",
]
DOMINANCE_COLUMNS = [
    "question", "knob_value", "alpha", "policy_a", "policy_b", "n_seeds",
    "mean_diff_combined_ms", "bootstrap_ci95_low", "bootstrap_ci95_high", "verdict",
]


def run_sweep(sweep_name: str, knob_name: str, knob_values: list[float], seeds: list[int],
              dataset_root_fn, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for knob_value in knob_values:
        for seed in seeds:
            root = dataset_root_fn(knob_value, seed)
            if not (root / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"{root} has no manifest.json - expected an already-generated dataset "
                    f"(this script does not generate datasets; run {sweep_name}_sweep.py first)")
            dataset = Dataset(root)
            outcomes: dict[str, dict[str, Any]] = {}
            for spec in specs:
                policy = build_policy(dataset, spec)
                outcome = replay(dataset, policy, EVALUATION)
                outcomes[spec["id"]] = outcome
                total_served = sum(outcome["latencies"])
                total_mig_rtt = total_migration_rtt_ms(outcome["rows"])
                migrations = outcome["migrations"]
                mig_bytes = outcome["migration_bytes_total"]
                row = {
                    "sweep": sweep_name,
                    "knob_name": knob_name,
                    "knob_value": knob_value,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": len(outcome["latencies"]),
                    "migration_count": migrations,
                    "total_served_ms": f"{total_served:.6f}",
                    "total_migration_rtt_ms": f"{total_mig_rtt:.6f}",
                    "mean_migration_rtt_ms": f"{(total_mig_rtt / migrations):.6f}" if migrations else "",
                    "migration_bytes_total": mig_bytes,
                    "mean_migration_bytes_per_migration": f"{(mig_bytes / migrations):.6f}" if migrations else "",
                }
                for alpha in ALPHAS:
                    row[f"combined_ms_alpha{alpha:g}"] = f"{combined_ms(total_served, total_mig_rtt, alpha):.6f}"
                rows.append(row)
            # Same equivalence guarantee ambiguity_sweep.py / switching_sweep.py assert on
            # every dataset: theta_G=0, theta_D=0, theta_R=inf must be byte-identical to
            # naive_target_aware. Re-checked here because this script re-replays from
            # scratch rather than reading their CSVs.
            check_equivalence(outcomes, f"{sweep_name} {knob_name}={knob_value} seed={seed}")
    return rows


def aggregate(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row["sweep"], row["knob_name"], row["knob_value"], row["policy_id"])
        groups.setdefault(key, []).append(row)

    out = []
    for (sweep_name, knob_name, knob_value, policy_id), rows in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        total_served = [float(r["total_served_ms"]) for r in rows]
        total_mig_rtt = [float(r["total_migration_rtt_ms"]) for r in rows]
        mig_counts = [float(r["migration_count"]) for r in rows]
        mig_bytes = [float(r["migration_bytes_total"]) for r in rows]
        mean_mig_rtt = [float(r["mean_migration_rtt_ms"]) for r in rows if r["mean_migration_rtt_ms"] != ""]
        mean_mig_bytes = [float(r["mean_migration_bytes_per_migration"]) for r in rows
                           if r["mean_migration_bytes_per_migration"] != ""]
        entry = {
            "sweep": sweep_name, "knob_name": knob_name, "knob_value": knob_value,
            "policy_id": policy_id, "n_seeds": len(rows),
            "total_served_ms_mean": f"{statistics.fmean(total_served):.6f}",
            "total_migration_rtt_ms_mean": f"{statistics.fmean(total_mig_rtt):.6f}",
            "mean_migration_rtt_ms_mean": f"{statistics.fmean(mean_mig_rtt):.6f}" if mean_mig_rtt else "",
            "migration_count_mean": f"{statistics.fmean(mig_counts):.6f}",
            "migration_bytes_total_mean": f"{statistics.fmean(mig_bytes):.6f}",
            "mean_migration_bytes_per_migration_mean": f"{statistics.fmean(mean_mig_bytes):.6f}" if mean_mig_bytes else "",
        }
        for alpha in ALPHAS:
            key = f"combined_ms_alpha{alpha:g}"
            values = [float(r[key]) for r in rows]
            entry[f"{key}_mean"] = f"{statistics.fmean(values):.6f}"
            entry[f"{key}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000"
        out.append(entry)
    return out


def combined_by_seed(summary_rows: list[dict[str, Any]], sweep_name: str, knob_value: float,
                      policy_id: str, alpha: float) -> list[float]:
    key = f"combined_ms_alpha{alpha:g}"
    matches = [r for r in summary_rows if r["sweep"] == sweep_name and r["knob_value"] == knob_value
               and r["policy_id"] == policy_id]
    matches.sort(key=lambda r: r["seed"])
    return [float(r[key]) for r in matches]


def print_ranking(agg_rows: list[dict[str, Any]], sweep_name: str, knob_value: float, alpha: float) -> None:
    key = f"combined_ms_alpha{alpha:g}_mean"
    rows = [r for r in agg_rows if r["sweep"] == sweep_name and r["knob_value"] == knob_value]
    rows.sort(key=lambda r: float(r[key]))
    print(f"\n  Ranking at {sweep_name} {rows[0]['knob_name']}={knob_value:g}, alpha={alpha:g} "
          f"(combined objective, lower=better):")
    for rank, row in enumerate(rows, start=1):
        print(f"    {rank:>2}. {row['policy_id']:<16} combined={float(row[key]):>10.1f} ms  "
              f"(served={float(row['total_served_ms_mean']):>9.1f}  "
              f"migration_cost={alpha * float(row['total_migration_rtt_ms_mean']):>9.1f}  "
              f"migrations={float(row['migration_count_mean']):>7.1f})")


def dominance_row(question: str, knob_value: float, alpha: float, policy_a: str, policy_b: str,
                   a_values: list[float], b_values: list[float]) -> dict[str, Any]:
    mean_diff, _std, lo, hi = paired_bootstrap(a_values, b_values)
    if lo > 0:
        verdict = f"{policy_a} SIGNIFICANTLY WORSE than {policy_b} (dominated)"
    elif hi < 0:
        verdict = f"{policy_a} SIGNIFICANTLY BETTER than {policy_b}"
    else:
        verdict = "not distinguishable from 0 (CI straddles zero)"
    return {
        "question": question, "knob_value": knob_value, "alpha": alpha,
        "policy_a": policy_a, "policy_b": policy_b, "n_seeds": len(a_values),
        "mean_diff_combined_ms": f"{mean_diff:.6f}",
        "bootstrap_ci95_low": f"{lo:.6f}", "bootstrap_ci95_high": f"{hi:.6f}",
        "verdict": verdict,
    }


def main() -> None:
    _verify_oracle_matches_engine_objective()

    specs = build_specs_with_all_oracles()
    print(f"\n[run] {len(specs)} policy specs x {len(RATIOS)} ratios x {len(AMBIGUITY_SEEDS)} seeds "
          f"(ambiguity sweep, reusing datasets/ambiguity/ratio_*)")
    ambiguity_rows = run_sweep("ambiguity", "dominant_peer_ratio", RATIOS, AMBIGUITY_SEEDS,
                                ambiguity_dataset_root, specs)
    print(f"[run] {len(specs)} policy specs x {len(PHASE_DURATIONS)} phase durations x "
          f"{len(SWITCHING_SEEDS)} seeds (switching sweep, reusing datasets/switching/phase_*)")
    switching_rows = run_sweep("switching", "phase_duration", PHASE_DURATIONS, SWITCHING_SEEDS,
                                switching_dataset_root, specs)
    summary_rows = ambiguity_rows + switching_rows

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    agg_rows = aggregate(summary_rows)
    aggregate_path = OUTPUT_ROOT / "aggregate.csv"
    write_csv(aggregate_path, AGGREGATE_COLUMNS, agg_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(agg_rows)} rows)")

    # ------------------------------------------------------------------
    # 1. Rankings at ratio 0.25 and 0.85, every alpha.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. COMBINED-OBJECTIVE RANKINGS (dominant_peer_ratio 0.25 and 0.85)")
    print("=" * 78)
    for ratio in (0.25, 0.85):
        for alpha in ALPHAS:
            print_ranking(agg_rows, "ambiguity", ratio, alpha)

    # ------------------------------------------------------------------
    # 2. Domination check: proposed (engagement_g00_d1.0) vs Static/Nearest at
    #    ratio 0.25, for each alpha.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. IS THE PROPOSED POLICY (engagement_g00_d1.0) DOMINATED AT ratio=0.25?")
    print("=" * 78)
    dominance_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        proposed_vals = combined_by_seed(summary_rows, "ambiguity", 0.25, PROPOSED_ID, alpha)
        for baseline in BASELINE_IDS:
            baseline_vals = combined_by_seed(summary_rows, "ambiguity", 0.25, baseline, alpha)
            drow = dominance_row("proposed_vs_baseline", 0.25, alpha, PROPOSED_ID, baseline,
                                  proposed_vals, baseline_vals)
            dominance_rows.append(drow)
            print(f"  alpha={alpha:g}: {PROPOSED_ID} - {baseline}: "
                  f"mean_diff={float(drow['mean_diff_combined_ms']):+9.2f} ms  "
                  f"95% CI [{float(drow['bootstrap_ci95_low']):+8.2f}, {float(drow['bootstrap_ci95_high']):+8.2f}]  "
                  f"-> {drow['verdict']}")
        naive_vals = combined_by_seed(summary_rows, "ambiguity", 0.25, NAIVE_ID, alpha)
        drow = dominance_row("proposed_vs_naive", 0.25, alpha, PROPOSED_ID, NAIVE_ID,
                              proposed_vals, naive_vals)
        dominance_rows.append(drow)
        print(f"  alpha={alpha:g}: {PROPOSED_ID} - {NAIVE_ID}: "
              f"mean_diff={float(drow['mean_diff_combined_ms']):+9.2f} ms  "
              f"95% CI [{float(drow['bootstrap_ci95_low']):+8.2f}, {float(drow['bootstrap_ci95_high']):+8.2f}]  "
              f"-> {drow['verdict']}")

    # ------------------------------------------------------------------
    # 3. Crossover ratio under alpha=1: where does the best interaction-
    #    following policy stop beating the best baseline?
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. CROSSOVER: best interaction-following vs best baseline, alpha=1")
    print("=" * 78)
    alpha = 1.0
    crossover_table = []
    for ratio in sorted(RATIOS):
        online_candidates = [r for r in agg_rows if r["sweep"] == "ambiguity" and r["knob_value"] == ratio
                              and (r["policy_id"] == NAIVE_ID
                                   or r["policy_id"].startswith(INTERACTION_FOLLOWING_PREFIXES))]
        baseline_candidates = [r for r in agg_rows if r["sweep"] == "ambiguity" and r["knob_value"] == ratio
                                and r["policy_id"] in BASELINE_IDS]
        key = f"combined_ms_alpha{alpha:g}_mean"
        best_online = min(online_candidates, key=lambda r: float(r[key]))
        best_baseline = min(baseline_candidates, key=lambda r: float(r[key]))
        diff = float(best_online[key]) - float(best_baseline[key])
        crossover_table.append((ratio, best_online["policy_id"], float(best_online[key]),
                                 best_baseline["policy_id"], float(best_baseline[key]), diff))
        print(f"  ratio={ratio:.2f}: best_online={best_online['policy_id']:<14} "
              f"({float(best_online[key]):>9.1f} ms)  best_baseline={best_baseline['policy_id']:<8} "
              f"({float(best_baseline[key]):>9.1f} ms)  online-baseline={diff:+9.1f} ms "
              f"{'[online wins]' if diff < 0 else '[baseline wins]'}")

    crossover_ratio = None
    for (r1, _, _, _, _, d1), (r2, _, _, _, _, d2) in zip(crossover_table, crossover_table[1:]):
        if d1 == 0 or d2 == 0 or (d1 < 0) != (d2 < 0):
            # linear interpolation in diff vs ratio between the two bracketing samples
            frac = -d1 / (d2 - d1) if d2 != d1 else 0.0
            crossover_ratio = r1 + frac * (r2 - r1)
            print(f"\n  Crossover (linear interpolation between sampled ratio={r1:g} and ratio={r2:g}): "
                  f"dominant_peer_ratio ~= {crossover_ratio:.4f}")
    if crossover_ratio is None:
        print("\n  No sign change across the sampled ratio grid "
              f"{sorted(set(r for r, *_ in crossover_table))} - "
              "see per-ratio diffs above for which side wins throughout.")

    # ------------------------------------------------------------------
    # 4. Oracle domination check.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. ORACLE DOMINATION CHECK (oracle_a{alpha} vs every online policy, per alpha)")
    print("=" * 78)
    online_ids = [s["id"] for s in specs if s["type"] != "oracle"]
    row_index = {(r["sweep"], r["knob_value"], r["seed"], r["policy_id"]): r for r in summary_rows}
    violations = []
    for sweep_name, knob_name, knob_values in (
            ("ambiguity", "dominant_peer_ratio", RATIOS), ("switching", "phase_duration", PHASE_DURATIONS)):
        seeds = AMBIGUITY_SEEDS if sweep_name == "ambiguity" else SWITCHING_SEEDS
        for knob_value in knob_values:
            for seed in seeds:
                for alpha in ALPHAS:
                    key = f"combined_ms_alpha{alpha:g}"
                    oracle_val = float(row_index[(sweep_name, knob_value, seed, alpha_oracle_id(alpha))][key])
                    for pid in online_ids:
                        online_val = float(row_index[(sweep_name, knob_value, seed, pid)][key])
                        if oracle_val > online_val + 1e-6:
                            violations.append((sweep_name, knob_name, knob_value, seed, alpha, pid,
                                                oracle_val, online_val))
    if violations:
        print(f"  {len(violations)} VIOLATIONS FOUND - the oracle does NOT dominate. Examples:")
        for v in violations[:10]:
            print(f"    {v}")
    else:
        n_checks = sum(len(seeds) * len(knob_values) * len(ALPHAS) * len(online_ids)
                        for (_, _, knob_values), seeds in
                        [((None, None, RATIOS), AMBIGUITY_SEEDS), ((None, None, PHASE_DURATIONS), SWITCHING_SEEDS)])
        print(f"  PASSED: oracle_a{{alpha}} combined objective <= every online policy's, "
              f"on every (sweep, knob, seed) at that alpha ({n_checks} comparisons).")

    # ------------------------------------------------------------------
    # 5. theta_D movement at ratio 0.25 and 0.85, alpha=1: does raising
    #    dwell move the engagement_g00 family toward or away from the best
    #    combined cost?
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. theta_D MOVEMENT (engagement_g00_d*, alpha=1): toward or away from best combined cost?")
    print("=" * 78)
    dwell_order = ["engagement_g00_d0.0", "engagement_g00_d0.25", "engagement_g00_d0.5", "engagement_g00_d1.0",
                   "engagement_g00_d2.0", "engagement_g00_d5.0", "engagement_g00_d10.0"]
    for ratio in (0.25, 0.85):
        key = "combined_ms_alpha1_mean"
        vals = []
        for pid in dwell_order:
            row = next(r for r in agg_rows if r["sweep"] == "ambiguity" and r["knob_value"] == ratio
                       and r["policy_id"] == pid)
            vals.append((pid, float(row[key])))
        base = vals[0][1]
        best = min(v for _, v in vals)
        best_id = next(pid for pid, v in vals if v == best)
        print(f"\n  ratio={ratio:.2f}: combined_ms(alpha=1) by dwell, theta_D=0 baseline={base:.1f} ms")
        for pid, v in vals:
            print(f"    {pid:<16} {v:>9.1f} ms  (delta from theta_D=0: {v - base:+8.1f})")
        direction = "AWAY from" if best_id == "engagement_g00_d0.0" else "TOWARD (then away from, past a minimum at)"
        print(f"    => best in this dwell family: {best_id} ({best:.1f} ms); "
              f"raising theta_D moves {direction} the best combined cost past that point.")

    # ------------------------------------------------------------------
    # 6. Migration cost in ms and bytes, per policy, ratio 0.25 and 0.85.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. MIGRATION COST IN ms AND BYTES (ratio 0.25 and 0.85)")
    print("=" * 78)
    for ratio in (0.25, 0.85):
        print(f"\n  ratio={ratio:.2f}:")
        print(f"    {'policy':<16} {'migrations':>10}  {'mean RTT (ms)':>14}  {'total mig cost (ms)':>20}  "
              f"{'mean bytes/mig':>15}  {'total bytes':>14}")
        rows = [r for r in agg_rows if r["sweep"] == "ambiguity" and r["knob_value"] == ratio
                and not r["policy_id"].startswith("oracle")]
        rows.sort(key=lambda r: r["policy_id"])
        for r in rows:
            mig = float(r["migration_count_mean"])
            mean_rtt = r["mean_migration_rtt_ms_mean"]
            total_rtt = float(r["total_migration_rtt_ms_mean"])
            mean_bytes = r["mean_migration_bytes_per_migration_mean"]
            total_bytes = float(r["migration_bytes_total_mean"])
            print(f"    {r['policy_id']:<16} {mig:>10.1f}  "
                  f"{(float(mean_rtt) if mean_rtt else 0.0):>14.2f}  {total_rtt:>20.1f}  "
                  f"{(float(mean_bytes) if mean_bytes else 0.0):>15.1f}  {total_bytes:>14.0f}")

    dominance_path = OUTPUT_ROOT / "dominance_checks.csv"
    write_csv(dominance_path, DOMINANCE_COLUMNS, dominance_rows)
    print(f"\n[write] {dominance_path.relative_to(ROOT)} ({len(dominance_rows)} rows)")

    print(f"\n[done] {len(summary_rows)} (sweep, knob, seed, policy) rows replayed and reduced. "
          "See datasets/combined_objective/{summary,aggregate,dominance_checks}.csv for the raw numbers "
          "behind every figure printed above.")


if __name__ == "__main__":
    main()
