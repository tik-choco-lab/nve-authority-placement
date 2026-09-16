"""Follow-up to combined_objective.py: is theta_D > 0 EVER the global optimum?

combined_objective.py established the two endpoints of the ratio sweep: at
ratio=0.25 raising theta_D monotonically improves the combined objective but
never enough to catch Nearest, and at ratio=0.85 raising theta_D monotonically
hurts it (best is theta_D=0). Neither endpoint says whether some ratio in
between ever makes a theta_D>0 policy the outright best of all 16 (17,
counting the engagement_g00_d0.0/naive_target_aware duplicate separately) non-oracle
policies - which is the question that decides whether the paper's
contribution is "a positive band of ratios where dwelling helps" or "dwelling
only ever reduces the size of a loss."

This reads the already-produced datasets/combined_objective/{summary,
aggregate}.csv (no re-replay, no new datasets) and, for every
(dominant_peer_ratio, alpha) cell in the ambiguity sweep:
  (a) the single non-oracle policy minimizing mean combined cost,
  (b) the argmin over theta_D within the theta_G=0 (engagement_g00_*) family alone,
then flags any cell where (a) has theta_D>0, with a paired bootstrap margin
(same paired_bootstrap as combined_objective.py/load_significance.py, since
scipy is not a dependency) against the runner-up so a win inside the noise
is not reported as a win.
"""
from __future__ import annotations

import csv
from pathlib import Path

from experiments.load_significance import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
CO_ROOT = ROOT / "datasets" / "combined_objective"

RATIOS = [0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25]
ALPHAS = [0.5, 1.0, 2.0]
ENGAGEMENT_G00_DWELLS = ["0.0", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0"]


def load_summary():
    with (CO_ROOT / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["sweep"] == "ambiguity"]


def load_aggregate():
    with (CO_ROOT / "aggregate.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["sweep"] == "ambiguity"]


def dwell_of(policy_id: str) -> float | None:
    if policy_id == "naive_target_aware":
        return 0.0
    if policy_id.startswith("engagement_g00_d") or policy_id.startswith("engagement_g05_d"):
        return float(policy_id.split("_d")[-1])
    return None  # static, nearest


def main() -> None:
    summary = load_summary()
    agg = load_aggregate()
    non_oracle_ids = sorted({r["policy_id"] for r in agg if not r["policy_id"].startswith("oracle")})
    print(f"[data] {len(non_oracle_ids)} non-oracle policy ids: {non_oracle_ids}")
    print(f"[note] engagement_g00_d0.0 and naive_target_aware are verified byte-identical "
          f"(check_equivalence, asserted on every dataset in combined_objective.py) - "
          f"they are two ids for one policy, so there are 16 DISTINCT behaviours among "
          f"these {len(non_oracle_ids)} ids.")

    agg_index = {(r["knob_value"], r["policy_id"]): r for r in agg}

    print("\n" + "=" * 100)
    print("PER-CELL GLOBAL ARGMIN (all non-oracle policies) AND theta_G=0-FAMILY ARGMIN")
    print("=" * 100)
    winners_with_dwell_gt0 = []
    header = f"{'ratio':>6} {'alpha':>6}  {'GLOBAL ARGMIN':<16}{'cost(ms)':>12}  {'theta_D':>8} | " \
             f"{'g00-FAMILY ARGMIN':<16}{'cost(ms)':>12}  {'theta_D':>8}"
    print(header)
    for ratio in RATIOS:
        for alpha in ALPHAS:
            key = f"combined_ms_alpha{alpha:g}_mean"
            ratio_str = f"{ratio}"
            candidates = [agg_index[(r, pid)] for r in [ratio_str] if False]  # placeholder, replaced below
            cell_rows = [r for r in agg if r["policy_id"] in non_oracle_ids and float(r["knob_value"]) == ratio]
            global_best = min(cell_rows, key=lambda r: float(r[key]))
            h00_rows = [r for r in cell_rows if r["policy_id"].startswith("engagement_g00_d")]
            h00_best = min(h00_rows, key=lambda r: float(r[key]))
            gd = dwell_of(global_best["policy_id"])
            hd = dwell_of(h00_best["policy_id"])
            print(f"{ratio:>6.2f} {alpha:>6.1f}  {global_best['policy_id']:<16}{float(global_best[key]):>12.1f}  "
                  f"{('%.2f' % gd) if gd is not None else '  n/a':>8} | "
                  f"{h00_best['policy_id']:<16}{float(h00_best[key]):>12.1f}  {hd:>8.2f}")
            if gd is not None and gd > 0.0:
                winners_with_dwell_gt0.append((ratio, alpha, global_best, cell_rows, key))

    print("\n" + "=" * 100)
    print("IS theta_D > 0 EVER THE GLOBAL MINIMUM?")
    print("=" * 100)
    if not winners_with_dwell_gt0:
        print("NO. Across all 7 ratios x 3 alphas = 21 cells, the global argmin among all "
              "non-oracle policies is ALWAYS a theta_D=0 policy (static, nearest, "
              "naive_target_aware/engagement_g00_d0.0, or engagement_g05_d0.0). theta_D>0 never wins outright; "
              "at best (ratio<=~0.3-0.4, see combined_objective.py's crossover figure) it only "
              "narrows the gap to whichever theta_D=0 policy is winning.")
    else:
        for ratio, alpha, global_best, cell_rows, key in winners_with_dwell_gt0:
            runner_up = min((r for r in cell_rows if r["policy_id"] != global_best["policy_id"]),
                             key=lambda r: float(r[key]))
            raw_key = key[: -len("_mean")]  # aggregate.csv suffixes "_mean"; summary.csv (per-seed) does not
            a_vals = [float(r[raw_key]) for r in sorted(
                (r for r in summary if r["policy_id"] == global_best["policy_id"] and r["knob_value"] == f"{ratio}"),
                key=lambda r: int(r["seed"]))]
            b_vals = [float(r[raw_key]) for r in sorted(
                (r for r in summary if r["policy_id"] == runner_up["policy_id"] and r["knob_value"] == f"{ratio}"),
                key=lambda r: int(r["seed"]))]
            mean_diff, _std, lo, hi = paired_bootstrap(a_vals, b_vals)
            sig = "SIGNIFICANT (CI excludes 0)" if hi < 0 else "NOT significant (CI includes 0)"
            print(f"YES at ratio={ratio:.2f}, alpha={alpha:g}: winner={global_best['policy_id']} "
                  f"({float(global_best[key]):.1f} ms) beats runner-up={runner_up['policy_id']} "
                  f"({float(runner_up[key]):.1f} ms) by mean {mean_diff:+.1f} ms, "
                  f"95% CI [{lo:+.1f}, {hi:+.1f}] -> {sig}")

    print("\n" + "=" * 100)
    print("COMPACT TABLE: ratio=0.85 and ratio=0.25, alpha=1, combined objective in SECONDS")
    print("=" * 100)
    row_ids = ["static", "nearest", "engagement_g00_d0.0", "engagement_g00_d1.0", "engagement_g00_d2.0",
               "engagement_g00_d5.0", "engagement_g00_d10.0", "oracle_a1"]
    labels = {"static": "Static", "nearest": "Nearest", "engagement_g00_d0.0": "theta_D=0",
              "engagement_g00_d1.0": "theta_D=1", "engagement_g00_d2.0": "theta_D=2",
              "engagement_g00_d5.0": "theta_D=5", "engagement_g00_d10.0": "theta_D=10", "oracle_a1": "Oracle"}
    key = "combined_ms_alpha1_mean"
    print(f"{'policy':<12} {'ratio=0.85 (s)':>16} {'ratio=0.25 (s)':>16}")
    for pid in row_ids:
        r85 = next(r for r in agg if r["policy_id"] == pid and float(r["knob_value"]) == 0.85)
        r25 = next(r for r in agg if r["policy_id"] == pid and float(r["knob_value"]) == 0.25)
        print(f"{labels[pid]:<12} {float(r85[key]) / 1000:>16.2f} {float(r25[key]) / 1000:>16.2f}")



def best_dwell_gt0_vs_best_dwell_eq0() -> None:
    """The literal global-argmin table above can have a theta_D>0 policy edge
    out ANOTHER theta_D>0 policy for 2nd/3rd place while both still lose to
    the best theta_D=0 policy. Compare the groups directly: best-of-{theta_D>0 policies} vs
    best-of-{theta_D=0 policies}, per (ratio, alpha), with a paired bootstrap
    on that specific margin.
    """
    agg = load_aggregate()
    summary = load_summary()
    non_oracle = [r for r in agg if not r["policy_id"].startswith("oracle")]
    print("\n" + "=" * 100)
    print("BEST theta_D>0 POLICY vs BEST theta_D=0 POLICY, PER (ratio, alpha)")
    print("=" * 100)
    print(f"{'ratio':>6} {'alpha':>6}  {'best dwell>0':<16}{'cost(ms)':>12}  vs  {'best dwell=0':<16}{'cost(ms)':>12}  "
          f"{'mean_diff':>10}  {'95% CI':>24}  verdict")
    any_significant_win = []
    for ratio in RATIOS:
        for alpha in ALPHAS:
            key = f"combined_ms_alpha{alpha:g}_mean"
            raw_key = f"combined_ms_alpha{alpha:g}"
            cell = [r for r in non_oracle if float(r["knob_value"]) == ratio]
            gt0 = [r for r in cell if (dwell_of(r["policy_id"]) or 0.0) > 0.0]
            eq0 = [r for r in cell if (dwell_of(r["policy_id"]) == 0.0) or dwell_of(r["policy_id"]) is None]
            best_gt0 = min(gt0, key=lambda r: float(r[key]))
            best_eq0 = min(eq0, key=lambda r: float(r[key]))
            a_vals = [float(r[raw_key]) for r in sorted(
                (r for r in summary if r["policy_id"] == best_gt0["policy_id"] and r["knob_value"] == f"{ratio}"),
                key=lambda r: int(r["seed"]))]
            b_vals = [float(r[raw_key]) for r in sorted(
                (r for r in summary if r["policy_id"] == best_eq0["policy_id"] and r["knob_value"] == f"{ratio}"),
                key=lambda r: int(r["seed"]))]
            mean_diff, _std, lo, hi = paired_bootstrap(a_vals, b_vals)
            if hi < 0:
                verdict = "dwell>0 SIGNIFICANTLY BETTER"
                any_significant_win.append((ratio, alpha, best_gt0["policy_id"], best_eq0["policy_id"], mean_diff, lo, hi))
            elif lo > 0:
                verdict = "dwell=0 significantly better"
            else:
                verdict = "not distinguishable"
            print(f"{ratio:>6.2f} {alpha:>6.1f}  {best_gt0['policy_id']:<16}{float(best_gt0[key]):>12.1f}  vs  "
                  f"{best_eq0['policy_id']:<16}{float(best_eq0[key]):>12.1f}  {mean_diff:>10.1f}  "
                  f"[{lo:>+9.1f},{hi:>+9.1f}]  {verdict}")
    print()
    if any_significant_win:
        print("theta_D > 0 SIGNIFICANTLY beats the best theta_D=0 policy in these cells:")
        for ratio, alpha, gt0_id, eq0_id, mean_diff, lo, hi in any_significant_win:
            print(f"  ratio={ratio:.2f} alpha={alpha:g}: {gt0_id} beats {eq0_id} by {-mean_diff:.1f} ms "
                  f"(95% CI on margin [{-hi:.1f}, {-lo:.1f}])")
    else:
        print("theta_D > 0 NEVER significantly beats the best theta_D=0 policy, at any tested "
              "(ratio, alpha) cell. Every apparent 'theta_D>0 wins' instant in the raw global-argmin "
              "table above is either (a) beaten by a theta_D=0 policy once that policy is included "
              "in the comparison, or (b) not distinguishable from the best theta_D=0 policy at the "
              "95% level.")


if __name__ == "__main__":
    main()
    best_dwell_gt0_vs_best_dwell_eq0()
