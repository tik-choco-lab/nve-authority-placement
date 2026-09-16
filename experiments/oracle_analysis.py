"""Is the Oracle's advantage "the same task done better", or a different objective?

`ambiguity_sweep.py` reports, at `dominant_peer_ratio=0.25`, a 26.7ms Offline
Oracle vs. a 91.1ms best online policy (`naive_target_aware` /
`engagement_g00_d0.0`, which are provably identical - see `check_equivalence` in
`experiments/common.py`). The natural reading is "the online policies leave
latency on the table that the Oracle collects". This script tests a sharper
and less comfortable hypothesis: that the online policies and the Oracle are
not competing to solve the same problem better or worse. The online policies
place authority on the demand ARGMAX - the single peer that most recently
sent the most requests (`InteractionAwarePolicy` / `WindowedDemandPolicy` in
`nve_policy/policies/simple.py`, reused here via the shared `DemandWindow`).
The Oracle (`nve_policy/policies/oracle.py`) instead solves, exactly, a
per-entity shortest-path/DP over (request index, authority):

    dp[i][a] = serve(i, a) + min_b ( dp[i-1][b] + alpha * RTT[b][a] )

with `alpha = migration_penalty_weight = 1.0`. This is NOT a static
RTT-weighted 1-median formula - it is a sequential optimum with full
lookahead, trading serve cost against transition cost hop by hop, over the
*entire* future request sequence for that entity, not a single window. But
because migration is charged at `alpha * RTT` per hop and self-RTT is 0, the
Oracle has every incentive to stop chasing every single requester and instead
sit at whichever peer cheaply serves the whole cluster of near-future
requests at once - which, within one `shifting` phase (demand drawn from a
fixed dominant peer plus uniform noise over the other N-1 peers, `nve_dataset/
scenario/patterns.py` `ScenarioPattern.choose_peer`), looks a lot like the
RTT-weighted 1-median over that phase's actual requesters rather than the
single most frequent one. Whether it actually behaves that way is an
empirical question, not something to read off the formula - hence this
script.

Three concrete checks, all replayed on the datasets `ambiguity_sweep.py`
already generated under `datasets/ambiguity/ratio_*/seed_*` (not regenerated
here):

1. Agreement rate: does the peer the Oracle holds authority on (its
   `authority_before` for each request) equal the demand argmax under the
   exact same count-10 sliding window the online policies use? The
   hypothesis predicts high agreement when `dominant_peer_ratio` is high
   (demand is unambiguous - argmax and the "true" target coincide) and
   collapsing agreement as the ratio drops toward uniform (demand spread
   across many peers - argmax is noise, but the Oracle should still know
   where to sit).
2. The 1-median test: for each entity and each 10s phase (`interaction.
   shifting.phase_duration`), compute the peer that minimises the
   request-count-weighted sum of RTTs to every peer that actually sent a
   request for that entity during that phase. Compare the Oracle's actual
   per-request choice, and the demand argmax's per-request choice, against
   this phase-level weighted median. If the Oracle tracks the median far
   more closely than the argmax does, the hypothesis is confirmed.
3. Latency decomposition: of the Oracle's total latency advantage over
   `naive_target_aware` (same requests, same RTTs, row-for-row aligned
   replays), how much comes from self-service (`authority_before ==
   request_peer`, RTT = 0) vs. from sitting on a good low-RTT *remote* peer
   for requests it does not self-serve? A hypothesis that reduces entirely
   to "the Oracle just self-serves more" would be a much less interesting
   finding than one where remote placement matters too.

Outputs (mirrors `ambiguity_sweep.py`'s summary/aggregate split):

    datasets/ambiguity/oracle_analysis_summary.csv     one row per (ratio, seed)
    datasets/ambiguity/oracle_analysis_aggregate.csv   mean/std over seeds per ratio
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ambiguity_sweep import PHASE_DURATION, RATIOS, SEEDS, dataset_root
from experiments.common import DEMAND_WINDOW, EVALUATION, build_policy_specs
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.policies import build_policy
from nve_policy.policies.simple import DemandWindow

ROOT = Path(__file__).resolve().parents[1]
AMBIGUITY_ROOT = ROOT / "datasets" / "ambiguity"

SUMMARY_COLUMNS = [
    "dominant_peer_ratio", "seed", "total_requests",
    "oracle_mean_latency_ms", "online_mean_latency_ms",
    "agree_oracle_demand_count", "agree_oracle_demand_rate",
    "agree_oracle_median_count", "agree_oracle_median_rate",
    "agree_demand_median_count", "agree_demand_median_rate",
    "oracle_self_service_count", "oracle_self_service_rate",
    "total_latency_advantage_ms", "self_service_advantage_ms",
    "remote_advantage_ms", "self_service_advantage_fraction",
]
AGGREGATE_METRICS = [
    "agree_oracle_demand_rate", "agree_oracle_median_rate", "agree_demand_median_rate",
    "oracle_self_service_rate", "self_service_advantage_fraction",
    "oracle_mean_latency_ms", "online_mean_latency_ms",
]


def _demand_argmax_series(dataset: Dataset) -> list[int]:
    """Reproduce the exact count-10 demand argmax the online policies see.

    Reuses `nve_policy.policies.simple.DemandWindow` (the shared windowing
    implementation) rather than re-deriving the window logic here.
    """
    window = DemandWindow(dataset, DEMAND_WINDOW, use_weight=False)
    return [int(np.argmax(window.observe(request))) for request in dataset.interactions]


def _phase_weighted_median_series(dataset: Dataset) -> list[int]:
    """For each (entity, phase) compute the RTT-weighted 1-median over the peers
    that actually sent a request for that entity during that phase, weighted by
    how many requests each sent. Returns one entry per request in
    `dataset.interactions` order (same phase-median value for every request
    that falls in the same entity+phase)."""
    entity_indices: dict[str, list[int]] = defaultdict(list)
    for i, request in enumerate(dataset.interactions):
        entity_indices[request.entity_id].append(i)

    result = [-1] * len(dataset.interactions)
    for indices in entity_indices.values():
        by_phase: dict[int, list[int]] = defaultdict(list)
        for i in indices:
            phase = int(dataset.interactions[i].timestamp // PHASE_DURATION)
            by_phase[phase].append(i)
        for idxs in by_phase.values():
            demand = np.zeros(len(dataset.peers))
            for i in idxs:
                demand[dataset.interactions[i].peer] += 1.0
            median_peer = int(np.argmin(demand @ dataset.rtt))
            for i in idxs:
                result[i] = median_peer
    assert all(value >= 0 for value in result)
    return result


def analyze_dataset(ratio: float, seed: int) -> dict[str, Any]:
    root = dataset_root(ratio, seed)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"missing dataset at {root}; run experiments/ambiguity_sweep.py first "
            "(this script only replays existing datasets, it does not generate them)")
    dataset = Dataset(root)
    peer_index = dataset.peer_index

    specs = {spec["id"]: spec for spec in build_policy_specs()}
    oracle_outcome = replay(dataset, build_policy(dataset, specs["oracle_a1"]), EVALUATION)
    online_outcome = replay(dataset, build_policy(dataset, specs["naive_target_aware"]), EVALUATION)

    demand_argmax = _demand_argmax_series(dataset)

    # Sanity check: naive_target_aware (InteractionAwarePolicy, min_holding_time_s=0)
    # migrates to the demand argmax on every single request with zero hysteresis,
    # so our independently-driven DemandWindow must reproduce its authority_after
    # exactly. If this ever fails it means the reused window logic has diverged
    # from what the online policy actually does, and the agreement numbers below
    # would be meaningless.
    online_after = [peer_index[row["authority_after"]] for row in online_outcome["rows"]]
    if demand_argmax != online_after:
        first = next(i for i, (a, b) in enumerate(zip(demand_argmax, online_after)) if a != b)
        raise AssertionError(
            f"DemandWindow reproduction diverges from naive_target_aware at "
            f"ratio={ratio} seed={seed} request {first}: "
            f"demand_argmax={demand_argmax[first]} online_authority_after={online_after[first]}")

    oracle_before = [peer_index[row["authority_before"]] for row in oracle_outcome["rows"]]
    request_peer = [request.peer for request in dataset.interactions]
    phase_median = _phase_weighted_median_series(dataset)

    n = len(dataset.interactions)
    agree_od = sum(1 for i in range(n) if oracle_before[i] == demand_argmax[i])
    agree_om = sum(1 for i in range(n) if oracle_before[i] == phase_median[i])
    agree_dm = sum(1 for i in range(n) if demand_argmax[i] == phase_median[i])
    self_service = [oracle_before[i] == request_peer[i] for i in range(n)]
    self_service_count = sum(self_service)

    oracle_lat = oracle_outcome["latencies"]
    online_lat = online_outcome["latencies"]
    total_advantage = sum(online_lat) - sum(oracle_lat)
    self_advantage = sum(online_lat[i] - oracle_lat[i] for i in range(n) if self_service[i])
    remote_advantage = total_advantage - self_advantage

    return {
        "dominant_peer_ratio": ratio,
        "seed": seed,
        "total_requests": n,
        "oracle_mean_latency_ms": f"{statistics.fmean(oracle_lat):.6f}",
        "online_mean_latency_ms": f"{statistics.fmean(online_lat):.6f}",
        "agree_oracle_demand_count": agree_od,
        "agree_oracle_demand_rate": f"{agree_od / n:.6f}",
        "agree_oracle_median_count": agree_om,
        "agree_oracle_median_rate": f"{agree_om / n:.6f}",
        "agree_demand_median_count": agree_dm,
        "agree_demand_median_rate": f"{agree_dm / n:.6f}",
        "oracle_self_service_count": self_service_count,
        "oracle_self_service_rate": f"{self_service_count / n:.6f}",
        "total_latency_advantage_ms": f"{total_advantage:.6f}",
        "self_service_advantage_ms": f"{self_advantage:.6f}",
        "remote_advantage_ms": f"{remote_advantage:.6f}",
        "self_service_advantage_fraction": (
            f"{(self_advantage / total_advantage):.6f}" if total_advantage else "nan"
        ),
    }


def aggregate_over_seeds(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[row["dominant_peer_ratio"]].append(row)

    aggregate_rows = []
    for ratio in sorted(groups):
        rows = groups[ratio]
        agg: dict[str, Any] = {"dominant_peer_ratio": ratio, "n_seeds": len(rows)}
        for metric in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in rows]
            agg[f"{metric}_mean"] = f"{statistics.fmean(values):.6f}"
            agg[f"{metric}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.0"
        aggregate_rows.append(agg)
    return aggregate_rows


def main() -> None:
    summary_rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for seed in SEEDS:
            row = analyze_dataset(ratio, seed)
            summary_rows.append(row)
            print(f"[ok]    ratio={ratio} seed={seed}: "
                  f"agree(oracle,demand_argmax)={row['agree_oracle_demand_rate']}  "
                  f"agree(oracle,phase_median)={row['agree_oracle_median_rate']}  "
                  f"self_service_rate={row['oracle_self_service_rate']}")

    summary_path = AMBIGUITY_ROOT / "oracle_analysis_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"\n[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    aggregate_rows = aggregate_over_seeds(summary_rows)
    aggregate_columns = ["dominant_peer_ratio", "n_seeds"]
    for metric in AGGREGATE_METRICS:
        aggregate_columns += [f"{metric}_mean", f"{metric}_std"]
    aggregate_path = AMBIGUITY_ROOT / "oracle_analysis_aggregate.csv"
    write_csv(aggregate_path, aggregate_columns, aggregate_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(aggregate_rows)} rows)")

    by_ratio = {row["dominant_peer_ratio"]: row for row in aggregate_rows}

    print("\n=== Table 1: agreement rate, Oracle authority vs. count-10 demand argmax ===")
    print(f"{'ratio':>6}  {'agree(oracle,argmax)':>22}  {'std':>7}")
    for ratio in RATIOS:
        row = by_ratio[ratio]
        print(f"{ratio:>6.2f}  {row['agree_oracle_demand_rate_mean']:>22}  "
              f"{row['agree_oracle_demand_rate_std']:>7}")

    print("\n=== Table 2: 1-median hypothesis test ===")
    print(f"{'ratio':>6}  {'agree(oracle,median)':>21}  {'std':>7}  "
          f"{'agree(argmax,median)':>21}  {'std':>7}")
    for ratio in RATIOS:
        row = by_ratio[ratio]
        print(f"{ratio:>6.2f}  {row['agree_oracle_median_rate_mean']:>21}  "
              f"{row['agree_oracle_median_rate_std']:>7}  "
              f"{row['agree_demand_median_rate_mean']:>21}  "
              f"{row['agree_demand_median_rate_std']:>7}")

    print("\n=== Table 3: latency advantage decomposition (Oracle vs naive_target_aware) ===")
    print(f"{'ratio':>6}  {'oracle_ms':>10}  {'online_ms':>10}  "
          f"{'self_svc_rate':>13}  {'self_svc_frac_of_advantage':>26}")
    for ratio in RATIOS:
        row = by_ratio[ratio]
        print(f"{ratio:>6.2f}  {row['oracle_mean_latency_ms_mean']:>10}  "
              f"{row['online_mean_latency_ms_mean']:>10}  "
              f"{row['oracle_self_service_rate_mean']:>13}  "
              f"{row['self_service_advantage_fraction_mean']:>26}")


if __name__ == "__main__":
    main()
