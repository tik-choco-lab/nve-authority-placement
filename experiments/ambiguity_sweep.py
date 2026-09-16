"""Central experiment: how target-demand ambiguity interacts with authority placement.

`switching_sweep.py` swept `interaction.shifting.phase_duration` (how FAST the
entity's demand target changes) and found that, at this dataset's default
`dominant_peer_ratio: 0.85`, the naive target-aware policy already migrates
almost exactly once per genuine target change (e.g. 152.6 migrations vs. the
theoretical floor of 150 at a 10s phase over 300s with 5 entities - see
`docs/DATASET_DESIGN.md` section 4 / `nve_dataset/scenario/patterns.py`
`ScenarioPattern.dominant_at`). There is therefore essentially no transient
target churn left for the proposed policy's dwell timer (theta_D) to filter
out at that ratio, and theta_D shows up there as pure added latency.

The knob that actually creates that churn is `interaction.shifting.
dominant_peer_ratio`: how concentrated the entity's interaction demand is on
its dominant peer within a phase. A LOW ratio means the demand-window argmax
can flicker to a non-dominant peer for a window or two even though the
"true" dominant peer for the phase has not changed - exactly the kind of
transient churn theta_D exists to filter. This sweep holds `phase_duration`
fixed at 10.0s (so the true target-change rate is constant across the sweep)
and varies `dominant_peer_ratio` instead, to isolate the ambiguity axis from
the switching-speed axis that `switching_sweep.py` already covers.

Every `engagement_aware` variant below sets `relevant_range: 1.0e9` for the same
reason as `switching_sweep.py`: the generator picks the shifting scenario's
dominant peer independently of peer coordinates, so spatial range and
interaction demand are uncorrelated by construction here, and a finite
`relevant_range` would inject release/candidacy noise unrelated to the
ambiguity question this experiment asks.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from experiments.common import (
    aggregate_over_seeds,
    build_policy_specs,
    check_equivalence,
    EVALUATION,
)
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.metrics import summarize
from nve_policy.policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
AMBIGUITY_ROOT = ROOT / "datasets" / "ambiguity"

DURATION = 300.0
PEER_COUNT = 20
ENTITY_COUNT = 5
PHASE_DURATION = 10.0  # FIXED: the controlled variable here, unlike switching_sweep.py

# The swept knob: interaction.shifting.dominant_peer_ratio.
RATIOS = [0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25]
# Use 30 fixed seeds; existing readable dataset manifests are reused.
SEEDS = list(range(1, 31))

# Theoretical floor on migration_count for a policy that migrates exactly once
# per genuine target change (docs/DATASET_DESIGN.md section 4): the shifting
# scenario rotates each entity's dominant peer every PHASE_DURATION seconds,
# independently per entity, so an entity changes target (duration /
# phase_duration) times over the run. This is CONSTANT across the ratio sweep
# (phase_duration and entity_count are both fixed here) - the point of this
# experiment is to see how far naive_target_aware's ACTUAL migration count
# drifts above this fixed floor as ratio drops, not to vary the floor itself.
MIGRATION_FLOOR = (DURATION / PHASE_DURATION) * ENTITY_COUNT

SUMMARY_COLUMNS = [
    "dominant_peer_ratio", "seed", "policy_id",
    "total_requests", "mean_latency_ms", "p95_latency_ms", "migration_count",
    "max_requests_per_peer", "request_imbalance",
]
AGGREGATE_COLUMNS = [
    "dominant_peer_ratio", "policy_id",
    "mean_latency_ms_mean", "mean_latency_ms_std",
    "p95_latency_ms_mean", "p95_latency_ms_std",
    "migration_count_mean", "migration_count_std",
    "max_requests_per_peer_mean", "max_requests_per_peer_std",
    "request_imbalance_mean", "request_imbalance_std",
    "migration_floor",
]


def ratio_dir_name(ratio: float) -> str:
    return f"ratio_{round(ratio * 100):03d}"


def dataset_root(ratio: float, seed: int) -> Path:
    return AMBIGUITY_ROOT / ratio_dir_name(ratio) / f"seed_{seed:03d}"


def ensure_dataset(base_config: dict[str, Any], ratio: float, seed: int) -> Path:
    root = dataset_root(ratio, seed)
    manifest_path = root / "manifest.json"
    if root.exists() and manifest_path.is_file():
        try:
            manifest_path.read_text(encoding="utf-8")
            print(f"[skip]  {root.relative_to(ROOT)} already exists, manifest readable")
            return root
        except OSError:
            pass  # fall through and regenerate: manifest exists but is unreadable

    config = copy.deepcopy(base_config)
    config["simulation"]["duration"] = DURATION
    config["peers"]["count"] = PEER_COUNT
    config["entities"]["count"] = ENTITY_COUNT
    config["interaction"]["shifting"]["phase_duration"] = PHASE_DURATION
    config["interaction"]["shifting"]["dominant_peer_ratio"] = ratio
    run_config = config_for_run(config, "shifting", seed)
    print(f"[gen]   {root.relative_to(ROOT)} (dominant_peer_ratio={ratio}, seed={seed})")
    generate_dataset(run_config, output_dir=root, analyze=False)
    return root


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")
    specs = build_policy_specs()

    summary_rows: list[dict[str, Any]] = []
    equivalence_checks = 0

    for ratio in RATIOS:
        for seed in SEEDS:
            root = ensure_dataset(base_config, ratio, seed)
            dataset = Dataset(root)
            outcomes: dict[str, dict[str, Any]] = {}
            for spec in specs:
                policy = build_policy(dataset, spec)
                outcome = replay(dataset, policy, EVALUATION)
                outcomes[spec["id"]] = outcome
                metrics = summarize(dataset, spec["id"], spec, outcome)
                summary_rows.append({
                    "dominant_peer_ratio": ratio,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": metrics["total_requests"],
                    "mean_latency_ms": f"{metrics['interaction_latency_ms']['mean']:.6f}",
                    "p95_latency_ms": f"{metrics['interaction_latency_ms']['p95']:.6f}",
                    "migration_count": metrics["migration_count"],
                    "max_requests_per_peer": metrics["processing_load"]["max_requests_per_peer"],
                    "request_imbalance": f"{metrics['processing_load']['request_imbalance']:.6f}",
                })
            check_equivalence(outcomes, f"dominant_peer_ratio={ratio} seed={seed}")
            equivalence_checks += 1
            print(f"[ok]    dominant_peer_ratio={ratio} seed={seed}: "
                  f"{len(specs)} policies replayed, engagement_g00_d0.0 == naive_target_aware")

    summary_path = AMBIGUITY_ROOT / "sweep_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    aggregate_rows = aggregate_over_seeds(summary_rows, "dominant_peer_ratio")
    for row in aggregate_rows:
        row["migration_floor"] = f"{MIGRATION_FLOOR:.6f}"
    aggregate_path = AMBIGUITY_ROOT / "sweep_aggregate.csv"
    write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregate_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(aggregate_rows)} rows)")

    print(f"\nEquivalence check (engagement_g00_d0.0 == naive_target_aware): "
          f"PASSED on all {equivalence_checks} (dominant_peer_ratio, seed) datasets.")

    print(f"\nTheoretical migration floor ((duration / phase_duration) * entity_count) "
          f"= ({DURATION:g} / {PHASE_DURATION:g}) * {ENTITY_COUNT} = {MIGRATION_FLOOR:.1f}, "
          f"constant across this sweep (phase_duration and entity_count are fixed).")
    print("The gap between this floor and naive_target_aware's actual migration_count "
          "is the transient churn theta_D has available to filter:\n")
    aggregate_by_key = {(row["dominant_peer_ratio"], row["policy_id"]): row for row in aggregate_rows}
    print(f"{'ratio':>6}  {'floor':>8}  {'naive_actual':>13}  {'gap':>8}")
    for ratio in RATIOS:
        naive = aggregate_by_key[(ratio, "naive_target_aware")]
        actual = float(naive["migration_count_mean"])
        gap = actual - MIGRATION_FLOOR
        print(f"{ratio:>6.2f}  {MIGRATION_FLOOR:>8.1f}  {actual:>13.1f}  {gap:>8.1f}")


if __name__ == "__main__":
    main()
