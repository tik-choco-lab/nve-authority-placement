"""Central experiment: how target switching speed interacts with authority placement.

The paper's "target switching interval" axis is `interaction.shifting.phase_duration`
(docs/DATASET_DESIGN.md section 4): `ScenarioPattern.dominant_at` rotates the
shifting scenario's dominant peer every `phase_duration` seconds, which is exactly
how fast the entity's demand target changes. `config.yaml`'s suite only ever
generates one value (20.0), so this sweep generates its own datasets and does not
touch `config.yaml`, `policies.yaml`, or the baseline `datasets/small` suite.

Every `engagement_aware` variant below sets `relevant_range: 1.0e9`. The generator picks
the shifting scenario's dominant peer independently of peer coordinates (see
`nve_dataset/scenario/patterns.py` - `dominant_at` never looks at `px`/`py`), so on
this dataset spatial range and interaction demand are uncorrelated by
construction. A finite `relevant_range` would inject release/candidacy noise that
has nothing to do with the switching-speed question this experiment asks, so the
range is set effectively infinite to structurally disable release/candidacy and
isolate the push path (theta_G, theta_D) on its own. Evaluating release/candidacy
behaviour is out of scope here.
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
SWITCHING_ROOT = ROOT / "datasets" / "switching"

# The swept knob: docs/DATASET_DESIGN.md section 4 / patterns.py `dominant_at`.
# 80s is close to "no switching within the 300s run" (fewer than 4 phases total).
PHASE_DURATIONS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
SEEDS = [1, 2, 3, 4, 5]

SUMMARY_COLUMNS = [
    "phase_duration", "seed", "policy_id",
    "total_requests", "mean_latency_ms", "p95_latency_ms", "migration_count",
    "max_requests_per_peer", "request_imbalance",
]
AGGREGATE_COLUMNS = [
    "phase_duration", "policy_id",
    "mean_latency_ms_mean", "mean_latency_ms_std",
    "p95_latency_ms_mean", "p95_latency_ms_std",
    "migration_count_mean", "migration_count_std",
    "max_requests_per_peer_mean", "max_requests_per_peer_std",
    "request_imbalance_mean", "request_imbalance_std",
]


def phase_dir_name(phase_duration: float) -> str:
    return f"phase_{int(round(phase_duration)):03d}"


def dataset_root(phase_duration: float, seed: int) -> Path:
    return SWITCHING_ROOT / phase_dir_name(phase_duration) / f"seed_{seed:03d}"


def ensure_dataset(base_config: dict[str, Any], phase_duration: float, seed: int) -> Path:
    root = dataset_root(phase_duration, seed)
    manifest_path = root / "manifest.json"
    if root.exists() and manifest_path.is_file():
        try:
            manifest_path.read_text(encoding="utf-8")
            print(f"[skip]  {root.relative_to(ROOT)} already exists, manifest readable")
            return root
        except OSError:
            pass  # fall through and regenerate: manifest exists but is unreadable

    config = copy.deepcopy(base_config)
    config["simulation"]["duration"] = 300.0
    config["peers"]["count"] = 20
    config["entities"]["count"] = 5
    config["interaction"]["shifting"]["phase_duration"] = phase_duration
    run_config = config_for_run(config, "shifting", seed)
    print(f"[gen]   {root.relative_to(ROOT)} (phase_duration={phase_duration}, seed={seed})")
    generate_dataset(run_config, output_dir=root, analyze=False)
    return root


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")
    specs = build_policy_specs()

    summary_rows: list[dict[str, Any]] = []
    equivalence_checks = 0

    for phase_duration in PHASE_DURATIONS:
        for seed in SEEDS:
            root = ensure_dataset(base_config, phase_duration, seed)
            dataset = Dataset(root)
            outcomes: dict[str, dict[str, Any]] = {}
            for spec in specs:
                policy = build_policy(dataset, spec)
                outcome = replay(dataset, policy, EVALUATION)
                outcomes[spec["id"]] = outcome
                metrics = summarize(dataset, spec["id"], spec, outcome)
                summary_rows.append({
                    "phase_duration": phase_duration,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": metrics["total_requests"],
                    "mean_latency_ms": f"{metrics['interaction_latency_ms']['mean']:.6f}",
                    "p95_latency_ms": f"{metrics['interaction_latency_ms']['p95']:.6f}",
                    "migration_count": metrics["migration_count"],
                    "max_requests_per_peer": metrics["processing_load"]["max_requests_per_peer"],
                    "request_imbalance": f"{metrics['processing_load']['request_imbalance']:.6f}",
                })
            check_equivalence(outcomes, f"phase_duration={phase_duration} seed={seed}")
            equivalence_checks += 1
            print(f"[ok]    phase_duration={phase_duration} seed={seed}: "
                  f"{len(specs)} policies replayed, engagement_g00_d0.0 == naive_target_aware")

    summary_path = SWITCHING_ROOT / "sweep_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    aggregate_rows = aggregate_over_seeds(summary_rows, "phase_duration")
    aggregate_path = SWITCHING_ROOT / "sweep_aggregate.csv"
    write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregate_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(aggregate_rows)} rows)")

    print(f"\nEquivalence check (engagement_g00_d0.0 == naive_target_aware): "
          f"PASSED on all {equivalence_checks} (phase_duration, seed) datasets.")


if __name__ == "__main__":
    main()
