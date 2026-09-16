"""Central experiment: does the proposed policy's release/candidacy path
(theta_R) do anything, once interaction locality and spatial locality are
actually correlated?

BACKGROUND. Every other experiment in this directory (ambiguity_sweep.py,
switching_sweep.py) sets `relevant_range: 1.0e9` on every `engagement_aware`
variant, structurally disabling `RELEASE_OUT_OF_RANGE` / `CLAIM_OWNERLESS` /
`KEEP_TARGET_OUT_OF_RANGE`. The reason is not that theta_R is uninteresting:
it is that the `shifting` scenario picks each entity's dominant peer
(`nve_dataset/scenario/patterns.py` `ScenarioPattern.dominant_at`) with no
reference whatsoever to peer coordinates, while peer motion
(`nve_dataset/mobility/models.py`) has never referenced the scenario
either. Interaction locality and spatial locality are independent random
variables by construction. Under any finite theta_R the demand argmax is
essentially never inside the required range of the current holder, so the
push rule (`EngagementAwarePolicy` steps 3-7 in its docstring) can basically never
fire and the whole release/candidacy path degenerates into noise unrelated
to demand. This experiment evaluates that dependence explicitly.

THE FIX (opt-in, off by default: `peers.target_coupling` in
`nve_dataset/config.py` / `nve_dataset/mobility/models.py`). When enabled, a
peer that the scenario currently calls one or more entities' dominant peer
heads its random-walk step straight at the centroid of those entities'
(static) positions, with probability `bias_probability`, instead of drawing a
uniform-random heading. Every existing dataset, sweep and test keeps
generating byte-identical output with the option absent (see
tests/test_target_coupling.py) - this experiment is the only caller that
turns it on.

WHY THIS DATASET IS SMALLER AND FASTER THAN config.yaml's DEFAULTS. The
`shifting` scenario reassigns *which physical peer* is "dominant" for a
given entity at every `phase_duration` boundary (a full walk down the peer-id
list, not the same peer moving) - see `dominant_at`. That means a newly
"dominant" peer starts its phase whatever distance it happens to be from the
entity, and the bias above gives it only `phase_duration` seconds to close
that gap before the spotlight moves to a *different* peer and the bias
target changes again. At config.yaml's defaults (`movement_speed: 2.0`,
world `1000x1000`, diagonal ~1414) and this experiment's required
`phase_duration: 10.0`, the travel budget is `speed * phase_duration = 20`
units against a world that size - hopeless; measured, bias_probability=1.0
there only pulls the mean entity-to-dominant-peer distance from 531 to 502.
So this experiment shrinks the world to 150x150 (diagonal ~212) and raises
`movement_speed` to 50.0 (budget 500, more than twice the diagonal), which
`datasets/range/coupling_check.csv` (written by this script, see
`measure_dominant_distance` below) confirms is enough: with the option on,
77.8% of interactions land within 5 units of their entity's current dominant
peer and 95.7% within 100 units, versus 0.3% and 63.1% respectively with the
option off on the *same* world/speed/phase_duration/ratio. `bias_probability`
is pinned to 1.0 (maximum correlation) throughout, since the question this
sweep asks is what a finite theta_R does GIVEN correlation, not how much
correlation `bias_probability` itself buys - that second question is exactly
what step 3 above already answered on its own.

THE SWEPT KNOB is `relevant_range` (theta_R), passed only to the policy
replay stage (dataset generation does not depend on it - the same five
coupled datasets, one per seed, are reused for every theta_R value). Values
were chosen directly from the coupling_check.csv numbers above to span
"almost always in range" to "almost never in range":

    theta_R      fraction of interactions with their current dominant
                 peer within theta_R (coupled dataset, mean over 5 seeds)
    1.0e9        1.000  (the "infinite" baseline every other experiment uses)
    150.0        0.994  (~= world diagonal)
    75.0         0.921
    30.0         0.836
    8.0          0.790
    3.0          0.452
    0.3          0.054

`phase_duration` is held at 10.0s and `dominant_peer_ratio` at 0.25 per the
task: docs/POLICY_EVALUATION.md's dwell-time discussion and
ambiguity_sweep.py's own findings identify low dominant_peer_ratio as the
regime where theta_D has actual transient churn to filter, so this is the
setting most likely to show theta_R and theta_D interacting.
"""
from __future__ import annotations

import copy
import statistics
from pathlib import Path
from typing import Any

from experiments.common import aggregate_over_seeds, EVALUATION
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset
from nve_dataset.scenario import build_pattern
from nve_dataset.util import write_csv
from nve_policy.dataset import Dataset
from nve_policy.engine import replay
from nve_policy.metrics import summarize
from nve_policy.policies import build_policy
from nve_policy.policies.engagement_aware import REASONS

ROOT = Path(__file__).resolve().parents[1]
RANGE_ROOT = ROOT / "datasets" / "range"

DURATION = 300.0
PEER_COUNT = 20
ENTITY_COUNT = 5
PHASE_DURATION = 10.0        # FIXED per task
DOMINANT_PEER_RATIO = 0.25   # FIXED per task: theta_D's known-to-help regime
WORLD_SIZE = 150.0           # shrunk from config.yaml's 1000.0, see module docstring
MOVEMENT_SPEED = 50.0        # raised from config.yaml's 2.0, see module docstring
BIAS_PROBABILITY = 1.0       # maximum correlation; theta_R is the only swept knob here
SEEDS = [1, 2, 3, 4, 5]

DEMAND_WINDOW = {"type": "count", "size": 10}
DWELL_VALUES = [0.0, 2.0]   # "a couple of theta_D values" (task wording)
ENGAGEMENT_THRESHOLD = 0.5        # the non-degenerate arm (g00 is naive_target_aware in disguise)

# theta_R sweep, chosen from the measured coupling_check numbers (see module
# docstring): spans "almost always in range" (1.0e9, 150) through the middle
# of the curve (75, 30, 8) down to "almost never in range" (3, 0.3).
RELEVANT_RANGES = [1.0e9, 150.0, 75.0, 30.0, 8.0, 3.0, 0.3]

# Candidate thresholds for the standalone coupling-verification measurement
# (step 3): finer-grained than RELEVANT_RANGES so the curve used to justify
# the sweep values above is itself visible in coupling_check.csv.
DISTANCE_THRESHOLDS = [0.3, 1.0, 3.0, 5.0, 8.0, 15.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]

SUMMARY_COLUMNS = [
    "relevant_range", "seed", "policy_id",
    "total_requests", "mean_latency_ms", "p95_latency_ms", "migration_count",
    "max_requests_per_peer", "request_imbalance",
]
AGGREGATE_COLUMNS = [
    "relevant_range", "policy_id",
    "mean_latency_ms_mean", "mean_latency_ms_std",
    "p95_latency_ms_mean", "p95_latency_ms_std",
    "migration_count_mean", "migration_count_std",
    "max_requests_per_peer_mean", "max_requests_per_peer_std",
    "request_imbalance_mean", "request_imbalance_std",
]
REASON_COLUMNS = ["relevant_range", "policy_id", "reason", "count", "fraction"]
COUPLING_COLUMNS = ["coupling", "seed", "threshold", "fraction_within"]


def _base_experiment_config(base_config: dict[str, Any], coupled: bool) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["simulation"]["duration"] = DURATION
    config["world"]["width"] = WORLD_SIZE
    config["world"]["height"] = WORLD_SIZE
    config["peers"]["count"] = PEER_COUNT
    config["peers"]["movement_speed"] = MOVEMENT_SPEED
    config["entities"]["count"] = ENTITY_COUNT
    config["interaction"]["shifting"]["phase_duration"] = PHASE_DURATION
    config["interaction"]["shifting"]["dominant_peer_ratio"] = DOMINANT_PEER_RATIO
    if coupled:
        config["peers"]["target_coupling"] = {"enabled": True, "bias_probability": BIAS_PROBABILITY}
    return config


def dataset_root(coupled: bool, seed: int) -> Path:
    return RANGE_ROOT / ("coupled" if coupled else "uncoupled") / f"seed_{seed:03d}"


def ensure_dataset(base_config: dict[str, Any], coupled: bool, seed: int) -> Path:
    root = dataset_root(coupled, seed)
    manifest_path = root / "manifest.json"
    if root.exists() and manifest_path.is_file():
        try:
            manifest_path.read_text(encoding="utf-8")
            print(f"[skip]  {root.relative_to(ROOT)} already exists, manifest readable")
            return root
        except OSError:
            pass
    run_config = config_for_run(_base_experiment_config(base_config, coupled), "shifting", seed)
    print(f"[gen]   {root.relative_to(ROOT)} (coupled={coupled}, seed={seed})")
    generate_dataset(run_config, output_dir=root, analyze=False)
    return root


# --- Step 3: verify the coupling on the raw trace, independent of any policy ---

def dominant_peer_distances(dataset: Dataset) -> list[float]:
    """For every interaction, the distance between the entity and whichever
    peer the scenario currently calls that entity's dominant peer - NOT the
    peer the request actually went to (choose_peer only sends
    `dominant_peer_ratio` of requests there). This is the geometric quantity
    `EngagementAwarePolicy`'s push rule (target_peer) and release rule (holder) both
    depend on, measured directly on the trace before any policy runs.
    """
    pattern = build_pattern(dataset.config, dataset.peers, list(dataset.entities))
    distances = []
    for interaction in dataset.interactions:
        entity = dataset.entities[interaction.entity_id]
        dominant_peer_id = pattern.dominant_at(interaction.entity_id, interaction.timestamp)
        if dominant_peer_id is None:
            continue
        peer_index = dataset.peer_index[dominant_peer_id]
        slot = dataset.position_slot(interaction.timestamp)
        dx = dataset.px[peer_index, slot] - entity.x
        dy = dataset.py[peer_index, slot] - entity.y
        distances.append(float((dx * dx + dy * dy) ** 0.5))
    return distances


def run_coupling_check(base_config: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    per_coupling_fractions: dict[bool, dict[float, list[float]]] = {True: {}, False: {}}
    for coupled in (True, False):
        for seed in SEEDS:
            root = ensure_dataset(base_config, coupled, seed)
            distances = dominant_peer_distances(Dataset(root))
            n = len(distances)
            for threshold in DISTANCE_THRESHOLDS:
                fraction = sum(1 for d in distances if d <= threshold) / n
                rows.append({"coupling": "on" if coupled else "off", "seed": seed,
                             "threshold": threshold, "fraction_within": f"{fraction:.6f}"})
                per_coupling_fractions[coupled].setdefault(threshold, []).append(fraction)

    path = RANGE_ROOT / "coupling_check.csv"
    write_csv(path, COUPLING_COLUMNS, rows)
    print(f"[write] {path.relative_to(ROOT)} ({len(rows)} rows)")

    print("\nStep 3 - coupling verification (mean fraction of interactions whose entity is "
          f"within `threshold` of its CURRENT scenario-dominant peer, over {len(SEEDS)} seeds):\n")
    print(f"{'threshold':>10}  {'coupling ON':>12}  {'coupling OFF':>13}")
    for threshold in DISTANCE_THRESHOLDS:
        on_mean = statistics.fmean(per_coupling_fractions[True][threshold])
        off_mean = statistics.fmean(per_coupling_fractions[False][threshold])
        print(f"{threshold:>10.1f}  {on_mean:>12.3f}  {off_mean:>13.3f}")


# --- Step 4: the policy comparison itself -----------------------------------

def build_specs(relevant_range: float) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"id": "static", "type": "static"},
        {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
        {
            "id": "naive_target_aware", "type": "interaction_aware",
            "window": DEMAND_WINDOW, "use_interaction_weight": False, "min_holding_time_s": 0.0,
        },
    ]
    for dwell in DWELL_VALUES:
        specs.append({
            "id": f"engagement_g05_d{dwell}",
            "type": "engagement_aware",
            "target_rule": "demand_argmax",
            "engagement_threshold": ENGAGEMENT_THRESHOLD,
            "dwell_time_s": dwell,
            "relevant_range": relevant_range,
            "window": DEMAND_WINDOW,
            "use_interaction_weight": False,
        })
    specs.append({"id": "oracle_a1", "type": "oracle", "migration_penalty_weight": 1.0})
    return specs


def range_label(value: float) -> str:
    return "inf" if value >= 1.0e8 else f"{value:g}"


def main() -> None:
    base_config = load_config(ROOT / "config.yaml")

    run_coupling_check(base_config)

    summary_rows: list[dict[str, Any]] = []
    # (relevant_range, policy_id) -> {reason: count}
    reason_counts: dict[tuple[float, str], dict[str, int]] = {}

    for relevant_range in RELEVANT_RANGES:
        for seed in SEEDS:
            root = dataset_root(True, seed)  # already generated by run_coupling_check
            dataset = Dataset(root)
            for spec in build_specs(relevant_range):
                policy = build_policy(dataset, spec)
                outcome = replay(dataset, policy, EVALUATION)
                metrics = summarize(dataset, spec["id"], spec, outcome)
                summary_rows.append({
                    "relevant_range": relevant_range,
                    "seed": seed,
                    "policy_id": spec["id"],
                    "total_requests": metrics["total_requests"],
                    "mean_latency_ms": f"{metrics['interaction_latency_ms']['mean']:.6f}",
                    "p95_latency_ms": f"{metrics['interaction_latency_ms']['p95']:.6f}",
                    "migration_count": metrics["migration_count"],
                    "max_requests_per_peer": metrics["processing_load"]["max_requests_per_peer"],
                    "request_imbalance": f"{metrics['processing_load']['request_imbalance']:.6f}",
                })
                decisions = getattr(policy, "decisions", None)
                if decisions:
                    counts = reason_counts.setdefault((relevant_range, spec["id"]), {})
                    for row in decisions:
                        reason = row["reason"]
                        assert reason in REASONS
                        counts[reason] = counts.get(reason, 0) + 1
        print(f"[ok]    relevant_range={range_label(relevant_range)}: "
              f"{len(SEEDS)} seeds x {len(build_specs(relevant_range))} policies replayed")

    summary_path = RANGE_ROOT / "sweep_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"[write] {summary_path.relative_to(ROOT)} ({len(summary_rows)} rows)")

    aggregate_rows = aggregate_over_seeds(summary_rows, "relevant_range")
    aggregate_path = RANGE_ROOT / "sweep_aggregate.csv"
    write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregate_rows)
    print(f"[write] {aggregate_path.relative_to(ROOT)} ({len(aggregate_rows)} rows)")

    reason_rows: list[dict[str, Any]] = []
    for (relevant_range, policy_id), counts in sorted(
            reason_counts.items(), key=lambda item: (item[0][0], item[0][1])):
        total = sum(counts.values())
        for reason in sorted(REASONS):
            count = counts.get(reason, 0)
            if count == 0 and reason not in counts:
                continue
            reason_rows.append({
                "relevant_range": relevant_range, "policy_id": policy_id, "reason": reason,
                "count": count, "fraction": f"{count / total:.6f}",
            })
    reason_path = RANGE_ROOT / "reason_codes.csv"
    write_csv(reason_path, REASON_COLUMNS, reason_rows)
    print(f"[write] {reason_path.relative_to(ROOT)} ({len(reason_rows)} rows)")

    # --- readable stdout tables ---
    print("\nStep 4 - metrics by theta_R (mean over 5 seeds):\n")
    aggregate_by_key = {(row["relevant_range"], row["policy_id"]): row for row in aggregate_rows}
    policy_ids = [spec["id"] for spec in build_specs(RELEVANT_RANGES[0])]
    header = f"{'theta_R':>10}  {'policy':>20}  {'mean_ms':>9}  {'p95_ms':>9}  {'migr':>7}  {'imbalance':>10}"
    print(header)
    for relevant_range in RELEVANT_RANGES:
        for policy_id in policy_ids:
            row = aggregate_by_key[(relevant_range, policy_id)]
            print(f"{range_label(relevant_range):>10}  {policy_id:>20}  "
                  f"{float(row['mean_latency_ms_mean']):>9.3f}  {float(row['p95_latency_ms_mean']):>9.3f}  "
                  f"{float(row['migration_count_mean']):>7.1f}  {float(row['request_imbalance_mean']):>10.3f}")
        print()

    print("Step 4 - engagement_aware reason-code distribution by theta_R (counts summed over 5 seeds):\n")
    for relevant_range in RELEVANT_RANGES:
        for dwell in DWELL_VALUES:
            policy_id = f"engagement_g05_d{dwell}"
            counts = reason_counts.get((relevant_range, policy_id), {})
            total = sum(counts.values())
            if total == 0:
                continue
            print(f"theta_R={range_label(relevant_range)}  policy={policy_id}  (n={total})")
            for reason in sorted(REASONS):
                count = counts.get(reason, 0)
                if count:
                    print(f"    {reason:<26} {count:>6}  ({count / total:.3f})")
        print()


if __name__ == "__main__":
    main()
