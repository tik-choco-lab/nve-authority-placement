from __future__ import annotations

import csv
import itertools
from pathlib import Path

import pytest

from nve_dataset.config import config_for_run
from nve_dataset.generator import generate_dataset
from nve_dataset.util import sha256_file
from nve_policy.config import load_policy_config
from nve_policy.dataset import Dataset
from nve_policy.engine import Evaluation, replay
from nve_policy.metrics import summarize
from nve_policy.policies import build_policy
from nve_policy.policies.engagement_aware import REASONS, EngagementAwarePolicy
from nve_policy.runner import replay_dataset
from tests.test_generator import compact_config

EVALUATION = Evaluation(request_bytes=128, response_bytes=128, migration_bandwidth_mbps=None)


def tiny_dataset(tmp_path: Path, scenario: str = "concentrated", seed: int = 31) -> Dataset:
    config = compact_config()
    config["peers"]["count"] = 3
    config["entities"]["count"] = 1
    config["interaction"]["rate_per_entity"] = 1.5
    root = generate_dataset(config_for_run(config, scenario, seed), tmp_path / "ds", analyze=False)
    return Dataset(root)


def churny_dataset(tmp_path: Path, scenario: str = "shifting", seed: int = 31) -> Dataset:
    """More peers and a higher request rate than tiny_dataset.

    engagement_aware needs several distinct demand-argmax changes to show a
    measurable difference between variants; tiny_dataset's ~6 requests over
    3 peers rarely produces more than one migration under any settings.
    """
    config = compact_config()
    config["peers"]["count"] = 5
    config["entities"]["count"] = 1
    config["interaction"]["rate_per_entity"] = 15.0
    config["interaction"]["shifting"]["phase_duration"] = 0.5
    root = generate_dataset(config_for_run(config, scenario, seed), tmp_path / "ds", analyze=False)
    return Dataset(root)


def run(dataset: Dataset, spec: dict) -> dict:
    return replay(dataset, build_policy(dataset, spec), EVALUATION)


def objective(dataset: Dataset, outcome: dict, alpha: float) -> float:
    total = sum(outcome["latencies"])
    for row in outcome["rows"]:
        if row["migration_occurred"] == 1:
            total += alpha * float(row["migration_network_latency_ms"])
    return total


def test_static_never_migrates(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    outcome = run(dataset, {"id": "static", "type": "static"})
    assert outcome["migrations"] == 0
    for row in outcome["rows"]:
        entity = dataset.entities[row["entity_id"]]
        assert row["authority_before"] == dataset.peers[entity.initial_authority]


def test_research_variables_have_no_hidden_defaults(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    for spec in (
        {"id": "n", "type": "nearest"},                                   # min_holding_time_s
        {"id": "l", "type": "lowest_rtt", "min_holding_time_s": 0.0},     # window / weight flag
        {"id": "o", "type": "oracle"},                                    # migration_penalty_weight
    ):
        with pytest.raises(ValueError, match="explicitly"):
            build_policy(dataset, spec)


ENGAGEMENT_AWARE_BASE = {
    "type": "engagement_aware",
    "target_rule": "demand_argmax",
    "relevant_range": 1500.0,
    "engagement_threshold": 0.5,
    "dwell_time_s": 2.0,
    "window": {"type": "count", "size": 10},
    "use_interaction_weight": False,
}


@pytest.mark.parametrize("missing", [
    "relevant_range", "engagement_threshold", "dwell_time_s", "window", "use_interaction_weight",
    "target_rule",
])
def test_engagement_aware_has_no_hidden_defaults(tmp_path: Path, missing: str) -> None:
    dataset = tiny_dataset(tmp_path)
    spec = {"id": "h", **ENGAGEMENT_AWARE_BASE}
    del spec[missing]
    with pytest.raises(ValueError, match="explicitly"):
        build_policy(dataset, spec)


def test_engagement_aware_dwell_time_suppresses_migrations(tmp_path: Path) -> None:
    dataset = churny_dataset(tmp_path)
    base = {**ENGAGEMENT_AWARE_BASE, "engagement_threshold": 0.0, "relevant_range": 1500.0,
            "window": {"type": "count", "size": 3}}
    quick = run(dataset, {"id": "quick", **base, "dwell_time_s": 0.0})
    slow = run(dataset, {"id": "slow", **base, "dwell_time_s": 1000.0})
    assert slow["migrations"] < quick["migrations"]


def test_engagement_aware_engagement_threshold_suppresses_migrations(tmp_path: Path) -> None:
    dataset = churny_dataset(tmp_path)
    base = {**ENGAGEMENT_AWARE_BASE, "dwell_time_s": 0.0, "relevant_range": 1500.0,
            "window": {"type": "count", "size": 3}}
    permissive = run(dataset, {"id": "low", **base, "engagement_threshold": 0.0})
    strict = run(dataset, {"id": "high", **base, "engagement_threshold": 0.95})
    assert strict["migrations"] < permissive["migrations"]


def test_engagement_aware_requires_future_is_false() -> None:
    assert EngagementAwarePolicy.requires_future is False


def test_engagement_aware_zero_threshold_zero_dwell_matches_naive_interaction_aware(tmp_path: Path) -> None:
    """Regression for the switching-interval sweep's equivalence check
    (experiments/switching_sweep.py). With engagement_threshold=0, dwell_time_s=0 and
    relevant_range effectively infinite, engagement_aware's push rule (steps 3-7 in
    the class docstring) degenerates to exactly interaction_aware's argmax
    target with min_holding_time_s=0.0: no hysteresis is left on either side,
    so both migrate to the same count-window argmax on the same requests. A
    divergence here would mean one of the two implementations observes the
    demand window at a different point (an off-by-one), not a legitimate
    behavioural difference.
    """
    dataset = churny_dataset(tmp_path)
    window = {"type": "count", "size": 10}
    naive = run(dataset, {
        "id": "naive", "type": "interaction_aware",
        "window": window, "use_interaction_weight": False, "min_holding_time_s": 0.0,
    })
    engagement = run(dataset, {
        "id": "engagement", "type": "engagement_aware",
        "target_rule": "demand_argmax",
        "engagement_threshold": 0.0, "dwell_time_s": 0.0, "relevant_range": 1.0e9,
        "window": window, "use_interaction_weight": False,
    })
    naive_seq = [row["authority_after"] for row in naive["rows"]]
    engagement_seq = [row["authority_after"] for row in engagement["rows"]]
    assert engagement_seq == naive_seq
    assert engagement["migrations"] == naive["migrations"]
    assert sum(engagement["latencies"]) == sum(naive["latencies"])


def test_engagement_aware_rtt_weighted_zero_threshold_zero_dwell_matches_lowest_rtt(tmp_path: Path) -> None:
    """Mirror of the demand_argmax equivalence test above, for the
    RTT-weighted rule. With engagement_threshold=0, dwell_time_s=0 and relevant_range
    effectively infinite, engagement_aware's push rule with
    target_rule=rtt_weighted degenerates to exactly LowestRttPolicy's
    argmin(demand @ rtt) target with min_holding_time_s=0.0: no hysteresis is
    left on either side, so both migrate to the same count-window RTT-argmin
    on the same requests. A divergence here would mean the two
    implementations observe the demand window or the RTT matrix differently,
    not a legitimate behavioural difference.
    """
    dataset = churny_dataset(tmp_path)
    window = {"type": "count", "size": 10}
    lowest_rtt = run(dataset, {
        "id": "lowest_rtt", "type": "lowest_rtt",
        "window": window, "use_interaction_weight": False, "min_holding_time_s": 0.0,
    })
    engagement = run(dataset, {
        "id": "engagement_rtt", "type": "engagement_aware",
        "target_rule": "rtt_weighted",
        "engagement_threshold": 0.0, "dwell_time_s": 0.0, "relevant_range": 1.0e9,
        "window": window, "use_interaction_weight": False,
    })
    # Byte-identical per-request output: compare the full engine-level row
    # dicts (timestamp, authority_before/after, latency, traffic, migration
    # fields), not just the authority_after sequence, since "rows" has an
    # identical schema regardless of which policy produced it.
    assert engagement["rows"] == lowest_rtt["rows"]
    assert engagement["migrations"] == lowest_rtt["migrations"]
    assert sum(engagement["latencies"]) == sum(lowest_rtt["latencies"])


def test_engagement_aware_tiny_range_still_serves_every_request(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    spec = {"id": "tiny", **ENGAGEMENT_AWARE_BASE, "relevant_range": 1.0}
    policy = build_policy(dataset, spec)
    outcome = replay(dataset, policy, EVALUATION)
    assert len(outcome["rows"]) == len(dataset.interactions)
    reasons = {row["reason"] for row in policy.decisions}
    assert reasons <= REASONS
    assert "RELEASE_OUT_OF_RANGE" in reasons
    assert "CLAIM_OWNERLESS" in reasons or "KEEP_OWNERLESS_NO_CANDIDATE" in reasons


def test_engagement_aware_reason_codes_are_documented(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    spec = {"id": "h", **ENGAGEMENT_AWARE_BASE}
    policy = build_policy(dataset, spec)
    replay(dataset, policy, EVALUATION)
    assert policy.decisions
    assert {row["reason"] for row in policy.decisions} <= REASONS


def test_min_holding_time_suppresses_migrations(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    base = {"type": "lowest_rtt", "window": {"type": "count", "size": 1},
            "use_interaction_weight": False}
    free = run(dataset, {"id": "free", **base, "min_holding_time_s": 0.0})
    held = run(dataset, {"id": "held", **base, "min_holding_time_s": 10.0})
    assert held["migrations"] < free["migrations"]


def test_lowest_rtt_window_one_targets_the_requesting_peer(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    outcome = run(dataset, {"id": "w1", "type": "lowest_rtt", "min_holding_time_s": 0.0,
                            "window": {"type": "count", "size": 1}, "use_interaction_weight": False})
    # Documented degeneracy: self-RTT is zero, so the requester always wins.
    assert all(row["authority_after"] == row["request_peer"] for row in outcome["rows"])


@pytest.mark.parametrize("alpha", [0.0, 1.0, 10.0])
def test_oracle_matches_brute_force_optimum(tmp_path: Path, alpha: float) -> None:
    dataset = tiny_dataset(tmp_path)
    requests = dataset.interactions[:7]
    assert len({r.entity_id for r in requests}) == 1, "tiny dataset must hold one entity"
    entity = dataset.entities[requests[0].entity_id]
    peers = range(len(dataset.peers))

    def cost(assignment: tuple[int, ...]) -> float:
        total = 0.0
        previous = entity.initial_authority
        for request, authority in zip(requests, assignment):
            total += dataset.rtt[request.peer, authority] + dataset.processing_delay_ms
            total += alpha * dataset.rtt[previous, authority]
            previous = authority
        return total

    best = min(cost((entity.initial_authority, *tail))
               for tail in itertools.product(peers, repeat=len(requests) - 1))
    policy = build_policy(dataset, {"id": "o", "type": "oracle", "migration_penalty_weight": alpha})
    planned = cost(tuple(policy._plan[r.index] for r in requests))
    assert planned == pytest.approx(best)


@pytest.mark.parametrize("scenario", ["uniform", "concentrated", "shifting", "burst"])
def test_oracle_beats_every_online_policy_on_its_own_objective(tmp_path: Path, scenario: str) -> None:
    dataset = tiny_dataset(tmp_path, scenario=scenario)
    _, specs = load_policy_config(Path(__file__).parents[1] / "policies.yaml")
    for alpha, oracle_id in [(0.0, "oracle_a0"), (1.0, "oracle_a1"), (10.0, "oracle_a10")]:
        scores = {spec["id"]: objective(dataset, run(dataset, spec), alpha) for spec in specs}
        assert scores[oracle_id] <= min(scores.values()) + 1e-9


def test_replay_leaves_input_untouched(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    files = sorted((dataset.root / "input").glob("*.csv"))
    before = {path.name: sha256_file(path) for path in files}
    evaluation, specs = load_policy_config(Path(__file__).parents[1] / "policies.yaml")
    replay_dataset(dataset.root, evaluation, specs)
    assert {path.name: sha256_file(path) for path in files} == before


@pytest.mark.parametrize("spec", [
    {"id": "static", "type": "static"},
    {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
])
def test_served_per_peer_sums_to_total_requests(tmp_path: Path, spec: dict) -> None:
    dataset = churny_dataset(tmp_path)
    outcome = run(dataset, spec)
    assert sum(outcome["served_per_peer"]) == len(dataset.interactions)


def test_request_imbalance_is_bounded(tmp_path: Path) -> None:
    dataset = churny_dataset(tmp_path)
    peer_count = len(dataset.peers)
    for spec in (
        {"id": "static", "type": "static"},
        {"id": "nearest", "type": "nearest", "min_holding_time_s": 0.0},
        {"id": "engagement", **ENGAGEMENT_AWARE_BASE},
    ):
        outcome = run(dataset, spec)
        metrics = summarize(dataset, spec["id"], spec, outcome)
        imbalance = metrics["processing_load"]["request_imbalance"]
        assert imbalance >= 1.0
        assert imbalance <= peer_count + 1e-9


def test_static_active_peer_count_bounded_by_entity_count(tmp_path: Path) -> None:
    """Static never migrates (see test_static_never_migrates), so authority
    never moves off the entities' initial holders: at most one active peer per
    entity, which pins active_peer_count's meaning as a concentration signal
    rather than an artifact of window/policy choice."""
    dataset = churny_dataset(tmp_path)
    outcome = run(dataset, {"id": "static", "type": "static"})
    metrics = summarize(dataset, "static", {"id": "static", "type": "static"}, outcome)
    assert metrics["processing_load"]["active_peer_count"] <= len(dataset.entities)


def test_results_are_written_per_policy(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path)
    evaluation, specs = load_policy_config(Path(__file__).parents[1] / "policies.yaml")
    replay_dataset(dataset.root, evaluation, specs)
    for spec in specs:
        results = dataset.root / "output" / spec["id"] / "results.csv"
        rows = list(csv.DictReader(results.open(newline="", encoding="utf-8")))
        assert len(rows) == len(dataset.interactions)
        assert (dataset.root / "output" / spec["id"] / "metrics.json").exists()
