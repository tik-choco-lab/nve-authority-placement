from __future__ import annotations

from typing import Any

from nve_policy.dataset import Dataset
from nve_policy.policies.base import Policy
from nve_policy.policies.engagement_aware import EngagementAwarePolicy
from nve_policy.policies.oracle import OraclePolicy
from nve_policy.policies.simple import (
    InteractionAwarePolicy,
    LowestRttPolicy,
    NearestPolicy,
    StaticPolicy,
)

REGISTRY: dict[str, type[Policy]] = {
    "static": StaticPolicy,
    "nearest": NearestPolicy,
    "lowest_rtt": LowestRttPolicy,
    "interaction_aware": InteractionAwarePolicy,
    "engagement_aware": EngagementAwarePolicy,
    "oracle": OraclePolicy,
}


def build_policy(dataset: Dataset, spec: dict[str, Any]) -> Policy:
    kind = spec.get("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown policy type {kind!r}; known types: {sorted(REGISTRY)}")
    return REGISTRY[kind](dataset, spec)


__all__ = ["REGISTRY", "Policy", "build_policy"]
