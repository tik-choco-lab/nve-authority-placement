from __future__ import annotations

from typing import Any

import numpy as np

from nve_policy.dataset import Dataset, Interaction
from nve_policy.policies.base import HoldingTimePolicy, Policy


class StaticPolicy(Policy):
    """Authority never leaves the peer it started on. No free parameters."""

    def on_request(self, request: Interaction, authority: int) -> int:
        return authority


class NearestPolicy(HoldingTimePolicy):
    """Authority follows the peer closest to the entity in the virtual world.

    Entities are static but peers move, so the target changes over time. Peer
    positions come from the sampled trace under the hold-last-value rule.
    """

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        super().__init__(dataset, spec)
        self._cache: dict[tuple[str, int], int] = {}

    def target(self, request: Interaction, authority: int) -> int:
        slot = self.dataset.position_slot(request.timestamp)
        key = (request.entity_id, slot)
        if key not in self._cache:
            entity = self.dataset.entities[request.entity_id]
            dx = self.dataset.px[:, slot] - entity.x
            dy = self.dataset.py[:, slot] - entity.y
            self._cache[key] = int(np.argmin(dx * dx + dy * dy))
        return self._cache[key]


class DemandWindow:
    """Sliding-window recent-demand bookkeeping, shared by every demand-driven policy.

    Factored out of ``WindowedDemandPolicy`` so that policies which must NOT
    inherit its post-migration cooldown (``HoldingTimePolicy.min_holding_time``)
    can still reuse the same windowing rules. ``EngagementAwarePolicy`` is the reason
    this exists: its hysteresis is the dwell time on the trigger side, not a
    cooldown after the fact, and stacking both forms of hysteresis on top of
    each other would double-count them.

    ``window`` must be stated explicitly as one of::

        {type: count, size: N}    the last N requests for that entity
        {type: time, seconds: W}  requests within the last W seconds
        {type: all}               every request seen so far

    ``use_weight`` decides whether request_types.csv weights count instead of
    a plain request tally.
    """

    def __init__(self, dataset: Dataset, window: dict[str, Any], use_weight: bool) -> None:
        if window.get("type") not in {"count", "time", "all"}:
            raise ValueError("window.type must be count, time or all")
        self.dataset = dataset
        self.window = window
        self.use_weight = use_weight
        self._history: dict[str, list[tuple[float, int, float]]] = {}
        self._demand: dict[str, np.ndarray] = {}

    def observe(self, request: Interaction) -> np.ndarray:
        history = self._history.setdefault(request.entity_id, [])
        demand = self._demand.setdefault(request.entity_id, np.zeros(len(self.dataset.peers)))
        weight = self.dataset.weights[request.request_type] if self.use_weight else 1.0
        history.append((request.timestamp, request.peer, weight))
        demand[request.peer] += weight
        if self.window["type"] == "count":
            size = int(self.window["size"])
            while len(history) > size:
                _, peer, dropped = history.pop(0)
                demand[peer] -= dropped
        elif self.window["type"] == "time":
            horizon = request.timestamp - float(self.window["seconds"])
            while history and history[0][0] < horizon:
                _, peer, dropped = history.pop(0)
                demand[peer] -= dropped
        return demand


class WindowedDemandPolicy(HoldingTimePolicy):
    """Shared machinery for policies driven by recent interaction demand.

    ``window`` must be stated explicitly as one of::

        {type: count, size: N}    the last N requests for that entity
        {type: time, seconds: W}  requests within the last W seconds
        {type: all}               every request seen so far

    ``use_interaction_weight`` decides whether request_types.csv weights count.
    Both are research variables and neither is defaulted.
    """

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        super().__init__(dataset, spec)
        for key in ("window", "use_interaction_weight"):
            if key not in spec:
                raise ValueError(f"policy '{spec.get('id')}' must set {key} explicitly")
        self.window = spec["window"]
        self.use_weight = bool(spec["use_interaction_weight"])
        self._demand_window = DemandWindow(dataset, self.window, self.use_weight)

    def _observe(self, request: Interaction) -> np.ndarray:
        return self._demand_window.observe(request)


class LowestRttPolicy(WindowedDemandPolicy):
    """Authority minimises total RTT to the peers that recently interacted.

    Note the degenerate case: with ``{type: count, size: 1}`` the answer is always
    the requesting peer itself (self-RTT is 0), which migrates on every request.
    That variant is kept available on purpose so the degeneracy shows up in the
    results instead of being hidden by a chosen default.
    """

    def target(self, request: Interaction, authority: int) -> int:
        demand = self._observe(request)
        return int(np.argmin(demand @ self.dataset.rtt))


class InteractionAwarePolicy(WindowedDemandPolicy):
    """Authority follows the peer with the largest recent share of requests."""

    def target(self, request: Interaction, authority: int) -> int:
        demand = self._observe(request)
        return int(np.argmax(demand))
