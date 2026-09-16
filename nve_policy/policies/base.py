from __future__ import annotations

from typing import Any

from nve_policy.dataset import Dataset, Interaction


class Policy:
    """Online authority placement policy.

    ``on_request`` is called *after* the request has been served by
    ``authority_before`` and must return the authority in effect from now on.
    An online policy may only use what it has already been shown; the engine
    never hands it a future request.
    """

    requires_future = False

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        self.dataset = dataset
        self.spec = spec

    def on_request(self, request: Interaction, authority: int) -> int:
        raise NotImplementedError


class HoldingTimePolicy(Policy):
    """Base for policies that may migrate, with the holding-time knob made explicit.

    ``min_holding_time_s`` is a research variable, so it is required rather than
    defaulted: a hidden 0.0 would silently pick one point of the design space.
    """

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        super().__init__(dataset, spec)
        if "min_holding_time_s" not in spec:
            raise ValueError(f"policy '{spec.get('id')}' must set min_holding_time_s explicitly")
        self.min_holding_time = float(spec["min_holding_time_s"])
        self._last_migration: dict[str, float] = {}

    def target(self, request: Interaction, authority: int) -> int:
        raise NotImplementedError

    def on_request(self, request: Interaction, authority: int) -> int:
        candidate = self.target(request, authority)
        if candidate == authority:
            return authority
        last = self._last_migration.get(request.entity_id, 0.0)
        if request.timestamp - last < self.min_holding_time:
            return authority
        self._last_migration[request.entity_id] = request.timestamp
        return candidate
