"""Offline optimal authority placement.

The oracle sees the whole request sequence in advance. Authority is independent
per entity, so each entity is solved exactly by a Viterbi-style dynamic program
over (request index, serving authority):

    dp[i][a] = latency(request i served by a)
             + min over b of ( dp[i-1][b] + alpha * RTT[b][a] )

Self-RTT is zero in every generated dataset, so staying put (``b == a``) costs
nothing and the minimisation needs no special case.

Two deliberate constraints keep the comparison honest:

* The first request of each entity is served by the dataset's ``initial_authority``,
  exactly as it is for every online policy. The oracle is optimal *subject to the
  same initial condition*, not optimal from a head start nobody else gets.
* ``migration_penalty_weight`` (alpha) is the objective, not a tuning constant.
  alpha = 0 makes migration free and yields the pure latency lower bound; larger
  alpha buys fewer migrations at the cost of latency. It must be stated
  explicitly, because choosing it for the reader would be choosing the conclusion.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from nve_policy.dataset import Dataset, Interaction
from nve_policy.policies.base import Policy

_UNREACHABLE = 1e18


class OraclePolicy(Policy):
    requires_future = True

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        super().__init__(dataset, spec)
        if "migration_penalty_weight" not in spec:
            raise ValueError(f"policy '{spec.get('id')}' must set migration_penalty_weight explicitly")
        self.alpha = float(spec["migration_penalty_weight"])
        self._plan: dict[int, int] = {}
        self._next_request: dict[int, int] = {}
        self._solve()

    def _solve(self) -> None:
        by_entity: dict[str, list[Interaction]] = defaultdict(list)
        for request in self.dataset.interactions:
            by_entity[request.entity_id].append(request)
        rtt = self.dataset.rtt
        transition = self.alpha * rtt
        peer_count = len(self.dataset.peers)

        for entity_id, requests in by_entity.items():
            for earlier, later in zip(requests, requests[1:]):
                self._next_request[earlier.index] = later.index
            entity = self.dataset.entities[entity_id]
            serve = rtt[[r.peer for r in requests], :] + self.dataset.processing_delay_ms

            # Request 0 is forced onto the dataset's initial authority.
            dp = np.full(peer_count, _UNREACHABLE)
            dp[entity.initial_authority] = serve[0, entity.initial_authority]
            back = np.empty((len(requests), peer_count), dtype=np.int32)
            back[0] = entity.initial_authority
            for step in range(1, len(requests)):
                candidates = dp[:, None] + transition
                back[step] = np.argmin(candidates, axis=0)
                dp = serve[step] + candidates[back[step], np.arange(peer_count)]

            authority = int(np.argmin(dp))
            for step in range(len(requests) - 1, 0, -1):
                self._plan[requests[step].index] = authority
                authority = int(back[step, authority])
            self._plan[requests[0].index] = entity.initial_authority

    def on_request(self, request: Interaction, authority: int) -> int:
        # Place the entity where the plan needs it before its next request arrives.
        following = self._next_request.get(request.index)
        return authority if following is None else self._plan[following]
