"""Engagement-aware authority placement (the paper's proposed method, Section 4).

Every symbol below maps 1:1 onto Section 4's equations. Read the required
spec keys as theta_G (``engagement_threshold``), theta_D (``dwell_time_s``) and
theta_R (``relevant_range``); none of them has a default (a
hidden default silently picks one point of the design space).

``target_rule`` (also required, also no default) selects WHICH single-peer
rule the dwell/engagement/range machinery below is hysteresis for:

- ``"demand_argmax"``: ``P* = argmax(demand)`` (the original reading of
  Section 4 - the peer with the largest recent share of requests).
- ``"rtt_weighted"``: ``P* = argmin(demand @ dataset.rtt)`` (the same
  ``LowestRttPolicy`` rule from ``simple.py``, promoted here so it can be
  gated by the same theta_D/theta_R/theta_G machinery as the argmax rule).
  Because self-RTT is 0, this reduces exactly to ``demand_argmax`` whenever
  demand is concentrated on a single peer - the two rules are the same
  selector pair as ``InteractionAwarePolicy`` vs.
  ``LowestRttPolicy`` in ``simple.py``, just re-hosted behind the dwell gate.

Whichever rule is named, ``_track_dwell`` measures dwell on THAT rule's
target, not always the argmax: the dwell timer's entire job is to ask "how
long has the peer this rule would currently hand authority to held that
role", so if the rule is ``rtt_weighted`` the clock must start over exactly
when the RTT-weighted target peer changes, not when the (possibly different)
demand-argmax peer changes. Conflating the two would silently re-introduce
the argmax rule's hysteresis underneath the RTT-weighted rule's decision.

Two design decisions are load-bearing and deliberate, not omissions:

1. ``G(E, P, t)`` (engagement) has no column in the dataset. It is instantiated
   here as the NORMALISED SHARE of recent interaction demand,
   ``demand[P] / demand.sum()``, which lands in [0, 1] exactly like the
   paper's G. Normalising (rather than using the raw request count) is what
   makes ``theta_G`` scale-free: a raw count would make the threshold mean
   something different for every ``window`` size and every interaction
   rate, which would turn "we chose theta_G = 0.5" into an artifact of the
   window instead of a statement about the workload.

2. ``A(E, t) = bot`` (ownerless) has no representation in the engine
   contract: ``on_request(request, authority) -> int`` must always name a
   concrete peer, because the replay model requires every request to be
   served by ``authority_before`` (see docs/POLICY_EVALUATION.md section 1).
   So bot is POLICY-INTERNAL ONLY: this class keeps a private
   ``_ownerless`` flag per entity, and whenever the paper's model would put
   the entity into bot, ``on_request`` still returns a real peer (the
   unchanged ``authority`` argument) and pays no migration cost. The engine
   never learns that the entity was, conceptually, unowned. The migration
   cost that bot would otherwise defer is instead paid once, later, at the
   moment a peer actually claims the entity (``CLAIM_OWNERLESS``). This is
   the only way to represent bot inside a harness that was built around
   "authority is always some peer" - see the note on ``RELEASE_OUT_OF_RANGE``
   vs. ``CLAIM_OWNERLESS`` below for exactly which request row the cost
   lands on.

This class derives directly from ``Policy``, NOT ``HoldingTimePolicy``. Its
hysteresis is the dwell time a peer must hold the ``target_rule``'s target
identity before it is worth transferring to (theta_D, checked on the *trigger* side, before any
migration happens). ``HoldingTimePolicy.min_holding_time`` is a *cooldown
after* a migration. Stacking both would double-count hysteresis and make it
impossible to attribute a suppressed migration to either knob. The sliding
demand window is still shared machinery, so it is reused via
``DemandWindow`` (factored out of ``WindowedDemandPolicy`` in simple.py for
exactly this purpose) rather than copy-pasted or inherited.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nve_policy.dataset import Dataset, Interaction
from nve_policy.policies.base import Policy
from nve_policy.policies.simple import DemandWindow

REQUIRED_KEYS = (
    "engagement_threshold", "dwell_time_s", "relevant_range", "window", "use_interaction_weight",
    "target_rule",
)

# The two rules `target_rule` may select. Kept as a mapping from spec value to
# the function computing P* from the demand vector, so on_request's step 3-7
# logic and _track_dwell (which must track the SAME rule's target) share one
# definition instead of two independently-maintained argmax/argmin branches.
TARGET_RULES = {
    "demand_argmax": lambda demand, rtt: int(np.argmax(demand)),
    "rtt_weighted": lambda demand, rtt: int(np.argmin(demand @ rtt)),
}

# The complete set of reason codes this policy can emit. Kept as a frozenset
# so tests (and any future analysis code) can assert recorded reasons are a
# subset of exactly this, rather than re-typing the seven strings.
REASONS = frozenset({
    "RELEASE_OUT_OF_RANGE",
    "KEEP_OWNERLESS_NO_CANDIDATE",
    "CLAIM_OWNERLESS",
    "KEEP_SAME_TARGET",
    "KEEP_TARGET_OUT_OF_RANGE",
    "KEEP_LOW_ENGAGEMENT",
    "KEEP_DWELL_NOT_MET",
    "TRANSFER_TARGET_CHANGE",
})


def _fmt(value: Any) -> str:
    return "" if value in ("", None) else f"{float(value):.6f}"


class EngagementAwarePolicy(Policy):
    """Push authority toward the peer named by ``target_rule`` (the demand
    argmax, or the RTT-weighted argmin over recent demand - see module
    docstring), gated by dwell time, engagement share and relevant range.

    Decision order per request (implements Section 4 exactly):

    1. If the entity is currently owned and the holder has drifted more than
       ``theta_R`` from the entity's (static) position, it is released
       (``RELEASE_OUT_OF_RANGE``). The same call immediately falls through to
       step 2 - "fall through" is implemented literally here: the candidate
       search below runs in this same call, so a peer that is already in
       range at the instant of release claims the entity on the spot
       (``CLAIM_OWNERLESS``, cost paid now) instead of waiting for the next
       request. Only when that search finds nobody does the row keep the
       ``RELEASE_OUT_OF_RANGE`` reason - which is what makes it distinct from
       ``KEEP_OWNERLESS_NO_CANDIDATE`` (an entity that was *already* ownerless
       coming into the call and still finds no one). Either way this branch
       never returns anything other than ``authority`` unchanged unless a
       claim actually happens, so a bare release never costs anything by
       itself; whatever migration cost is eventually paid is attributed
       entirely to the claim.
    2. If ownerless: search every peer's position (not just the requester's -
       candidacy is about who *could* serve the entity right now, mirrored on
       ``NearestPolicy``'s use of ``dataset.px``/``py``) for those within
       ``theta_R``. Nobody in range keeps it ownerless
       (``KEEP_OWNERLESS_NO_CANDIDATE``). Some peers in range resolve to the
       nearest one deterministically (ties broken by lowest peer index via
       ``np.argmin``): this models Raft-like candidacy converging on exactly
       one holder, and a deterministic tie-break is what keeps the replay
       reproducible run to run.
    3-7. Otherwise the entity is owned and in range: compute the demand
       argmax ``P*``, and migrate to it only if it differs from the current
       holder, ``P*`` is itself within ``theta_R``, its engagement share clears
       ``theta_G``, and it has held the ``target_rule`` identity continuously
       for at least ``theta_D`` seconds. ``P*`` itself is ``argmax(demand)``
       when ``target_rule: demand_argmax`` and ``argmin(demand @ rtt)`` when
       ``target_rule: rtt_weighted`` (see module docstring); everything past
       this point in the decision order is identical for both, since both
       rules produce nothing but a single candidate peer index. The
       engagement check uses the instantaneous share at time t (matching the
       paper's G(E,P,t)); only the target-*identity* condition is required to
       hold continuously.

    The ``theta_R`` guard on ``P*`` is what keeps the push rule (step 7) and
    the release rule (step 1) from cancelling each other out. Without it the
    two rules contradict: step 7 hands authority to ``P*`` wherever it
    happens to be, step 1 then evicts that same holder on the very next
    request for being out of range, a nearby peer claims, and the unchanged
    ``P*`` pulls authority straight back out again. The result is a migration
    on essentially every request, and ``theta_D`` cannot damp it because the
    target never actually changes - the oscillation is between the two
    *rules*, not between two targets. Requiring the transfer target to be
    inside the same ``theta_R`` that governs release makes the pair
    consistent: authority is only ever granted to a peer that is allowed to
    keep it.

    ``_observe`` (via ``DemandWindow.observe``) runs on every single request,
    including the release/claim rows, so the demand window never develops a
    hole just because the entity happened to be ownerless for a while.
    """

    requires_future = False

    def __init__(self, dataset: Dataset, spec: dict[str, Any]) -> None:
        super().__init__(dataset, spec)
        for key in REQUIRED_KEYS:
            if key not in spec:
                raise ValueError(f"policy '{spec.get('id')}' must set {key} explicitly")
        self.engagement_threshold = float(spec["engagement_threshold"])
        self.dwell_time_s = float(spec["dwell_time_s"])
        self.relevant_range = float(spec["relevant_range"])
        self.use_weight = bool(spec["use_interaction_weight"])
        self._demand_window = DemandWindow(dataset, spec["window"], self.use_weight)
        target_rule = spec["target_rule"]
        if target_rule not in TARGET_RULES:
            raise ValueError(
                f"policy '{spec.get('id')}' has target_rule={target_rule!r}, "
                f"must be one of {sorted(TARGET_RULES)}")
        self.target_rule = target_rule
        self._compute_target = TARGET_RULES[target_rule]

        self._ownerless: dict[str, bool] = {}
        self._target_since: dict[str, tuple[int, float]] = {}
        self._dist2_cache: dict[tuple[str, int], np.ndarray] = {}
        self.decisions: list[dict[str, Any]] = []

    # -- geometry, mirrors NearestPolicy's hold-last-value position lookup ----

    def _dist2(self, entity_id: str, timestamp: float) -> np.ndarray:
        slot = self.dataset.position_slot(timestamp)
        key = (entity_id, slot)
        cached = self._dist2_cache.get(key)
        if cached is None:
            entity = self.dataset.entities[entity_id]
            dx = self.dataset.px[:, slot] - entity.x
            dy = self.dataset.py[:, slot] - entity.y
            cached = dx * dx + dy * dy
            self._dist2_cache[key] = cached
        return cached

    def _distance_to(self, entity_id: str, peer: int, timestamp: float) -> float:
        return float(np.sqrt(self._dist2(entity_id, timestamp)[peer]))

    def _nearest_in_range(self, entity_id: str, timestamp: float) -> int | None:
        dist2 = self._dist2(entity_id, timestamp)
        within_range = dist2 <= self.relevant_range ** 2
        if not np.any(within_range):
            return None
        masked = np.where(within_range, dist2, np.inf)
        return int(np.argmin(masked))

    # -- dwell bookkeeping -----------------------------------------------

    def _track_dwell(self, request: Interaction, demand: np.ndarray) -> None:
        """Restart the dwell clock exactly when the SELECTED rule's target
        peer changes identity - not when the demand argmax changes, if
        ``target_rule`` names ``rtt_weighted``. Using the wrong rule's target
        here would silently gate the RTT-weighted push with the argmax
        rule's hysteresis instead of its own.
        """
        target_peer = self._compute_target(demand, self.dataset.rtt)
        entry = self._target_since.get(request.entity_id)
        if entry is None or entry[0] != target_peer:
            self._target_since[request.entity_id] = (target_peer, request.timestamp)

    # -- decision + logging -------------------------------------------------

    def _record(self, request: Interaction, before: int, after: int, reason: str, *,
                target_peer: str = "", engagement: Any = "", dwell_s: Any = "",
                holder_distance: Any = "") -> None:
        self.decisions.append({
            "timestamp": f"{request.timestamp:.6f}",
            "request_id": request.request_id,
            "entity_id": request.entity_id,
            "request_peer": self.dataset.peers[request.peer],
            "authority_before": self.dataset.peers[before],
            "authority_after": self.dataset.peers[after],
            "reason": reason,
            "target_peer": target_peer,
            "engagement": _fmt(engagement),
            "dwell_s": _fmt(dwell_s),
            "holder_distance": _fmt(holder_distance),
        })

    def on_request(self, request: Interaction, authority: int) -> int:
        demand = self._demand_window.observe(request)
        self._track_dwell(request, demand)

        entity_id = request.entity_id
        ownerless = self._ownerless.get(entity_id, False)
        holder_distance: float | None = None
        fresh_release = False

        if not ownerless:
            holder_distance = self._distance_to(entity_id, authority, request.timestamp)
            if holder_distance > self.relevant_range:
                ownerless = True
                fresh_release = True
                self._ownerless[entity_id] = True

        if ownerless:
            candidate = self._nearest_in_range(entity_id, request.timestamp)
            if candidate is None:
                reason = "RELEASE_OUT_OF_RANGE" if fresh_release else "KEEP_OWNERLESS_NO_CANDIDATE"
                result = authority
            else:
                reason = "CLAIM_OWNERLESS"
                self._ownerless[entity_id] = False
                result = candidate
            self._record(request, authority, result, reason, holder_distance=holder_distance)
            return result

        target_peer, since = self._target_since[entity_id]
        engagement = float(demand[target_peer] / demand.sum())
        dwell = request.timestamp - since

        if target_peer == authority:
            reason, result = "KEEP_SAME_TARGET", authority
        elif self._distance_to(entity_id, target_peer, request.timestamp) > self.relevant_range:
            reason, result = "KEEP_TARGET_OUT_OF_RANGE", authority
        elif engagement < self.engagement_threshold:
            reason, result = "KEEP_LOW_ENGAGEMENT", authority
        elif dwell < self.dwell_time_s:
            reason, result = "KEEP_DWELL_NOT_MET", authority
        else:
            reason, result = "TRANSFER_TARGET_CHANGE", target_peer

        self._record(request, authority, result, reason,
                     target_peer=self.dataset.peers[target_peer],
                     engagement=engagement, dwell_s=dwell, holder_distance=holder_distance)
        return result
