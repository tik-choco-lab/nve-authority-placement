"""RTT-only policy registered with the authors' unchanged replay engine."""
from nve_policy.policies import REGISTRY
from nve_policy.policies.base import Policy


class MedoidPolicy(Policy):
    """Choose one peer from RTTs alone; migrate after each entity's first service.

    replay() initializes each entity at its original holder and calls
    on_request only after service. Returning the same peer on every call
    therefore causes at most one migration per entity, with the engine
    charging its normal migration RTT. No request attributes or history are
    consulted, and an entity initially at the medoid needs no migration.
    """

    def __init__(self, dataset, spec):
        super().__init__(dataset, spec)
        totals = dataset.rtt.sum(axis=1)
        self.medoid = min(range(len(dataset.peers)),
                          key=lambda p: (float(totals[p]), dataset.peers[p]))

    def on_request(self, request, authority):
        return self.medoid


REGISTRY["medoid"] = MedoidPolicy
