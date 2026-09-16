from __future__ import annotations

from typing import Any

from nve_dataset.util import rng_for


def generate_network(config: dict[str, Any], peer_ids: list[str]) -> list[dict[str, Any]]:
    network = config["network"]
    model = network["model"]
    low = float(network[f"{model}_rtt_min_ms"])
    high = float(network[f"{model}_rtt_max_ms"])
    if low < 0 or high < low:
        raise ValueError("invalid RTT range")
    rng = rng_for(config["seed"], f"network-{model}")
    symmetric = bool(network.get("symmetric", True))
    values: dict[tuple[str, str], float] = {}
    for src in peer_ids:
        for dst in peer_ids:
            if src == dst:
                rtt = 0.0
            elif symmetric and (dst, src) in values:
                rtt = values[(dst, src)]
            else:
                # RTT is independent of virtual-space coordinates by design.
                rtt = rng.uniform(low, high)
            values[(src, dst)] = rtt
    return [
        {"src_peer": src, "dst_peer": dst, "rtt_ms": f"{values[(src, dst)]:.6f}"}
        for src in peer_ids for dst in peer_ids
    ]
