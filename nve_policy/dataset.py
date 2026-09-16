"""Read-only view of one generated dataset.

Nothing in this package writes to ``input/``. Policies receive this object and
may only look at what the replay contract allows them to look at.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nve_dataset.util import read_csv


@dataclass(frozen=True)
class Interaction:
    index: int
    timestamp: float
    request_id: str
    entity_id: str
    peer: int
    request_type: str


@dataclass(frozen=True)
class Entity:
    entity_id: str
    x: float
    y: float
    initial_authority: int
    state_size_bytes: int


class Dataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.config: dict[str, Any] = yaml.safe_load(
            (self.root / "config_effective.yaml").read_text(encoding="utf-8"))
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8"))

        peer_rows = read_csv(self.root / "input" / "peers.csv")
        self.peers: list[str] = [row["peer_id"] for row in peer_rows]
        self.peer_index: dict[str, int] = {peer: i for i, peer in enumerate(self.peers)}

        self.entities: dict[str, Entity] = {}
        for row in read_csv(self.root / "input" / "entities.csv"):
            self.entities[row["entity_id"]] = Entity(
                row["entity_id"], float(row["initial_x"]), float(row["initial_y"]),
                self.peer_index[row["initial_authority"]], int(row["state_size_bytes"]))

        count = len(self.peers)
        self.rtt = np.zeros((count, count), dtype=float)
        for row in read_csv(self.root / "input" / "network.csv"):
            self.rtt[self.peer_index[row["src_peer"]], self.peer_index[row["dst_peer"]]] = float(row["rtt_ms"])

        self.weights: dict[str, float] = {
            row["request_type"]: float(row["interaction_weight"])
            for row in read_csv(self.root / "input" / "request_types.csv")
        }

        self.interactions: list[Interaction] = [
            Interaction(i, float(row["timestamp"]), row["request_id"], row["entity_id"],
                        self.peer_index[row["peer_id"]], row["request_type"])
            for i, row in enumerate(read_csv(self.root / "input" / "interactions.csv"))
        ]
        self._load_positions()

    def _load_positions(self) -> None:
        rows = read_csv(self.root / "input" / "peer_positions.csv")
        times = sorted({float(row["timestamp"]) for row in rows})
        slot = {value: i for i, value in enumerate(times)}
        self.position_times = np.array(times, dtype=float)
        self.px = np.zeros((len(self.peers), len(times)), dtype=float)
        self.py = np.zeros((len(self.peers), len(times)), dtype=float)
        for row in rows:
            peer, column = self.peer_index[row["peer_id"]], slot[float(row["timestamp"])]
            self.px[peer, column] = float(row["x"])
            self.py[peer, column] = float(row["y"])

    def position_slot(self, timestamp: float) -> int:
        """Hold-last-value lookup: positions are samples, never interpolated here.

        The replay contract requires this rule to be stated rather than assumed,
        so it lives in one place and every policy shares it.
        """
        return max(0, int(np.searchsorted(self.position_times, timestamp, side="right")) - 1)

    @property
    def duration(self) -> float:
        return float(self.manifest["duration"])

    @property
    def processing_delay_ms(self) -> float:
        return float(self.config["processing"]["delay_ms"])
