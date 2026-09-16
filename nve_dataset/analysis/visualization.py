from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nve_dataset.util import read_csv


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_visualizations(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    output = root / "analysis" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    interactions = read_csv(root / "input" / "interactions.csv")
    peers = read_csv(root / "input" / "peers.csv")
    entities = read_csv(root / "input" / "entities.csv")
    network = read_csv(root / "input" / "network.csv")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    peer_ids = [row["peer_id"] for row in peers]
    paths: list[Path] = []

    counts = Counter(row["peer_id"] for row in interactions)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(peer_ids, [counts[p] for p in peer_ids])
    ax.set(xlabel="Peer", ylabel="Interaction count", title="Interaction count per peer")
    ax.tick_params(axis="x", rotation=45)
    paths.append(output / "01_interaction_count_per_peer.png"); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4))
    peer_index = {peer: index for index, peer in enumerate(peer_ids)}
    ax.scatter([float(r["timestamp"]) for r in interactions], [peer_index[r["peer_id"]] for r in interactions], s=5, alpha=0.5)
    ax.set(xlabel="Time (s)", ylabel="Requesting peer index", title="Interaction timeline")
    paths.append(output / "02_interaction_timeline.png"); _save(fig, paths[-1])

    window = max(1.0, float(manifest["duration"]) / 20.0)
    buckets: dict[int, Counter[str]] = defaultdict(Counter)
    for row in interactions:
        buckets[int(float(row["timestamp"]) // window)][row["peer_id"]] += 1
    bucket_ids = list(range(int(float(manifest["duration"]) // window) + 1))
    dominant_indices = [peer_index[buckets[b].most_common(1)[0][0]] if buckets[b] else -1 for b in bucket_ids]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step([(b + 0.5) * window for b in bucket_ids], dominant_indices, where="mid")
    ax.set(xlabel="Time (s)", ylabel="Empirical dominant peer index", title=f"Dominant peer by {window:g}s window")
    paths.append(output / "03_dominant_peer_changes.png"); _save(fig, paths[-1])

    rtts = [float(row["rtt_ms"]) for row in network if row["src_peer"] != row["dst_peer"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rtts, bins=min(30, max(5, len(rtts) // 4)))
    ax.set(xlabel="RTT (ms)", ylabel="Directed peer pairs", title="RTT distribution")
    paths.append(output / "04_rtt_distribution.png"); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter([float(r["initial_x"]) for r in peers], [float(r["initial_y"]) for r in peers], label="Peers", marker="o")
    ax.scatter([float(r["initial_x"]) for r in entities], [float(r["initial_y"]) for r in entities], label="Entities", marker="x", s=80)
    for row in peers:
        ax.annotate(row["peer_id"], (float(row["initial_x"]), float(row["initial_y"])), fontsize=7)
    for row in entities:
        ax.annotate(row["entity_id"], (float(row["initial_x"]), float(row["initial_y"])), fontsize=8)
    ax.set(xlabel="x", ylabel="y", title="Initial peer / entity positions")
    ax.legend()
    paths.append(output / "05_initial_positions.png"); _save(fig, paths[-1])

    hhi = []
    for bucket in bucket_ids:
        total = sum(buckets[bucket].values())
        hhi.append(sum((value / total) ** 2 for value in buckets[bucket].values()) if total else 0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([(b + 0.5) * window for b in bucket_ids], hhi, marker=".")
    ax.axhline(1 / len(peer_ids), color="gray", linestyle="--", label="Uniform baseline")
    ax.set(xlabel="Time (s)", ylabel="HHI", title=f"Interaction concentration by {window:g}s window", ylim=(0, 1))
    ax.legend()
    paths.append(output / "06_interaction_concentration.png"); _save(fig, paths[-1])
    return paths
