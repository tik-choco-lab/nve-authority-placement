from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping


def derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(seed: int, namespace: str) -> random.Random:
    return random.Random(derived_seed(seed, namespace))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_choice(rng: random.Random, items: list[str], probabilities: list[float]) -> str:
    point = rng.random() * sum(probabilities)
    cumulative = 0.0
    for item, probability in zip(items, probabilities, strict=True):
        cumulative += probability
        if point < cumulative:
            return item
    return items[-1]
