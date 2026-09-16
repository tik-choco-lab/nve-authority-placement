from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nve_dataset.util import write_csv
from nve_policy.config import load_policy_config
from nve_policy.dataset import Dataset
from nve_policy.engine import Evaluation, replay, write_decisions, write_metrics, write_results
from nve_policy.metrics import COMPARISON_COLUMNS, comparison_row, summarize
from nve_policy.policies import build_policy


def replay_dataset(root: str | Path, evaluation: Evaluation,
                   specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = Dataset(root)
    results = []
    for spec in specs:
        policy = build_policy(dataset, spec)
        outcome = replay(dataset, policy, evaluation)
        metrics = summarize(dataset, spec["id"], spec, outcome)
        write_results(dataset.root, spec["id"], outcome["rows"])
        write_metrics(dataset.root, spec["id"], metrics)
        decisions = getattr(policy, "decisions", None)
        if decisions:
            write_decisions(dataset.root, spec["id"], decisions)
        results.append(metrics)
    return results


def replay_all(roots: list[str | Path], config_path: str | Path) -> list[dict[str, Any]]:
    evaluation, specs = load_policy_config(config_path)
    return [row for root in roots for row in replay_dataset(root, evaluation, specs)]


def compare(roots: list[str | Path], output_path: str | Path) -> Path:
    rows = []
    for root in sorted(Path(item) for item in roots):
        for metrics_path in sorted((root / "output").glob("*/metrics.json")):
            rows.append(comparison_row(json.loads(metrics_path.read_text(encoding="utf-8"))))
    output = Path(output_path)
    write_csv(output, COMPARISON_COLUMNS, rows)
    return output
