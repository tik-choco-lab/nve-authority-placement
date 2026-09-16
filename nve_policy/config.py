from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nve_policy.engine import Evaluation


def load_policy_config(path: str | Path) -> tuple[Evaluation, list[dict[str, Any]]]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "policies" not in config or "evaluation" not in config:
        raise ValueError("policy config needs 'evaluation' and 'policies' sections")
    section = config["evaluation"]
    for key in ("request_bytes", "response_bytes", "migration_bandwidth_mbps"):
        if key not in section:
            raise ValueError(f"evaluation.{key} must be stated explicitly (use null to disable)")
    evaluation = Evaluation(
        request_bytes=int(section["request_bytes"]),
        response_bytes=int(section["response_bytes"]),
        migration_bandwidth_mbps=(
            None if section["migration_bandwidth_mbps"] is None
            else float(section["migration_bandwidth_mbps"])
        ),
    )
    policies = config["policies"]
    identifiers = [spec.get("id") for spec in policies]
    if len(set(identifiers)) != len(identifiers) or None in identifiers:
        raise ValueError("every policy needs a unique id")
    return evaluation, policies
