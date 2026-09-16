# NVE Authority Placement

[English](README.md) | [日本語](README.ja.md)

Reproducible synthetic workloads and policy evaluation for authority placement
in peer-to-peer networked virtual environments (NVEs).

An entity, such as a shared NPC or object, has one authoritative peer that
processes its requests. This project compares where that authority is placed
and when it migrates as interaction demand changes. Workloads are generated
from explicit parameters and random seeds; they are not real player logs.

## Quick start

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
Run these commands from the repository root in a shell supporting glob expansion,
such as Bash. Dependencies are recorded in `uv.lock`.

```bash
uv sync --locked

# Generate 12 demonstration datasets: four scenarios, seeds 1, 2, and 3.
# Each run has 10 peers, 3 entities, and a duration of 60 seconds.
uv run nve-dataset generate-suite --config config.yaml

# Validate one dataset and check regeneration against its input checksums.
uv run nve-dataset validate datasets/small/concentrated/seed_001
uv run nve-dataset reproduce-check datasets/small/concentrated/seed_001

# Summarize the inputs, then replay and compare policy variants.
uv run nve-dataset aggregate datasets/small/*/seed_* --output datasets/small/summary_all.csv
uv run nve-policy replay datasets/small/*/seed_* --config policies.yaml
uv run nve-policy compare datasets/small/*/seed_* --output datasets/small/policy_comparison.csv

uv run pytest
```

The demonstration suite is smaller than the paper's experiment sweeps.

## Documentation

| Document | Contents |
|---|---|
| [Reproducing the experiments](REPRODUCING.md) | Commands, seeds, statistics, and output mappings. |
| [Dataset design](docs/DATASET_DESIGN.md) | Scenarios, generation rules, assumptions, and validation. |
| [CSV schemas](docs/CSV_SCHEMA.md) | Input columns, units, and sampling conventions. |
| [Policy evaluation](docs/POLICY_EVALUATION.md) | Replay ordering, policies, cost model, limitations, and output fields. |

## License

Code is distributed under the [MIT License](LICENSE); generated datasets use
[CC BY 4.0](datasets/LICENSE).
