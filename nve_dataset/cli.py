from __future__ import annotations

import argparse
import json
from pathlib import Path

from nve_dataset.analysis import aggregate_summaries, generate_summary, generate_visualizations
from nve_dataset.config import config_for_run, load_config
from nve_dataset.generator import generate_dataset, generate_suite
from nve_dataset.validation import check_reproducibility, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2P NVE synthetic interaction dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)
    suite = sub.add_parser("generate-suite", help="generate every scenario/seed in config suite")
    suite.add_argument("--config", default="config.yaml")
    suite.add_argument("--no-plots", action="store_true")
    one = sub.add_parser("generate", help="generate one dataset")
    one.add_argument("--config", default="config.yaml")
    one.add_argument("--scenario", required=True, choices=["uniform", "concentrated", "shifting", "burst"])
    one.add_argument("--seed", required=True, type=int)
    one.add_argument("--output")
    validate = sub.add_parser("validate", help="validate one generated dataset")
    validate.add_argument("dataset")
    reproduce = sub.add_parser("reproduce-check", help="regenerate in a temporary directory and compare hashes")
    reproduce.add_argument("dataset")
    summary = sub.add_parser("summarize", help="regenerate summary statistics")
    summary.add_argument("dataset")
    plots = sub.add_parser("visualize", help="regenerate validation plots")
    plots.add_argument("dataset")
    aggregate = sub.add_parser("aggregate", help="collapse dataset summaries into one CSV table")
    aggregate.add_argument("datasets", nargs="+")
    aggregate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-suite":
        paths = generate_suite(load_config(args.config), analyze=not args.no_plots)
        print(f"generated {len(paths)} datasets")
        for path in paths:
            print(path)
    elif args.command == "generate":
        config = config_for_run(load_config(args.config), args.scenario, args.seed)
        print(generate_dataset(config, args.output))
    elif args.command == "validate":
        report = validate_dataset(args.dataset)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["valid"] else 1)
    elif args.command == "reproduce-check":
        report = check_reproducibility(args.dataset)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["reproducible"] else 1)
    elif args.command == "summarize":
        print(json.dumps(generate_summary(args.dataset), indent=2))
    elif args.command == "visualize":
        for path in generate_visualizations(args.dataset):
            print(path)
    elif args.command == "aggregate":
        print(aggregate_summaries(args.datasets, args.output))
