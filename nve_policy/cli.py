from __future__ import annotations

import argparse

from nve_policy.runner import compare, replay_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authority placement policy evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("replay", help="replay datasets under every configured policy")
    run.add_argument("datasets", nargs="+")
    run.add_argument("--config", default="policies.yaml")
    table = sub.add_parser("compare", help="collect policy metrics into one CSV table")
    table.add_argument("datasets", nargs="+")
    table.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "replay":
        results = replay_all(args.datasets, args.config)
        print(f"replayed {len(results)} dataset/policy combinations")
    elif args.command == "compare":
        print(compare(args.datasets, args.output))
