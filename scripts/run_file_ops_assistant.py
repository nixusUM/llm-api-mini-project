#!/usr/bin/env python3
"""Run goal-based file operations assistant."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from file_ops_assistant import run_file_ops_goal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goal-based file operations assistant")
    p.add_argument(
        "--goal",
        required=True,
        choices=["find_usages", "update_docs", "prepare_diff", "check_invariants"],
    )
    p.add_argument("--query", default="", help="Query/symbol for find_usages")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_file_ops_goal(goal=args.goal, query=args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
