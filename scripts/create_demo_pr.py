#!/usr/bin/env python3
"""Create demo change branch and open PR automatically."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_FILE = ROOT / "docs" / "API.md"


def run(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "git command failed").strip()
        raise RuntimeError(msg)
    return (proc.stdout or "").strip()


def run_gh(*args: str) -> str:
    proc = subprocess.run(
        ["gh", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "gh command failed").strip()
        raise RuntimeError(msg)
    return (proc.stdout or "").strip()


def current_branch() -> str:
    return run("rev-parse", "--abbrev-ref", "HEAD")


def append_demo_line() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"\nDemo note for AI review run at {now}\n"
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


def create_pr(branch: str, base: str) -> str:
    title = "Demo: trigger tg review"
    body = "Demo PR to test Telegram /review_pr flow."
    return run_gh(
        "pr",
        "create",
        "--repo",
        "nixusUM/llm-api-mini-project",
        "--base",
        base,
        "--head",
        branch,
        "--title",
        title,
        "--body",
        body,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create demo PR automatically")
    p.add_argument("--base", default="main", help="Base branch for PR")
    p.add_argument("--prefix", default="demo/tg-review-auto", help="Demo branch prefix")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_branch = current_branch()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"{args.prefix}-{stamp}"
    run("checkout", "-b", branch)
    append_demo_line()
    run("add", str(TARGET_FILE.relative_to(ROOT)))
    run("commit", "-m", "demo: trigger tg review")
    run("push", "-u", "origin", branch)
    pr_url = create_pr(branch=branch, base=args.base)
    print("Created PR:", pr_url)
    print("Demo branch:", branch)
    print("Started from branch:", start_branch)


if __name__ == "__main__":
    main()
