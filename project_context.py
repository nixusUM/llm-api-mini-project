"""Shared project context for MCP tools and the Telegram /help assistant."""

import subprocess
from pathlib import Path

from document_indexer.corpus import PROJECT_ROOT


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    base = cwd or PROJECT_ROOT
    return subprocess.run(
        ["git", *args],
        cwd=str(base),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def get_git_branch(repo_root: str | None = None) -> dict[str, object]:
    """Current git branch and short commit (repository root must be inside project)."""
    root = Path(repo_root).resolve() if repo_root else PROJECT_ROOT
    if not str(root).startswith(str(PROJECT_ROOT.resolve())):
        return {"ok": False, "error": "repo_root outside project"}
    proc = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "git failed").strip()}
    branch = (proc.stdout or "").strip()
    short = _git("rev-parse", "--short", "HEAD", cwd=root)
    commit = (short.stdout or "").strip() if short.returncode == 0 else ""
    return {"ok": True, "branch": branch, "commit_short": commit, "repo_root": str(root)}


def list_tracked_files(
    max_files: int = 100,
    pattern: str | None = None,
    repo_root: str | None = None,
) -> dict[str, object]:
    """List paths tracked by git (optional pathspec filter)."""
    root = Path(repo_root).resolve() if repo_root else PROJECT_ROOT
    if not str(root).startswith(str(PROJECT_ROOT.resolve())):
        return {"ok": False, "error": "repo_root outside project"}
    cap = max(1, min(max_files, 500))
    args = ["ls-files", "-z"]
    if pattern and pattern.strip():
        args.append("--")
        args.append(pattern.strip())
    proc = _git(*args, cwd=root)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "git ls-files failed").strip()}
    raw = proc.stdout or ""
    parts = [p for p in raw.split("\0") if p.strip()]
    truncated = len(parts) > cap
    return {
        "ok": True,
        "files": parts[:cap],
        "total_returned": min(len(parts), cap),
        "truncated": truncated,
    }


def get_git_diff_stat(max_lines: int = 40, repo_root: str | None = None) -> dict[str, object]:
    """Unstaged diff stat (working tree vs index)."""
    root = Path(repo_root).resolve() if repo_root else PROJECT_ROOT
    if not str(root).startswith(str(PROJECT_ROOT.resolve())):
        return {"ok": False, "error": "repo_root outside project"}
    proc = _git("diff", "--stat", cwd=root)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "git diff failed").strip()}
    text = (proc.stdout or "").strip()
    lines = text.splitlines()
    cap = max(5, min(max_lines, 200))
    truncated = len(lines) > cap
    body = "\n".join(lines[:cap])
    return {
        "ok": True,
        "diff_stat": body or "(no unstaged changes)",
        "truncated": truncated,
    }
