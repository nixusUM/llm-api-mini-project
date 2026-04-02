"""Goal-driven assistant for real project file operations."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "pipeline_outputs"
TOOLS_DOC_PATH = ROOT / "docs" / "MCP_TOOLS_INDEX.md"
USAGE_REPORT_PATH = OUT_DIR / "file_ops_usage_report.md"
DIFF_REPORT_PATH = OUT_DIR / "file_ops_diff_summary.md"

SCAN_SUFFIXES = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".js", ".ts")
SKIP_PARTS = {".git", ".venv", "__pycache__", "models", "document_indices"}


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(ROOT))


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        files.append(p)
    return files


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "").strip()
    return (proc.stdout or "").strip()


def find_symbol_usages(symbol: str, max_hits: int = 220) -> dict[str, Any]:
    query = symbol.strip()
    if not query:
        return {"ok": False, "error": "symbol is empty"}
    matches: list[dict[str, Any]] = []
    for path in _iter_files():
        rel = str(path.relative_to(ROOT))
        text = _safe_read(path)
        if not text:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if query not in line:
                continue
            matches.append({"path": rel, "line": idx, "text": line.strip()[:180]})
            if len(matches) >= max_hits:
                break
        if len(matches) >= max_hits:
            break
    by_file = {}
    for item in matches:
        by_file[item["path"]] = by_file.get(item["path"], 0) + 1
    top_files = sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:20]
    lines = [
        "# File Ops Usage Report",
        "",
        f"- symbol: `{query}`",
        f"- matches: **{len(matches)}**",
        "",
        "## Top files",
    ]
    lines.extend([f"- `{path}`: {count}" for path, count in top_files] or ["- no matches"])
    lines.append("")
    lines.append("## Matches")
    for item in matches[:120]:
        lines.append(f"- `{item['path']}:{item['line']}` — {item['text']}")
    out = _write(USAGE_REPORT_PATH, "\n".join(lines) + "\n")
    return {"ok": True, "goal": "find_usages", "symbol": query, "matches": len(matches), "output_file": out}


def _mcp_tools_from_ast(py_path: Path) -> list[tuple[str, str]]:
    src = _safe_read(py_path)
    if not src:
        return []
    tree = ast.parse(src)
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        has_mcp = False
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                if getattr(d.func.value, "id", "") == "mcp" and d.func.attr == "tool":
                    has_mcp = True
            if isinstance(d, ast.Attribute):
                if getattr(d.value, "id", "") == "mcp" and d.attr == "tool":
                    has_mcp = True
        if not has_mcp:
            continue
        doc = ast.get_docstring(node) or ""
        first = doc.strip().splitlines()[0] if doc.strip() else "-"
        out.append((node.name, first))
    return out


def update_docs_from_code() -> dict[str, Any]:
    mcp_file = ROOT / "mcp_local_server.py"
    tools = _mcp_tools_from_ast(mcp_file)
    tools.sort(key=lambda x: x[0])
    lines = [
        "# MCP Tools Index (Auto-generated)",
        "",
        "Этот файл обновляется ассистентом операций с файлами.",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]
    for name, descr in tools:
        clean = descr.replace("|", "\\|")
        lines.append(f"| `{name}` | {clean} |")
    changed = _git("diff", "--name-only")
    lines += ["", "## Working tree snapshot", ""]
    for row in [x.strip() for x in changed.splitlines() if x.strip()][:50]:
        lines.append(f"- `{row}`")
    out = _write(TOOLS_DOC_PATH, "\n".join(lines) + "\n")
    return {"ok": True, "goal": "update_docs", "tools_count": len(tools), "output_file": out}


def prepare_diff_summary() -> dict[str, Any]:
    stat = _git("diff", "--stat")
    names = _git("diff", "--name-only")
    patch_head = _git("diff")
    touched = [x.strip() for x in names.splitlines() if x.strip()]
    lines = [
        "# File Ops Diff Summary",
        "",
        "## Files changed",
    ]
    lines.extend([f"- `{f}`" for f in touched] or ["- (no unstaged changes)"])
    lines += ["", "## git diff --stat", "", "```", stat or "(empty)", "```", "", "## Diff preview", "", "```"]
    lines.append((patch_head[:3500] or "(empty)").strip())
    lines += ["```", ""]
    out = _write(DIFF_REPORT_PATH, "\n".join(lines))
    return {"ok": True, "goal": "prepare_diff", "changed_files": len(touched), "output_file": out}


def check_project_invariants() -> dict[str, Any]:
    violations: list[str] = []
    for path in _iter_files():
        if path.suffix != ".py":
            continue
        rel = str(path.relative_to(ROOT))
        text = _safe_read(path)
        if "TODO" in text:
            violations.append(f"{rel}: contains TODO")
        for i, line in enumerate(text.splitlines(), start=1):
            if len(line) > 140:
                violations.append(f"{rel}:{i} line > 140 chars")
                break
    payload = {
        "ok": True,
        "goal": "check_invariants",
        "violations_count": len(violations),
        "violations": violations[:120],
    }
    out = _write(OUT_DIR / "file_ops_invariants_report.json", json.dumps(payload, ensure_ascii=False, indent=2))
    payload["output_file"] = out
    return payload


def run_file_ops_goal(goal: str, query: str = "") -> dict[str, Any]:
    g = goal.strip().lower()
    if g == "find_usages":
        return find_symbol_usages(query or "get_support_ticket_context")
    if g == "update_docs":
        return update_docs_from_code()
    if g == "prepare_diff":
        return prepare_diff_summary()
    if g == "check_invariants":
        return check_project_invariants()
    return {"ok": False, "error": f"unknown goal: {goal}"}
