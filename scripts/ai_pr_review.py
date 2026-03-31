#!/usr/bin/env python3
"""Automated PR review with diff + lightweight RAG over docs and code."""

from __future__ import annotations

import argparse
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAX_DIFF_CHARS = 36000
MAX_SNIPPET_CHARS = 1400
MAX_CONTEXT_SNIPPETS = 8
MARKER = "<!-- ai-pr-review -->"


@dataclass
class Snippet:
    path: str
    score: float
    text: str


def run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "git failed").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {err}")
    return (proc.stdout or "").strip()


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", text.lower())}


def list_docs_files() -> list[Path]:
    out: list[Path] = []
    readme = ROOT / "README.md"
    if readme.exists():
        out.append(readme)
    docs = ROOT / "docs"
    if docs.exists():
        for p in sorted(docs.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}:
                out.append(p)
    return out


def list_code_files(limit: int = 120) -> list[Path]:
    candidates: list[Path] = []
    for ext in ("*.py", "*.js", "*.ts", "*.tsx"):
        candidates.extend(ROOT.rglob(ext))
    clean = [p for p in sorted(candidates) if ".venv" not in str(p)]
    return clean[:limit]


def chunk_text(text: str, chunk_lines: int = 45) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[str] = []
    for i in range(0, len(lines), chunk_lines):
        part = "\n".join(lines[i : i + chunk_lines]).strip()
        if part:
            chunks.append(part)
    return chunks


def score_chunk(query_tokens: set[str], path_tokens: set[str], chunk: str) -> float:
    text_tokens = tokenize(chunk)
    overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    path_bonus = len(query_tokens & path_tokens) / max(1, len(query_tokens))
    return 0.78 * overlap + 0.22 * path_bonus


def build_rag_context(diff_text: str, changed_files: list[str]) -> list[Snippet]:
    query_tokens = tokenize(diff_text + "\n" + "\n".join(changed_files))
    picked: list[Snippet] = []
    corpora = list_docs_files() + list_code_files()
    for path in corpora:
        rel = str(path.relative_to(ROOT))
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        path_tokens = tokenize(rel.replace("/", " "))
        for chunk in chunk_text(content):
            score = score_chunk(query_tokens, path_tokens, chunk)
            if score <= 0.015:
                continue
            picked.append(Snippet(path=rel, score=score, text=chunk[:MAX_SNIPPET_CHARS]))
    picked.sort(key=lambda s: s.score, reverse=True)
    return picked[:MAX_CONTEXT_SNIPPETS]


def build_prompt(base: str, head: str, changed_files: list[str], diff_text: str, ctx: list[Snippet]) -> str:
    files_block = "\n".join(f"- {f}" for f in changed_files) or "- (none)"
    ctx_block = "\n\n".join(
        f"[{i + 1}] {s.path} (score={s.score:.3f})\n{s.text}" for i, s in enumerate(ctx)
    )
    safe_diff = diff_text[:MAX_DIFF_CHARS]
    if not ctx_block:
        ctx_block = "(RAG context not found)"
    return (
        "Ты senior reviewer. Сделай code review PR.\n"
        "Фокус: потенциальные баги, архитектурные проблемы, рекомендации.\n"
        "Опирайся на diff и RAG-контекст (доки + код).\n"
        "Пиши кратко, но предметно, на русском.\n\n"
        f"PR range: {base}...{head}\n\n"
        f"Изменённые файлы:\n{files_block}\n\n"
        f"Diff:\n{safe_diff}\n\n"
        f"RAG context:\n{ctx_block}\n\n"
        "Верни markdown в формате:\n"
        "## AI Review\n"
        "### Потенциальные баги\n"
        "- ...\n"
        "### Архитектурные проблемы\n"
        "- ...\n"
        "### Рекомендации\n"
        "- ...\n"
        "### Risk level\n"
        "- low|medium|high + обоснование\n"
    )


def fallback_review(changed_files: list[str], diff_text: str) -> str:
    risk = "medium" if ("TODO" in diff_text or "except Exception" in diff_text) else "low"
    files = ", ".join(changed_files[:8]) if changed_files else "нет файлов"
    return (
        "## AI Review\n"
        "### Потенциальные баги\n"
        "- Не удалось вызвать LLM (нет `ANTHROPIC_API_KEY` или ошибка API); проверьте вручную обработку ошибок и пограничные кейсы.\n"
        "### Архитектурные проблемы\n"
        "- Проверьте связность изменений в файлах: "
        f"{files}.\n"
        "### Рекомендации\n"
        "- Добавьте/обновите тесты на изменённый функционал.\n"
        "- Проверьте обратную совместимость API и форматов данных.\n"
        "### Risk level\n"
        f"- {risk}: автоматический fallback-режим без модели.\n"
    )


def build_report(base: str, head: str) -> str:
    changed = run_git("diff", "--name-only", f"{base}...{head}")
    changed_files = [x.strip() for x in changed.splitlines() if x.strip()]
    diff_text = run_git("diff", f"{base}...{head}")
    ctx = build_rag_context(diff_text, changed_files)
    prompt = build_prompt(base, head, changed_files, diff_text, ctx)
    try:
        from anthropic_client import ask_claude_with_meta

        review, model, usage = ask_claude_with_meta(
            prompt=prompt,
            max_tokens=1300,
            temperature=0.1,
        )
        meta = (
            f"{MARKER}\n"
            f"_Model: `{model}` · Input tokens: `{usage.get('input_tokens', 0)}` · "
            f"Output tokens: `{usage.get('output_tokens', 0)}`_\n\n"
        )
        return meta + review.strip() + "\n"
    except Exception:
        return f"{MARKER}\n\n" + fallback_review(changed_files, diff_text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AI PR review text")
    p.add_argument("--base", required=True, help="Base commit SHA")
    p.add_argument("--head", required=True, help="Head commit SHA")
    p.add_argument("--output", default="ai_review.md", help="Output markdown path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    report = build_report(args.base, args.head)
    out_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
