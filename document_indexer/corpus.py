"""Helpers that describe the document corpus used by indexers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def collect_dev_assistant_paths() -> list[Path]:
    """Paths for developer-assistant RAG: README, docs/ (incl. API.md), deps & env hints."""
    candidates: list[Path] = []
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        candidates.append(readme)
    docs_dir = PROJECT_ROOT / "docs"
    if docs_dir.exists():
        for path in sorted(docs_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}:
                candidates.append(path)
    for extra in ("requirements.txt", ".env.example"):
        p = PROJECT_ROOT / extra
        if p.exists():
            candidates.append(p)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def collect_corpus_paths() -> list[Path]:
    """Collect README, articles, code, and PDFs used by RAG components."""
    candidates: list[Path] = []

    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        candidates.append(readme)

    sample_docs = PROJECT_ROOT / "sample_documents"
    if sample_docs.exists():
        for pattern in ("*.md", "*.txt", "*.pdf"):
            candidates.extend(sorted(sample_docs.glob(pattern)))

    for script in sorted(PROJECT_ROOT.glob("*.py")):
        if script.name == "document_indexer_app.py":
            continue
        candidates.append(script)

    indexer_dir = PROJECT_ROOT / "document_indexer"
    if indexer_dir.exists():
        candidates.extend(sorted(indexer_dir.glob("*.py")))

    # Deduplicate while preserving order
    seen = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def relative_path(path: Path) -> str:
    """Return path relative to repo root for UI usage."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def resolve_selected_path(relative_path: str) -> Path | None:
    """Resolve a selected document ensuring it stays inside the repo."""
    try:
        candidate = (PROJECT_ROOT / relative_path).resolve()
    except Exception:
        return None
    if not str(candidate).startswith(str(PROJECT_ROOT.resolve())):
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    return None
