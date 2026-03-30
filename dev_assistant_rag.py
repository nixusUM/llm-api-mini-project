"""RAG over project documentation (README, docs/, requirements) for /help."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_indexer import (
    DocumentLoader,
    Embedder,
    FixedSizeChunker,
    IndexerPipeline,
    JSONStorage,
)
from document_indexer.corpus import collect_dev_assistant_paths, relative_path
from project_context import get_git_branch, get_git_diff_stat, list_tracked_files

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = PROJECT_ROOT / "document_indices" / "index_dev_assistant.json"

# Локальные OpenAI-совместимые серверы часто отбрасывают role=system; Qwen по умолчанию тянет в китайский шаблон.
# Дублируем правила в user-тексте через build_dev_assistant_local_llm_prompt().
LLM_SYSTEM_DEV_ASSISTANT = (
    "You are the developer assistant for the repository described in the user message "
    "(LLM API mini project).\n"
    "Hard rules:\n"
    "1) Answer ONLY in Russian (Cyrillic). Do not use Chinese, English, or other languages.\n"
    "2) Use only facts from the sections about repository context and documentation excerpts "
    "in the user message. No generic chatbot behavior.\n"
    "3) Do not ask what kind of help the user wants; do not list unrelated topics. "
    "Answer the question about this project directly and briefly.\n"
    "4) If the excerpts do not contain the answer, say so in Russian.\n"
    # Явная подсказка для Qwen/китайских весов: мета-инструкция на 中文 сильнее удерживает русский ответ.
    "5) 必须用俄语（西里尔字母）回答；禁止用中文或英文作答；不要写「当然可以」或泛泛的「需要什么帮助」。\n"
)


@dataclass
class _Snippet:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    vector_score: float
    final_score: float = 0.0


class DevAssistantRAG:
    def __init__(self) -> None:
        self._pipeline: IndexerPipeline | None = None
        self._source_map: dict[str, str] = {}
        self._index_path = Path(
            os.getenv("DEV_ASSISTANT_INDEX_PATH", str(DEFAULT_INDEX))
        ).resolve()

    def ensure_index_built(self) -> Path:
        """Load or create on-disk index; returns path to JSON index."""
        self._ensure_pipeline()
        return self._index_path

    def _ensure_pipeline(self) -> None:
        if self._pipeline:
            return
        chunker = FixedSizeChunker(chunk_size=420, overlap=80)
        embedder = Embedder(api_key="LOCAL_ONLY", force_mock=True)
        storage = JSONStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        if self._index_path.exists():
            pipeline.load(str(self._index_path))
            self._pipeline = pipeline
            return
        self._build_and_save(pipeline)

    def _build_and_save(self, pipeline: IndexerPipeline) -> None:
        loader = DocumentLoader()
        paths = collect_dev_assistant_paths()
        documents = []
        self._source_map = {}
        for path in paths:
            doc = loader.load_file(str(path))
            if doc:
                documents.append(doc)
                self._source_map[path.name] = relative_path(path)
        pipeline.index_documents(documents)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save(str(self._index_path))
        self._pipeline = pipeline

    def _normalize_score(self, raw: float) -> float:
        normalized = (float(raw) + 1.0) / 2.0
        return max(0.0, min(normalized, 1.0))

    def _retrieve(self, query: str, top_k: int) -> list[_Snippet]:
        self._ensure_pipeline()
        assert self._pipeline is not None
        results = self._pipeline.search(query, top_k=top_k)
        out: list[_Snippet] = []
        for entry, raw_score in results:
            vec = self._normalize_score(raw_score)
            src = self._source_map.get(entry.source, entry.source)
            out.append(
                _Snippet(
                    chunk_id=entry.chunk_id,
                    text=entry.text.strip(),
                    source=src,
                    title=entry.title,
                    section=entry.section,
                    vector_score=vec,
                    final_score=vec,
                )
            )
        return out

    def _rerank(self, question: str, candidates: list[_Snippet]) -> list[_Snippet]:
        q_tokens = {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", question.lower())}
        ranked: list[_Snippet] = []
        for chunk in candidates:
            text_tokens = {
                w
                for w in re.findall(
                    r"[a-zA-Zа-яА-Я0-9_]{3,}",
                    f"{chunk.title} {chunk.section} {chunk.text}".lower(),
                )
            }
            overlap = len(q_tokens & text_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
            final = max(0.0, min(1.0, 0.62 * chunk.vector_score + 0.38 * overlap))
            ranked.append(
                _Snippet(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    title=chunk.title,
                    section=chunk.section,
                    vector_score=chunk.vector_score,
                    final_score=final,
                )
            )
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    def build_context_prompt(self, question: str, project_context: str, top_k: int = 6) -> str:
        raw = self._retrieve(question, max(12, top_k * 2))
        chunks = self._rerank(question, raw)[:top_k]
        if not chunks:
            doc_block = "(В индексе нет фрагментов — проверьте README и папку docs.)"
        else:
            doc_block = "\n\n".join(
                f"[{i + 1}] {c.source} / {c.section}:\n{c.text}"
                for i, c in enumerate(chunks)
            )
        return (
            "[Задача] Ответь ТОЛЬКО по-русски, по этому репозиторию. Без китайского и без английского.\n"
            "Ты ассистент разработчика этого репозитория (LLM API mini project).\n"
            "Отвечай по фрагментам документации ниже и по контексту репозитория.\n"
            "Если в фрагментах нет ответа, честно скажи и опирайся только на контекст git/структуры.\n"
            "Не задавай общих вопросов «чем помочь» — дай прямой ответ на вопрос.\n"
            "Запрещено начинать ответ с китайских фраз (当然可以, 请告诉我, 你需要) и перечислять нерелевантные темы.\n\n"
            f"Контекст репозитория (MCP/git):\n{project_context}\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Фрагменты документации:\n{doc_block}\n\n"
            "Пиши ответ сразу на русском (только кириллица для основного текста), без приветствий на других языках:\n"
        )


def build_project_context_block(ctx: dict[str, Any]) -> str:
    lines = [str(ctx.get("git", ""))]
    files = ctx.get("files_preview")
    if files:
        lines.append("Примеры файлов (git ls-files):")
        lines.extend(f"  - {f}" for f in files[:25])
    diff_preview = ctx.get("diff_stat")
    if diff_preview:
        lines.append("git diff --stat (если есть изменения):")
        lines.append(diff_preview)
    src = ctx.get("context_source")
    if src:
        lines.append(f"(источник контекста: {src})")
    return "\n".join(lines)


def gather_dev_assistant_context() -> dict[str, Any]:
    """Repository context via local git (same data as MCP tools in this project)."""
    g = get_git_branch()
    files = list_tracked_files(max_files=40)
    diff = get_git_diff_stat(max_lines=14)
    git_line = (
        f"branch={g.get('branch')} commit={g.get('commit_short', '')}"
        if g.get("ok")
        else f"git_error: {g.get('error', '')}"
    )
    file_list = list(files.get("files", [])) if files.get("ok") else []
    diff_stat = str(diff.get("diff_stat", "")) if diff.get("ok") else ""
    return {
        "git": git_line,
        "files_preview": file_list,
        "diff_stat": diff_stat,
        "context_source": "git subprocess (same logic as MCP)",
    }


def _coerce_mcp_dict(tool_result: dict[str, Any]) -> dict[str, Any]:
    structured = tool_result.get("structured")
    if isinstance(structured, dict) and structured:
        return structured
    text = str(tool_result.get("text", "")).strip()
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def gather_dev_assistant_context_via_mcp() -> dict[str, Any]:
    """Load branch / file list / diff through MCP stdio server (for demos)."""
    from mcp_list_tools import call_mcp_tool_sync

    g = _coerce_mcp_dict(call_mcp_tool_sync("get_current_git_branch", {}))
    files_raw = _coerce_mcp_dict(
        call_mcp_tool_sync("list_project_tracked_files", {"max_files": 40})
    )
    diff_raw = _coerce_mcp_dict(
        call_mcp_tool_sync("get_working_tree_diff_stat", {"max_lines": 14})
    )
    git_line = (
        f"branch={g.get('branch')} commit={g.get('commit_short', '')}"
        if g.get("ok")
        else f"mcp_git_error: {g or 'empty'}"
    )
    file_list = list(files_raw.get("files", [])) if files_raw.get("ok") else []
    diff_stat = str(diff_raw.get("diff_stat", "")) if diff_raw.get("ok") else ""
    return {
        "git": git_line,
        "files_preview": file_list,
        "diff_stat": diff_stat,
        "context_source": "MCP (get_current_git_branch, list_project_tracked_files, get_working_tree_diff_stat)",
    }


def _mcp_project_context_ok(ctx: dict[str, Any]) -> bool:
    git = str(ctx.get("git", "")).strip()
    return bool(git) and not git.startswith("mcp_git_error")


def gather_dev_assistant_context_mcp_or_git() -> dict[str, Any]:
    """Prefer MCP (assignment); fall back to git subprocess if MCP is unavailable."""
    try:
        mcp_ctx = gather_dev_assistant_context_via_mcp()
        if _mcp_project_context_ok(mcp_ctx):
            mcp_ctx["context_source"] = (
                "MCP (get_current_git_branch, list_project_tracked_files, "
                "get_working_tree_diff_stat)"
            )
            return mcp_ctx
    except Exception:
        pass
    return gather_dev_assistant_context()


def build_dev_assistant_prompt(
    question: str,
    *,
    use_mcp_context: bool | None = False,
) -> tuple[str, dict[str, Any]]:
    if use_mcp_context is True:
        ctx = gather_dev_assistant_context_via_mcp()
    elif use_mcp_context is False:
        ctx = gather_dev_assistant_context()
    else:
        ctx = gather_dev_assistant_context_mcp_or_git()
    block = build_project_context_block(ctx)
    rag = get_dev_assistant_rag()
    prompt = rag.build_context_prompt(question, block)
    return prompt, ctx


def build_dev_assistant_local_llm_prompt(
    question: str,
    *,
    use_mcp_context: bool | None = False,
) -> tuple[str, dict[str, Any]]:
    """Промпт для локальной LLM: правила внутри user, т.к. часть бэкендов игнорирует role=system."""
    body, ctx = build_dev_assistant_prompt(question, use_mcp_context=use_mcp_context)
    combined = f"{LLM_SYSTEM_DEV_ASSISTANT.strip()}\n\n---\n\n{body}"
    return combined, ctx


_singleton: DevAssistantRAG | None = None


def get_dev_assistant_rag() -> DevAssistantRAG:
    global _singleton
    if _singleton is None:
        _singleton = DevAssistantRAG()
    return _singleton
