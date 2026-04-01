"""Support assistant RAG: FAQ/docs + ticket context via MCP."""

from __future__ import annotations

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
from mcp_list_tools import call_mcp_tool_sync

ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "document_indices" / "index_support_assistant.json"
SUPPORT_DOCS = ("README.md", "docs/SUPPORT_FAQ.md", "docs/API.md")


@dataclass
class _Chunk:
    text: str
    source: str
    section: str
    score: float


class SupportAssistantRAG:
    def __init__(self) -> None:
        self._pipeline: IndexerPipeline | None = None
        self._index_path = Path(
            os.getenv("SUPPORT_ASSISTANT_INDEX_PATH", str(DEFAULT_INDEX))
        ).resolve()
        self._source_map: dict[str, str] = {}

    def ensure_index(self) -> Path:
        self._ensure_pipeline()
        return self._index_path

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        chunker = FixedSizeChunker(chunk_size=420, overlap=80)
        embedder = Embedder(api_key="LOCAL_ONLY", force_mock=True)
        storage = JSONStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        if self._index_path.exists():
            pipeline.load(str(self._index_path))
            self._pipeline = pipeline
            return
        self._build_index(pipeline)

    def _build_index(self, pipeline: IndexerPipeline) -> None:
        loader = DocumentLoader()
        docs = []
        for rel in SUPPORT_DOCS:
            p = ROOT / rel
            if not p.exists():
                continue
            loaded = loader.load_file(str(p))
            if not loaded:
                continue
            docs.append(loaded)
            self._source_map[p.name] = rel
        pipeline.index_documents(docs)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save(str(self._index_path))
        self._pipeline = pipeline

    def _retrieve(self, query: str, top_k: int = 6) -> list[_Chunk]:
        self._ensure_pipeline()
        assert self._pipeline is not None
        out: list[_Chunk] = []
        q_tokens = {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", query.lower())}
        for entry, raw in self._pipeline.search(query, top_k=max(12, top_k * 2)):
            base = max(0.0, min((float(raw) + 1.0) / 2.0, 1.0))
            text = str(entry.text or "").strip()
            t_tokens = {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", text.lower())}
            overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
            score = 0.65 * base + 0.35 * overlap
            source = self._source_map.get(entry.source, entry.source)
            out.append(_Chunk(text=text, source=source, section=str(entry.section), score=score))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]

    def prompt(self, question: str, ticket_ctx: dict[str, Any]) -> str:
        docs = self._retrieve(question, top_k=6)
        docs_block = "\n\n".join(
            f"[{i + 1}] {d.source} / {d.section}\n{d.text}" for i, d in enumerate(docs)
        )
        if not docs_block:
            docs_block = "(Нет релевантных фрагментов в FAQ/docs)"
        ctx = json.dumps(ticket_ctx, ensure_ascii=False, indent=2)
        return (
            "Ты AI-ассистент поддержки пользователей.\n"
            "Отвечай по-русски, коротко и по делу.\n"
            "Используй контекст тикета и FAQ.\n"
            "Если данных недостаточно, напиши какие поля нужны.\n\n"
            f"Контекст тикета/пользователя (MCP JSON):\n{ctx}\n\n"
            f"Вопрос пользователя:\n{question}\n\n"
            f"Фрагменты FAQ/документации:\n{docs_block}\n\n"
            "Формат ответа:\n"
            "1) Краткая причина\n"
            "2) Что проверить пользователю\n"
            "3) Что делает поддержка дальше\n"
            "4) Нужна ли эскалация (да/нет и почему)\n"
        )


def _coerce_struct(res: dict[str, Any]) -> dict[str, Any]:
    structured = res.get("structured")
    if isinstance(structured, dict) and structured:
        return structured
    text = str(res.get("text", "")).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def get_ticket_context_via_mcp(ticket_id: str) -> dict[str, Any]:
    payload = call_mcp_tool_sync("get_support_ticket_context", {"ticket_id": ticket_id})
    data = _coerce_struct(payload)
    if data:
        return data
    return {"ok": False, "error": "empty mcp response", "ticket_id": ticket_id}


def build_support_assistant_local_prompt(ticket_id: str, question: str) -> tuple[str, dict[str, Any]]:
    ctx = get_ticket_context_via_mcp(ticket_id)
    rag = _singleton()
    return rag.prompt(question=question, ticket_ctx=ctx), ctx


_SINGLETON: SupportAssistantRAG | None = None


def _singleton() -> SupportAssistantRAG:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SupportAssistantRAG()
    return _SINGLETON
