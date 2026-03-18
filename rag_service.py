"""RAG helper for baseline vs improved retrieval comparison."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic_client import ask_claude_with_meta

from document_indexer import (
    DocumentLoader,
    Embedder,
    FAISSStorage,
    FixedSizeChunker,
    IndexerPipeline,
)
from document_indexer.corpus import collect_corpus_paths, relative_path

CONTROL_QUESTIONS_FILE = Path(__file__).resolve().parent / "rag_control_questions.json"


@dataclass
class ChunkSnippet:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    vector_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0


class RAGService:
    def __init__(self):
        self._pipeline: Optional[IndexerPipeline] = None
        self.model_id = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.35"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "500"))
        self.default_top_k_before = int(os.getenv("RAG_TOP_K_BEFORE", "8"))
        self.default_top_k_after = int(os.getenv("RAG_TOP_K_AFTER", "4"))
        self.default_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.52"))
        self._questions = self._load_control_questions()
        self._source_map: Dict[str, str] = {}

    def control_questions(self) -> List[Dict[str, Any]]:
        return self._questions

    def answer_question(
        self,
        question: str,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
        enable_query_rewrite: bool = True,
    ) -> Dict[str, Any]:
        if not question:
            return {"error": "Question is empty."}

        top_before = max(1, top_k_before or self.default_top_k_before)
        top_after = max(1, top_k_after or self.default_top_k_after)
        threshold_value = threshold if threshold is not None else self.default_threshold
        threshold_value = max(0.0, min(threshold_value, 1.0))

        baseline_chunks = self._retrieve_raw(question, top_before)
        baseline_final = baseline_chunks[:top_after]

        rewritten_query = self._rewrite_query(question) if enable_query_rewrite else question
        improved_candidates = self._retrieve_raw(rewritten_query, top_before)
        reranked = self._rerank(question, improved_candidates)
        filtered = [item for item in reranked if item.final_score >= threshold_value]
        improved_final = filtered[:top_after]
        if not improved_final and reranked:
            improved_final = reranked[: min(top_after, len(reranked))]

        baseline_prompt = self._build_context_prompt(question, baseline_final)
        improved_prompt = self._build_context_prompt(question, improved_final)
        baseline_answer = self._call_llm(baseline_prompt, rag_mode=True)
        improved_answer = self._call_llm(improved_prompt, rag_mode=True)

        return {
            "question": question,
            "settings": {
                "threshold": round(threshold_value, 3),
                "top_k_before": top_before,
                "top_k_after": top_after,
                "rewrite_enabled": bool(enable_query_rewrite),
            },
            "query_rewrite": {
                "original": question,
                "rewritten": rewritten_query,
                "changed": rewritten_query != question,
            },
            "baseline": {
                "label": "Без rewrite/filter",
                "text": baseline_answer["text"],
                "model": baseline_answer["model"],
                "tokens": baseline_answer["total_tokens"],
                "chunks_before": [self._chunk_to_dict(c) for c in baseline_chunks],
                "chunks_after": [self._chunk_to_dict(c) for c in baseline_final],
                "sources": sorted({c.source for c in baseline_final}),
            },
            "improved": {
                "label": "С rewrite + rerank/filter",
                "text": improved_answer["text"],
                "model": improved_answer["model"],
                "tokens": improved_answer["total_tokens"],
                "chunks_before": [self._chunk_to_dict(c) for c in improved_candidates],
                "chunks_after": [self._chunk_to_dict(c) for c in improved_final],
                "sources": sorted({c.source for c in improved_final}),
                "filtered_out": max(0, len(reranked) - len(improved_final)),
            },
        }

    def _chunk_to_dict(self, chunk: ChunkSnippet) -> Dict[str, Any]:
        text = chunk.text
        if len(text) > 380:
            text = text[:380].rsplit(" ", 1)[0] + "…"
        return {
            "chunk_id": chunk.chunk_id,
            "text": text,
            "source": chunk.source,
            "title": chunk.title,
            "section": chunk.section,
            "vector_score": round(chunk.vector_score, 4),
            "rerank_score": round(chunk.rerank_score, 4),
            "final_score": round(chunk.final_score, 4),
        }

    def _retrieve_raw(self, query: str, top_k: int) -> List[ChunkSnippet]:
        self._ensure_indexed()
        results = self._pipeline.search(query, top_k=top_k)
        snippets: List[ChunkSnippet] = []
        for entry, raw_score in results:
            vector_score = self._normalize_vector_score(raw_score)
            snippets.append(
                ChunkSnippet(
                    chunk_id=entry.chunk_id,
                    text=entry.text.strip(),
                    source=self._source_map.get(entry.source, entry.source),
                    title=entry.title,
                    section=entry.section,
                    vector_score=vector_score,
                    rerank_score=0.0,
                    final_score=vector_score,
                )
            )
        return snippets

    def _rerank(self, question: str, candidates: List[ChunkSnippet]) -> List[ChunkSnippet]:
        query_tokens = self._tokenize(question)
        reranked: List[ChunkSnippet] = []
        for chunk in candidates:
            text_tokens = self._tokenize(f"{chunk.title} {chunk.section} {chunk.text}")
            overlap = self._token_overlap(query_tokens, text_tokens)
            bonus = self._source_bonus(question, chunk.source)
            final = max(0.0, min(1.0, 0.62 * chunk.vector_score + 0.33 * overlap + bonus))
            reranked.append(
                ChunkSnippet(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    title=chunk.title,
                    section=chunk.section,
                    vector_score=chunk.vector_score,
                    rerank_score=overlap,
                    final_score=final,
                )
            )
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked

    def _source_bonus(self, question: str, source: str) -> float:
        q = question.lower()
        s = source.lower()
        bonus = 0.0
        if "compose" in q and "compose" in s:
            bonus += 0.08
        if "kotlin" in q and "kotlin" in s:
            bonus += 0.08
        if ("архитектур" in q or "ci/cd" in q or "релиз" in q) and "architecture" in s:
            bonus += 0.06
        return bonus

    def _build_context_prompt(self, question: str, chunks: List[ChunkSnippet]) -> str:
        if not chunks:
            return question
        context = "\n\n".join(
            f"[{idx + 1}] {chunk.source} / {chunk.section}:\n{chunk.text}"
            for idx, chunk in enumerate(chunks)
        )
        return (
            "Ты отвечаешь по базе знаний мобильной разработки.\n"
            "Используй факты из предоставленных фрагментов.\n"
            "Если полного ответа нет, дай максимально полезный частичный ответ по фрагментам.\n"
            "В конце добавь блок «Источники» с source/section.\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Фрагменты:\n{context}\n\n"
            "Ответ:"
        )

    def _rewrite_query(self, question: str) -> str:
        q = question.strip()
        if not q:
            return q
        q_low = q.lower()
        additions: List[str] = []
        if "nullable" in q_low or "null" in q_low:
            additions.append("kotlin null safety safe call elvis operator")
        if "compose" in q_low:
            additions.append("compose multiplatform recomposition remember key")
        if "multiplatform" in q_low or "mpp" in q_low:
            additions.append("kotlin multiplatform expect actual shared module")
        if "архитектур" in q_low or "architecture" in q_low:
            additions.append("mobile clean architecture mvvm mvi data domain ui")
        if "релиз" in q_low or "ci/cd" in q_low:
            additions.append("mobile release pipeline staged rollout testflight")
        if not additions:
            return q
        return f"{q}. Контекст: {'; '.join(additions)}"

    def _call_llm(self, user_message: str, rag_mode: bool = False) -> Dict[str, Any]:
        system_instruction = None
        if rag_mode:
            system_instruction = (
                "Ты RAG-ассистент. Если в фрагментах есть релевантные факты, "
                "используй их и не отвечай расплывчато."
            )
        text, used_model, usage = ask_claude_with_meta(
            prompt=user_message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model_override=self.model_id,
            system_instruction=system_instruction,
        )
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        return {
            "text": text.strip(),
            "model": used_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _ensure_indexed(self):
        if self._pipeline:
            return
        chunker = FixedSizeChunker(chunk_size=420, overlap=80)
        embedder = Embedder(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        storage = FAISSStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        loader = DocumentLoader()
        target_sources = self._target_mobile_sources()
        all_paths = collect_corpus_paths()
        documents = [
            doc
            for path in all_paths
            if self._allow_path(path, target_sources) and (doc := loader.load_file(str(path)))
        ]
        self._source_map = {
            path.name: relative_path(path)
            for path in all_paths
            if self._allow_path(path, target_sources)
        }
        pipeline.index_documents(documents)
        self._pipeline = pipeline

    def _target_mobile_sources(self) -> set[str]:
        values: set[str] = set()
        for row in self._questions:
            sources = row.get("sources", [])
            if isinstance(sources, list):
                for item in sources:
                    if isinstance(item, str) and item.strip():
                        values.add(item.strip())
        return values

    def _allow_path(self, path: Path, target_sources: set[str]) -> bool:
        if not target_sources:
            return True
        return relative_path(path) in target_sources

    def _load_control_questions(self) -> List[Dict[str, Any]]:
        if not CONTROL_QUESTIONS_FILE.exists():
            return []
        try:
            with CONTROL_QUESTIONS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _normalize_vector_score(self, raw: float) -> float:
        normalized = (float(raw) + 1.0) / 2.0
        return max(0.0, min(normalized, 1.0))

    def _tokenize(self, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", text.lower())}

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(len(left), 1)
