"""RAG helper used by the main UI to compare augmented and non-augmented answers."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic_client import ask_claude_with_meta

from document_indexer import (
    DocumentLoader,
    Embedder,
    FixedSizeChunker,
    FAISSStorage,
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
    score: float


class RAGService:
    def __init__(self):
        self._pipeline: Optional[IndexerPipeline] = None
        self.model_id = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.35"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "500"))
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))
        self._questions = self._load_control_questions()
        self._source_map: Dict[str, str] = {}

    def control_questions(self) -> List[Dict[str, Any]]:
        return self._questions

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if not question:
            return {"error": "Question is empty."}
        top_k = top_k or self.top_k
        retrieved = self._retrieve(question, top_k=top_k)
        no_rag = self._call_llm(question, rag_mode=False)
        context_prompt = self._build_context_prompt(question, retrieved)
        rag_resp = self._call_llm(context_prompt, rag_mode=True)
        rag_text = rag_resp["text"]
        if retrieved and self._needs_rag_rewrite(rag_text):
            rag_text = self._rewrite_rag_answer(question, retrieved, rag_text)

        sources = sorted({chunk.source for chunk in retrieved})
        return {
            "question": question,
            "no_rag": {
                "text": no_rag["text"],
                "model": no_rag["model"],
                "tokens": no_rag["total_tokens"],
            },
            "rag": {
                "text": rag_text,
                "model": rag_resp["model"],
                "tokens": rag_resp["total_tokens"],
                "chunks": [self._chunk_to_dict(chunk) for chunk in retrieved],
                "sources": sources,
            },
        }

    def _chunk_to_dict(self, chunk: ChunkSnippet) -> Dict[str, Any]:
        text = chunk.text
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + "…"
        return {
            "chunk_id": chunk.chunk_id,
            "text": text,
            "source": chunk.source,
            "title": chunk.title,
            "section": chunk.section,
            "score": round(chunk.score, 4),
        }

    def _retrieve(self, question: str, top_k: int) -> List[ChunkSnippet]:
        self._ensure_indexed()
        results = self._pipeline.search(question, top_k=top_k)
        snippets: List[ChunkSnippet] = []
        for entry, score in results:
            snippets.append(
                ChunkSnippet(
                    chunk_id=entry.chunk_id,
                    text=entry.text.strip(),
                    source=self._source_map.get(entry.source, entry.source),
                    title=entry.title,
                    section=entry.section,
                    score=score,
                )
            )
        return snippets

    def _build_context_prompt(self, question: str, chunks: List[ChunkSnippet]) -> str:
        if not chunks:
            return question
        context = "\n\n".join(
            f"[{idx + 1}] {chunk.source} / {chunk.section}:\n{chunk.text}"
            for idx, chunk in enumerate(chunks)
        )
        return (
            "Ты отвечаешь по базе знаний мобильной разработки.\n"
            "Правила:\n"
            "1) Используй факты из фрагментов как основной контекст.\n"
            "2) Не пиши «нет информации», если из фрагментов можно дать частичный или полный ответ.\n"
            "3) Если информации недостаточно, явно укажи, какая часть доступна во фрагментах.\n"
            "4) В конце добавь блок «Источники» с source/section.\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Фрагменты:\n{context}\n\n"
            "Дай краткий и практичный ответ."
        )

    def _call_llm(self, user_message: str, rag_mode: bool = False) -> Dict[str, Any]:
        # Pure LLM call (no saved branch history/invariants) for honest RAG-vs-noRAG comparison.
        system_instruction = None
        if rag_mode:
            system_instruction = (
                "Ты RAG-ассистент. Если в предоставленных фрагментах есть релевантные факты, "
                "используй их и не давай уклончивых ответов."
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
            "text": text,
            "model": used_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _needs_rag_rewrite(self, text: str) -> bool:
        low = text.lower()
        markers = [
            "к сожалению",
            "нет информации",
            "не наш",
            "не найден",
            "недостаточно данных",
        ]
        return any(marker in low for marker in markers)

    def _rewrite_rag_answer(
        self, question: str, chunks: List[ChunkSnippet], current_answer: str
    ) -> str:
        context = "\n\n".join(
            f"[{idx + 1}] {chunk.source} / {chunk.section}:\n{chunk.text}"
            for idx, chunk in enumerate(chunks)
        )
        prompt = (
            f"Перепиши ответ без фраз вроде 'к сожалению' и 'нет информации'.\n"
            f"Используй только факты из фрагментов.\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Текущий ответ:\n{current_answer}\n\n"
            f"Фрагменты:\n{context}\n\n"
            "Новый ответ (коротко и по делу, с блоком 'Источники'):"
        )
        rewritten = self._call_llm(prompt, rag_mode=True)["text"]
        return self._sanitize_rag_text(rewritten)

    def _sanitize_rag_text(self, text: str) -> str:
        cleaned = text
        for token in ("К сожалению,", "к сожалению,", "К сожалению", "к сожалению"):
            cleaned = cleaned.replace(token, "")
        return cleaned.strip()

    def _ensure_indexed(self):
        if self._pipeline:
            return
        chunker = FixedSizeChunker(chunk_size=400, overlap=60)
        embedder = Embedder(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        storage = FAISSStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        loader = DocumentLoader()
        target_sources = self._target_mobile_sources()
        documents = [
            doc
            for path in collect_corpus_paths()
            if self._allow_path(path, target_sources) and (doc := loader.load_file(str(path)))
        ]
        self._source_map = {
            path.name: relative_path(path) for path in collect_corpus_paths() if self._allow_path(path, target_sources)
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
