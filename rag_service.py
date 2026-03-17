"""RAG helper used by the main UI to compare augmented and non-augmented answers."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_agent import LLMAgent

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
        self.agent = LLMAgent()
        self.model_id = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.35"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "500"))
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))
        self._questions = self._load_control_questions()

    def control_questions(self) -> List[Dict[str, Any]]:
        return self._questions

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if not question:
            return {"error": "Question is empty."}
        top_k = top_k or self.top_k
        retrieved = self._retrieve(question, top_k=top_k)
        no_rag = self._call_llm(question)
        context_prompt = self._build_context_prompt(question, retrieved)
        rag_resp = self._call_llm(context_prompt)

        sources = sorted({chunk.source for chunk in retrieved})
        return {
            "question": question,
            "no_rag": {
                "text": no_rag.text,
                "model": no_rag.used_model,
                "tokens": no_rag.total_tokens,
            },
            "rag": {
                "text": rag_resp.text,
                "model": rag_resp.used_model,
                "tokens": rag_resp.total_tokens,
                "chunks": [self._chunk_to_dict(chunk) for chunk in retrieved],
                "sources": [relative_path(Path(src)) for src in sources],
            },
        }

    def _chunk_to_dict(self, chunk: ChunkSnippet) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
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
            text = entry.text.strip()
            if len(text) > 400:
                text = text[:400].rsplit(" ", 1)[0] + "…"
            snippets.append(
                ChunkSnippet(
                    chunk_id=entry.chunk_id,
                    text=text,
                    source=entry.source,
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
            f"Вопрос:\n{question}\n\n"
            "Ниже — извлечённые чанки из базы знаний. Постарайся "
            "ответить, опираясь на них и упоминая источники:\n\n"
            f"{context}\n\nОтвет:"
        )

    def _call_llm(self, user_message: str):
        return self.agent.run_chat_preview(
            user_message=user_message,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            strategy="sliding",
            window_n=8,
            include_memory_layers=False,
            context_limit_override=200000,
        )

    def _ensure_indexed(self):
        if self._pipeline:
            return
        chunker = FixedSizeChunker(chunk_size=400, overlap=60)
        embedder = Embedder(api_key=os.getenv("OPENAI_API_KEY"), model=self.model_id)
        storage = FAISSStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        loader = DocumentLoader()
        documents = [
            doc
            for path in collect_corpus_paths()
            if (doc := loader.load_file(str(path)))
        ]
        pipeline.index_documents(documents)
        self._pipeline = pipeline

    def _load_control_questions(self) -> List[Dict[str, Any]]:
        if not CONTROL_QUESTIONS_FILE.exists():
            return []
        try:
            with CONTROL_QUESTIONS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
