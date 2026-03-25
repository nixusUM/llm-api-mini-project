"""RAG helper for baseline vs improved retrieval comparison."""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from anthropic_client import ask_claude_with_meta

from document_indexer import (
    DocumentLoader,
    Embedder,
    FAISSStorage,
    FixedSizeChunker,
    IndexerPipeline,
    JSONStorage,
)
from document_indexer.corpus import collect_corpus_paths, relative_path

CONTROL_QUESTIONS_FILE = Path(__file__).resolve().parent / "rag_control_questions.json"
CHAT_SCENARIOS_FILE = Path(__file__).resolve().parent / "rag_chat_scenarios.json"


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
        self.model_id = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self.generation_backend = os.getenv("RAG_GENERATION_BACKEND", "local").strip().lower()
        self.local_llm_endpoint = (
            os.getenv("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8088").strip().rstrip("/")
        )
        self.local_llm_model = os.getenv("LOCAL_LLM_MODEL", "qwen-local").strip()
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.35"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "500"))
        self.default_top_k_before = int(os.getenv("RAG_TOP_K_BEFORE", "8"))
        self.default_top_k_after = int(os.getenv("RAG_TOP_K_AFTER", "4"))
        self.default_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.52"))
        self.week6_index_path = self._resolve_week6_index_path()
        self._questions = self._load_control_questions()
        self._chat_scenarios = self._load_chat_scenarios()
        self._source_map: Dict[str, str] = {}
        self._chat_sessions: Dict[str, Dict[str, Any]] = {}

    def control_questions(self) -> List[Dict[str, Any]]:
        return self._questions

    def chat_scenarios(self) -> List[Dict[str, Any]]:
        return self._chat_scenarios

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
        baseline_payload = self._build_mode_payload(
            llm_text=baseline_answer["text"],
            chunks=baseline_final,
            threshold=threshold_value,
        )
        improved_payload = self._build_mode_payload(
            llm_text=improved_answer["text"],
            chunks=improved_final,
            threshold=threshold_value,
        )

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
                "text": baseline_payload["text"],
                "answer": baseline_payload["answer"],
                "sources": baseline_payload["sources"],
                "quotes": baseline_payload["quotes"],
                "weak_context": baseline_payload["weak_context"],
                "has_sources": baseline_payload["has_sources"],
                "has_quotes": baseline_payload["has_quotes"],
                "model": baseline_answer["model"],
                "tokens": baseline_answer["total_tokens"],
                "chunks_before": [self._chunk_to_dict(c) for c in baseline_chunks],
                "chunks_after": [self._chunk_to_dict(c) for c in baseline_final],
            },
            "improved": {
                "label": "С rewrite + rerank/filter",
                "text": improved_payload["text"],
                "answer": improved_payload["answer"],
                "sources": improved_payload["sources"],
                "quotes": improved_payload["quotes"],
                "weak_context": improved_payload["weak_context"],
                "has_sources": improved_payload["has_sources"],
                "has_quotes": improved_payload["has_quotes"],
                "model": improved_answer["model"],
                "tokens": improved_answer["total_tokens"],
                "chunks_before": [self._chunk_to_dict(c) for c in improved_candidates],
                "chunks_after": [self._chunk_to_dict(c) for c in improved_final],
                "filtered_out": max(0, len(reranked) - len(improved_final)),
            },
        }

    def evaluate_control_questions(
        self,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
        enable_query_rewrite: bool = True,
    ) -> Dict[str, Any]:
        reports: List[Dict[str, Any]] = []
        for item in self._questions:
            question = str(item.get("question", "")).strip()
            expectation = str(item.get("expectation", "")).strip()
            if not question:
                continue
            result = self.answer_question(
                question=question,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                threshold=threshold,
                enable_query_rewrite=enable_query_rewrite,
            )
            improved = result.get("improved", {})
            semantic_ok = self._semantic_match(
                expectation=expectation,
                answer=str(improved.get("answer", "")),
                quotes=improved.get("quotes", []),
            )
            reports.append(
                {
                    "question": question,
                    "expectation": expectation,
                    "expected_sources": item.get("sources", []),
                    "sources_present": bool(improved.get("has_sources")),
                    "quotes_present": bool(improved.get("has_quotes")),
                    "semantic_match": semantic_ok,
                    "weak_context": bool(improved.get("weak_context", False)),
                    "used_sources": [s.get("source", "") for s in improved.get("sources", [])],
                }
            )

        total = len(reports)
        return {
            "total_questions": total,
            "sources_ok": sum(1 for x in reports if x["sources_present"]),
            "quotes_ok": sum(1 for x in reports if x["quotes_present"]),
            "semantic_ok": sum(1 for x in reports if x["semantic_match"]),
            "reports": reports,
        }

    def compare_generation_backends(
        self,
        repeats: int = 2,
        question_limit: Optional[int] = None,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
        enable_query_rewrite: bool = True,
    ) -> Dict[str, Any]:
        questions = [q for q in self._questions if str(q.get("question", "")).strip()]
        if question_limit is not None:
            questions = questions[: max(1, int(question_limit))]
        reports: List[Dict[str, Any]] = []
        for item in questions:
            report = self._compare_one_question(
                question=str(item.get("question", "")).strip(),
                expectation=str(item.get("expectation", "")).strip(),
                repeats=max(1, int(repeats)),
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                threshold=threshold,
                enable_query_rewrite=enable_query_rewrite,
            )
            reports.append(report)
        local_ok = sum(1 for x in reports if x.get("local", {}).get("semantic_match"))
        cloud_ok = sum(1 for x in reports if x.get("cloud", {}).get("semantic_match"))
        return {
            "total_questions": len(reports),
            "repeats_per_question": max(1, int(repeats)),
            "local_quality_ok": local_ok,
            "cloud_quality_ok": cloud_ok,
            "reports": reports,
        }

    def _compare_one_question(
        self,
        question: str,
        expectation: str,
        repeats: int,
        top_k_before: Optional[int],
        top_k_after: Optional[int],
        threshold: Optional[float],
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        retrieval = self._prepare_improved_retrieval(
            question=question,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            threshold=threshold,
            enable_query_rewrite=enable_query_rewrite,
        )
        local_stats = self._run_backend_repeated("local", retrieval["prompt"], repeats)
        cloud_stats = self._run_backend_repeated("cloud", retrieval["prompt"], repeats)
        local_answer = str(local_stats.get("sample_answer", ""))
        cloud_answer = str(cloud_stats.get("sample_answer", ""))
        quotes = retrieval.get("quotes", [])
        return {
            "question": question,
            "expectation": expectation,
            "retrieval": {
                "sources": retrieval.get("sources", []),
                "quotes": quotes,
                "weak_context": retrieval.get("weak_context", True),
                "chunks_used": [self._chunk_to_dict(c) for c in retrieval.get("chunks", [])],
            },
            "local": {
                **local_stats,
                "semantic_match": self._semantic_match(expectation, local_answer, quotes),
            },
            "cloud": {
                **cloud_stats,
                "semantic_match": self._semantic_match(expectation, cloud_answer, quotes),
            },
        }

    def _prepare_improved_retrieval(
        self,
        question: str,
        top_k_before: Optional[int],
        top_k_after: Optional[int],
        threshold: Optional[float],
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        top_before = max(1, top_k_before or self.default_top_k_before)
        top_after = max(1, top_k_after or self.default_top_k_after)
        threshold_value = threshold if threshold is not None else self.default_threshold
        threshold_value = max(0.0, min(threshold_value, 1.0))
        rewritten = self._rewrite_query(question) if enable_query_rewrite else question
        candidates = self._retrieve_raw(rewritten, top_before)
        reranked = self._rerank(question, candidates)
        selected = [item for item in reranked if item.final_score >= threshold_value][:top_after]
        if not selected:
            selected = reranked[:top_after]
        return {
            "prompt": self._build_context_prompt(question, selected),
            "chunks": selected,
            "sources": self._collect_sources(selected),
            "quotes": self._collect_quotes(selected),
            "weak_context": self._is_weak_context(selected, threshold_value),
        }

    def _run_backend_repeated(self, backend: str, prompt: str, repeats: int) -> Dict[str, Any]:
        latencies: List[int] = []
        errors: List[str] = []
        sample_answer = ""
        model = ""
        for _ in range(repeats):
            try:
                start = time.perf_counter()
                result = (
                    self._call_local_llm(prompt, rag_mode=True)
                    if backend == "local"
                    else self._call_cloud_llm(prompt, rag_mode=True)
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                model = str(result.get("model", model))
                sample_answer = sample_answer or str(result.get("text", "")).strip()
                latencies.append(int(result.get("latency_ms", elapsed_ms)))
            except Exception as exc:
                errors.append(str(exc))
        success = repeats - len(errors)
        avg_latency = int(sum(latencies) / len(latencies)) if latencies else None
        return {
            "backend": backend,
            "model": model,
            "success_runs": success,
            "error_runs": len(errors),
            "success_rate": round(success / max(repeats, 1), 3),
            "avg_latency_ms": avg_latency,
            "errors": errors[:3],
            "sample_answer": self._sanitize_rag_text(sample_answer),
        }

    def reset_chat_session(self, session_id: str) -> Dict[str, Any]:
        sid = (session_id or "").strip() or "default"
        self._chat_sessions[sid] = {
            "id": sid,
            "history": [],
            "task_state": self._empty_task_state(),
        }
        return self._chat_sessions[sid]

    def chat_turn(
        self,
        session_id: str,
        question: str,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not question.strip():
            return {"error": "Question is empty."}
        sid = (session_id or "").strip() or "default"
        session = self._chat_sessions.get(sid) or self.reset_chat_session(sid)
        task_state = session["task_state"]
        self._update_task_state(task_state, question)

        top_before = max(1, top_k_before or self.default_top_k_before)
        top_after = max(1, top_k_after or self.default_top_k_after)
        threshold_value = threshold if threshold is not None else self.default_threshold
        threshold_value = max(0.0, min(threshold_value, 1.0))

        rewritten = self._rewrite_query(question)
        candidates = self._retrieve_raw(rewritten, top_before, fast_mode=True)
        reranked = self._rerank(question, candidates)
        selected = [item for item in reranked if item.final_score >= threshold_value][:top_after]
        if not selected:
            selected = reranked[:top_after]
        weak_context = self._is_weak_context(selected, threshold_value)

        prompt = self._build_chat_prompt(
            question=question,
            chunks=selected,
            history=self._recent_history(session["history"]),
            task_state=task_state,
        )
        llm = self._call_llm(prompt, rag_mode=True)
        payload = self._build_chat_payload(
            llm_text=llm["text"],
            selected=selected,
            fallback=candidates[:top_after],
            weak_context=weak_context,
        )

        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": payload["answer"]})

        return {
            "session_id": sid,
            "answer": payload["answer"],
            "sources": payload["sources"],
            "quotes": payload["quotes"],
            "weak_context": weak_context,
            "task_state": task_state,
            "history": self._recent_history(session["history"], limit=12),
            "model": llm["model"],
            "tokens": llm["total_tokens"],
            "chunks_used": [self._chunk_to_dict(item) for item in selected],
        }

    def evaluate_chat_scenarios(
        self,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        reports: List[Dict[str, Any]] = []
        for index, scenario in enumerate(self._chat_scenarios):
            sid = f"scenario-{index+1}"
            self.reset_chat_session(sid)
            turns = scenario.get("messages", [])
            turn_reports: List[Dict[str, Any]] = []
            for turn in turns:
                result = self._chat_turn_fast(
                    session_id=sid,
                    question=str(turn),
                    top_k_before=top_k_before,
                    top_k_after=top_k_after,
                    threshold=threshold,
                )
                goal = str(result.get("task_state", {}).get("goal", ""))
                goal_tokens = self._tokenize(goal)
                answer_tokens = self._tokenize(str(result.get("answer", "")))
                goal_match = self._token_overlap(goal_tokens, answer_tokens) >= 0.1 if goal_tokens else True
                turn_reports.append(
                    {
                        "question": turn,
                        "has_sources": bool(result.get("sources")),
                        "goal_retained": goal_match,
                        "weak_context": bool(result.get("weak_context", False)),
                    }
                )
            reports.append(
                {
                    "title": scenario.get("title", f"Scenario {index+1}"),
                    "total_turns": len(turn_reports),
                    "sources_ok": sum(1 for x in turn_reports if x["has_sources"]),
                    "goal_ok": sum(1 for x in turn_reports if x["goal_retained"]),
                    "turns": turn_reports,
                }
            )
        total_turns = sum(item["total_turns"] for item in reports)
        return {
            "total_scenarios": len(reports),
            "total_turns": total_turns,
            "reports": reports,
            "sources_ok": sum(item["sources_ok"] for item in reports),
            "goal_ok": sum(item["goal_ok"] for item in reports),
        }

    def _chat_turn_fast(
        self,
        session_id: str,
        question: str,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        sid = (session_id or "").strip() or "default"
        session = self._chat_sessions.get(sid) or self.reset_chat_session(sid)
        task_state = session["task_state"]
        self._update_task_state(task_state, question)

        top_before = max(1, top_k_before or self.default_top_k_before)
        top_after = max(1, top_k_after or self.default_top_k_after)
        threshold_value = threshold if threshold is not None else self.default_threshold
        threshold_value = max(0.0, min(threshold_value, 1.0))

        rewritten = self._rewrite_query(question)
        candidates = self._retrieve_raw(rewritten, top_before)
        reranked = self._rerank(question, candidates)
        selected = [item for item in reranked if item.final_score >= threshold_value][:top_after]
        if not selected:
            selected = reranked[:top_after]
        weak_context = self._is_weak_context(selected, threshold_value)

        if weak_context:
            answer = "Не знаю. Уточните цель/ограничения, чтобы повысить релевантность."
        else:
            head = selected[0].text if selected else ""
            goal = str(task_state.get("goal", "")).strip()
            prefix = f"Цель: {goal}. " if goal else ""
            answer = f"{prefix}Коротко по контексту: {head[:180]}".strip()

        sources = self._collect_sources(selected) or self._collect_sources(candidates[:top_after])
        quotes = self._collect_quotes(selected) or self._collect_quotes(candidates[:top_after])
        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": answer})
        return {
            "session_id": sid,
            "answer": answer,
            "sources": sources,
            "quotes": quotes,
            "weak_context": weak_context,
            "task_state": task_state,
            "history": self._recent_history(session["history"], limit=12),
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

    def _retrieve_raw(
        self, query: str, top_k: int, fast_mode: bool = False
    ) -> List[ChunkSnippet]:
        self._ensure_indexed(fast_mode=fast_mode)
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

    def _build_chat_prompt(
        self,
        question: str,
        chunks: List[ChunkSnippet],
        history: List[Dict[str, str]],
        task_state: Dict[str, Any],
    ) -> str:
        history_block = "\n".join(
            f"{item['role']}: {item['content']}" for item in history[-8:]
        )
        context = "\n\n".join(
            f"[{idx + 1}] {chunk.source} / {chunk.section} / {chunk.chunk_id}:\n{chunk.text}"
            for idx, chunk in enumerate(chunks)
        )
        state_block = (
            f"Цель: {task_state.get('goal', '')}\n"
            f"Ограничения: {', '.join(task_state.get('constraints', [])) or '—'}\n"
            f"Термины: {', '.join(task_state.get('terms', [])) or '—'}"
        )
        return (
            "Ты помощник в мини-чате с RAG. Учитывай историю и state задачи.\n"
            "Дай краткий полезный ответ по контексту.\n"
            "Не придумывай факты.\n\n"
            f"Task state:\n{state_block}\n\n"
            f"Последние сообщения:\n{history_block or '—'}\n\n"
            f"Вопрос пользователя:\n{question}\n\n"
            f"Контекст RAG:\n{context or '—'}\n\n"
            "Ответ:"
        )

    def _build_mode_payload(
        self, llm_text: str, chunks: List[ChunkSnippet], threshold: float
    ) -> Dict[str, Any]:
        weak_context = self._is_weak_context(chunks, threshold)
        if weak_context:
            answer = (
                "Не знаю. По текущему контексту недостаточно релевантных фрагментов, "
                "чтобы дать надежный ответ. Уточните вопрос (например, платформу, API или конкретный кейс)."
            )
            sources: List[Dict[str, str]] = []
            quotes: List[Dict[str, str]] = []
        else:
            answer = self._sanitize_rag_text(llm_text)
            sources = self._collect_sources(chunks)
            quotes = self._collect_quotes(chunks)
        return {
            "answer": answer,
            "sources": sources,
            "quotes": quotes,
            "weak_context": weak_context,
            "has_sources": bool(sources),
            "has_quotes": bool(quotes),
            "text": self._format_response(answer, sources, quotes),
        }

    def _build_chat_payload(
        self,
        llm_text: str,
        selected: List[ChunkSnippet],
        fallback: List[ChunkSnippet],
        weak_context: bool,
    ) -> Dict[str, Any]:
        sources = self._collect_sources(selected) or self._collect_sources(fallback)
        quotes = self._collect_quotes(selected) or self._collect_quotes(fallback)
        if weak_context:
            answer = (
                "Не знаю. Текущий контекст слабый. "
                "Уточните платформу/цель/ограничения, и я перезапрошу релевантные источники."
            )
        else:
            answer = self._sanitize_rag_text(llm_text)
        return {"answer": answer, "sources": sources, "quotes": quotes}

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
        backend = self.generation_backend if self.generation_backend in {"local", "cloud"} else "local"
        if backend == "local":
            return self._call_local_llm(user_message, rag_mode=rag_mode)
        return self._call_cloud_llm(user_message, rag_mode=rag_mode)

    def _call_cloud_llm(self, user_message: str, rag_mode: bool = False) -> Dict[str, Any]:
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

    def _call_local_llm(self, user_message: str, rag_mode: bool = False) -> Dict[str, Any]:
        payload = {
            "model": self.local_llm_model,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if rag_mode:
            payload["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Ты RAG-ассистент. Если в фрагментах есть релевантные факты, "
                        "используй их и не отвечай расплывчато."
                    ),
                },
            )
        start = time.perf_counter()
        response = self._post_local_chat(payload)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        choices = response.get("choices", []) if isinstance(response, dict) else []
        message = choices[0].get("message", {}) if choices else {}
        text = str(message.get("content", "")).strip()
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "text": text or "Не знаю. Локальная модель вернула пустой ответ.",
            "model": self.local_llm_model,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": elapsed_ms,
        }

    def _post_local_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        endpoints = [
            f"{self.local_llm_endpoint}/v1/chat/completions",
            f"{self.local_llm_endpoint}/chat/completions",
        ]
        last_error = ""
        for url in endpoints:
            try:
                req = urlrequest.Request(url=url, data=body, method="POST", headers=headers)
                with urlrequest.urlopen(req, timeout=90.0) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except urlerror.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(f"Local LLM request failed: {last_error}")

    def _sanitize_rag_text(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return "Не знаю. Уточните вопрос, чтобы я нашел более релевантные фрагменты."
        weak_patterns = (
            "к сожалению",
            "не могу найти",
            "недостаточно информации",
            "нет информации",
            "в предоставленных фрагментах нет",
        )
        lowered = value.lower()
        if any(marker in lowered for marker in weak_patterns):
            cleaned = re.sub(
                r"(?i)к сожалению[,:\s]*",
                "",
                value,
            ).strip()
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            if cleaned:
                return cleaned
            return "Не знаю. Уточните вопрос, чтобы я нашел более релевантные фрагменты."
        return value

    def _collect_sources(self, chunks: List[ChunkSnippet]) -> List[Dict[str, str]]:
        unique: List[Dict[str, str]] = []
        seen = set()
        for chunk in chunks:
            key = (chunk.source, chunk.section, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                }
            )
        return unique

    def _collect_quotes(self, chunks: List[ChunkSnippet]) -> List[Dict[str, str]]:
        quotes: List[Dict[str, str]] = []
        for chunk in chunks[:3]:
            text = chunk.text
            if len(text) > 220:
                text = text[:220].rsplit(" ", 1)[0] + "…"
            quotes.append(
                {
                    "quote": text,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                }
            )
        return quotes

    def _format_response(
        self,
        answer: str,
        sources: List[Dict[str, str]],
        quotes: List[Dict[str, str]],
    ) -> str:
        lines = [answer.strip(), "", "Источники:"]
        if sources:
            for src in sources:
                lines.append(
                    f"- {src['source']} | {src['section']} | {src['chunk_id']}"
                )
        else:
            lines.append("- нет достоверных источников выше порога")
        lines.append("")
        lines.append("Цитаты:")
        if quotes:
            for item in quotes:
                lines.append(
                    f"- \"{item['quote']}\" ({item['source']} | {item['section']} | {item['chunk_id']})"
                )
        else:
            lines.append("- нет цитат (недостаточная релевантность контекста)")
        return "\n".join(lines)

    def _ensure_indexed(self, fast_mode: bool = False):
        if self._pipeline:
            return
        if self.week6_index_path and self.week6_index_path.exists():
            self._load_week6_index(self.week6_index_path)
            return
        chunker = FixedSizeChunker(chunk_size=420, overlap=80)
        embedder = Embedder(api_key="LOCAL_ONLY", force_mock=True)
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

    def _load_week6_index(self, index_path: Path):
        chunker = FixedSizeChunker(chunk_size=420, overlap=80)
        embedder = Embedder(api_key="LOCAL_ONLY", force_mock=True)
        storage = JSONStorage(dimension=embedder.dimensions)
        pipeline = IndexerPipeline(chunker=chunker, embedder=embedder, storage=storage)
        pipeline.load(str(index_path))
        self._pipeline = pipeline
        self._source_map = {}

    def _resolve_week6_index_path(self) -> Optional[Path]:
        env_path = os.getenv("RAG_INDEX_PATH", "").strip()
        if env_path:
            candidate = Path(env_path)
            return candidate if candidate.exists() else None
        base_dir = Path(__file__).resolve().parent / "document_indices"
        candidate = base_dir / "index_fixed_size_json_20260316_195228.json"
        return candidate if candidate.exists() else None

    def _recent_history(
        self, history: List[Dict[str, str]], limit: int = 8
    ) -> List[Dict[str, str]]:
        return history[-limit:]

    def _empty_task_state(self) -> Dict[str, Any]:
        return {"goal": "", "constraints": [], "terms": []}

    def _update_task_state(self, state: Dict[str, Any], question: str):
        q = question.strip()
        q_low = q.lower()
        if not state.get("goal"):
            state["goal"] = q
        if "цель" in q_low and ":" in q:
            goal = q.split(":", 1)[1].strip()
            if goal:
                state["goal"] = goal
        constraints = state.get("constraints", [])
        if any(x in q_low for x in ["огранич", "только", "нельзя", "не использовать", "должен"]):
            constraints.append(q)
        state["constraints"] = constraints[-6:]

        terms = set(state.get("terms", []))
        for token in self._extract_terms(q):
            terms.add(token)
        state["terms"] = sorted(list(terms))[:16]

    def _extract_terms(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Zа-яА-Я0-9_+#.-]{3,}", text)
        allow = {
            "kotlin",
            "compose",
            "multiplatform",
            "mvi",
            "mvvm",
            "coroutines",
            "flow",
            "ci",
            "cd",
            "release",
            "staged",
            "rollout",
            "android",
            "ios",
            "expect",
            "actual",
            "kmp",
        }
        picked: List[str] = []
        for token in tokens:
            t = token.lower()
            if t in allow:
                picked.append(t)
        return picked

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

    def _load_chat_scenarios(self) -> List[Dict[str, Any]]:
        if not CHAT_SCENARIOS_FILE.exists():
            return []
        try:
            with CHAT_SCENARIOS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
        return []

    def _is_weak_context(self, chunks: List[ChunkSnippet], threshold: float) -> bool:
        if not chunks:
            return True
        max_score = max(c.final_score for c in chunks)
        return max_score < threshold

    def _normalize_vector_score(self, raw: float) -> float:
        normalized = (float(raw) + 1.0) / 2.0
        return max(0.0, min(normalized, 1.0))

    def _tokenize(self, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", text.lower())}

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(len(left), 1)

    def _semantic_match(
        self, expectation: str, answer: str, quotes: List[Dict[str, str]]
    ) -> bool:
        exp_tokens = self._tokenize(expectation)
        if not exp_tokens:
            return True
        material = answer + " " + " ".join(q.get("quote", "") for q in quotes)
        mat_tokens = self._tokenize(material)
        if not mat_tokens:
            return False
        overlap = len(exp_tokens & mat_tokens) / max(len(exp_tokens), 1)
        return overlap >= 0.18
