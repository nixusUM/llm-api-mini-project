import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from anthropic_client import ask_claude_with_meta


@dataclass
class AgentResponse:
    text: str
    used_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    cost_text: str
    current_request_tokens: int
    history_tokens: int
    context_tokens_estimate: int
    context_limit_tokens: int
    overflowed: bool
    history_tokens_full: int
    history_tokens_effective: int
    summary_tokens: int
    token_savings: int
    compression_enabled: bool


class LLMAgent:
    def __init__(self, history_path: str = "data/chat_history.json") -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> list[dict]:
        return self._load_state()["all_messages"]

    def clear_history(self) -> None:
        self._save_state(self._default_state())

    def run_chat_persistent(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        context_limit_override: int | None = None,
        compression_enabled: bool = True,
        keep_last_n: int = 8,
        summarize_every_n: int = 10,
    ) -> AgentResponse:
        state = self._load_state()
        all_messages = state["all_messages"]
        summary = state["summary"]
        recent_messages = state["recent_messages"]

        if compression_enabled:
            summary, recent_messages = self._compress_recent_messages(
                summary=summary,
                recent_messages=recent_messages,
                model_id=model_id,
                keep_last_n=keep_last_n,
                summarize_every_n=summarize_every_n,
            )

        active_history = recent_messages if compression_enabled else all_messages
        response = self.run_chat(
            history=active_history,
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            context_limit_override=context_limit_override,
            summary=summary if compression_enabled else "",
            compression_enabled=compression_enabled,
            full_history=all_messages,
        )

        user_item = {"role": "user", "content": user_message.strip()}
        assistant_item = {
            "role": "assistant",
            "content": response.text,
            "meta": {
                "current_request_tokens": response.current_request_tokens,
                "history_tokens": response.history_tokens,
                "context_tokens_estimate": response.context_tokens_estimate,
                "context_limit_tokens": response.context_limit_tokens,
                "response_tokens": response.output_tokens,
                "overflowed": response.overflowed,
                "history_tokens_full": response.history_tokens_full,
                "history_tokens_effective": response.history_tokens_effective,
                "summary_tokens": response.summary_tokens,
                "token_savings": response.token_savings,
                "compression_enabled": response.compression_enabled,
            },
        }
        all_messages.append(user_item)
        all_messages.append(assistant_item)
        recent_messages.append(user_item)
        recent_messages.append(assistant_item)

        if compression_enabled:
            summary, recent_messages = self._compress_recent_messages(
                summary=summary,
                recent_messages=recent_messages,
                model_id=model_id,
                keep_last_n=keep_last_n,
                summarize_every_n=summarize_every_n,
            )

        state_to_save = {
            "summary": summary,
            "recent_messages": recent_messages[-60:],
            "all_messages": all_messages[-200:],
        }
        self._save_state(state_to_save)
        return response

    def compare_compression(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        context_limit_override: int | None = None,
        keep_last_n: int = 8,
        summarize_every_n: int = 10,
    ) -> dict[str, AgentResponse]:
        state = self._load_state()
        all_messages = list(state["all_messages"])

        no_compression = self.run_chat(
            history=all_messages,
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            context_limit_override=context_limit_override,
            summary="",
            compression_enabled=False,
            full_history=all_messages,
        )

        summary, recent = self._compress_recent_messages(
            summary=state["summary"],
            recent_messages=list(state["recent_messages"]),
            model_id=model_id,
            keep_last_n=keep_last_n,
            summarize_every_n=summarize_every_n,
        )
        with_compression = self.run_chat(
            history=recent,
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            context_limit_override=context_limit_override,
            summary=summary,
            compression_enabled=True,
            full_history=all_messages,
        )

        return {"no_compression": no_compression, "with_compression": with_compression}

    def run_chat(
        self,
        history: list[dict[str, str]],
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        context_limit_override: int | None = None,
        summary: str = "",
        compression_enabled: bool = False,
        full_history: list[dict] | None = None,
    ) -> AgentResponse:
        full = full_history if full_history is not None else history
        current_request_tokens = self._estimate_tokens(user_message)
        summary_tokens = self._estimate_tokens(summary)
        history_tokens_effective = self._estimate_history_tokens(history) + summary_tokens
        history_tokens_full = self._estimate_history_tokens(full)
        token_savings = max(0, history_tokens_full - history_tokens_effective)
        context_tokens = current_request_tokens + history_tokens_effective
        context_limit = context_limit_override or self._infer_context_limit_tokens(model_id)

        if context_tokens + max_tokens > context_limit:
            return self._from_overflow(
                model=model_id,
                current_request_tokens=current_request_tokens,
                history_tokens=history_tokens_effective,
                context_tokens=context_tokens,
                context_limit=context_limit,
                history_tokens_full=history_tokens_full,
                history_tokens_effective=history_tokens_effective,
                summary_tokens=summary_tokens,
                token_savings=token_savings,
                compression_enabled=compression_enabled,
            )

        prompt = self._build_chat_prompt(history=history, user_message=user_message, summary=summary)
        return self.run(
            prompt=prompt,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens_effective,
            context_tokens=context_tokens,
            context_limit=context_limit,
            history_tokens_full=history_tokens_full,
            history_tokens_effective=history_tokens_effective,
            summary_tokens=summary_tokens,
            token_savings=token_savings,
            compression_enabled=compression_enabled,
        )

    def run(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        current_request_tokens: int = 0,
        history_tokens: int = 0,
        context_tokens: int = 0,
        context_limit: int = 0,
        history_tokens_full: int = 0,
        history_tokens_effective: int = 0,
        summary_tokens: int = 0,
        token_savings: int = 0,
        compression_enabled: bool = False,
    ) -> AgentResponse:
        started = perf_counter()
        try:
            text, used_model, usage = ask_claude_with_meta(
                prompt=prompt,
                model_override=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._from_success(
                text=text,
                model=used_model,
                usage=usage,
                started=started,
                current_request_tokens=current_request_tokens,
                history_tokens=history_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
                history_tokens_full=history_tokens_full,
                history_tokens_effective=history_tokens_effective,
                summary_tokens=summary_tokens,
                token_savings=token_savings,
                compression_enabled=compression_enabled,
            )
        except Exception as exc:
            return self._from_error(
                error_text=str(exc),
                model=model_id,
                started=started,
                current_request_tokens=current_request_tokens,
                history_tokens=history_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
                history_tokens_full=history_tokens_full,
                history_tokens_effective=history_tokens_effective,
                summary_tokens=summary_tokens,
                token_savings=token_savings,
                compression_enabled=compression_enabled,
            )

    def _from_success(
        self,
        text: str,
        model: str,
        usage: dict,
        started: float,
        current_request_tokens: int,
        history_tokens: int,
        context_tokens: int,
        context_limit: int,
        history_tokens_full: int,
        history_tokens_effective: int,
        summary_tokens: int,
        token_savings: int,
        compression_enabled: bool,
    ) -> AgentResponse:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = input_tokens + output_tokens
        latency_ms = int((perf_counter() - started) * 1000)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        return AgentResponse(
            text=text,
            used_model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_text=self._format_cost(cost),
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=False,
            history_tokens_full=history_tokens_full,
            history_tokens_effective=history_tokens_effective,
            summary_tokens=summary_tokens,
            token_savings=token_savings,
            compression_enabled=compression_enabled,
        )

    def _from_error(
        self,
        error_text: str,
        model: str,
        started: float,
        current_request_tokens: int,
        history_tokens: int,
        context_tokens: int,
        context_limit: int,
        history_tokens_full: int,
        history_tokens_effective: int,
        summary_tokens: int,
        token_savings: int,
        compression_enabled: bool,
    ) -> AgentResponse:
        latency_ms = int((perf_counter() - started) * 1000)
        friendly_error = self._friendly_error_text(error_text)
        return AgentResponse(
            text=f"Error: {friendly_error}",
            used_model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            cost_text="N/A",
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=False,
            history_tokens_full=history_tokens_full,
            history_tokens_effective=history_tokens_effective,
            summary_tokens=summary_tokens,
            token_savings=token_savings,
            compression_enabled=compression_enabled,
        )

    def _estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float | None:
        rates = self._infer_rates(model_id)
        if not rates:
            return None
        input_rate, output_rate = rates
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        return input_cost + output_cost

    def _infer_rates(self, model_id: str) -> tuple[float, float] | None:
        model = model_id.lower()
        if "haiku" in model:
            return (0.80, 4.00)
        if "sonnet" in model:
            return (3.00, 15.00)
        if "opus" in model:
            return (15.00, 75.00)
        return None

    def _format_cost(self, cost: float | None) -> str:
        if cost is None:
            return "N/A"
        return f"${cost:.6f}"

    def _friendly_error_text(self, error_text: str) -> str:
        normalized = error_text.lower()
        if "overloaded" in normalized or "(529)" in normalized:
            return "LLM provider is temporarily overloaded (529). Please retry in a few seconds."
        return error_text

    def _build_chat_prompt(self, history: list[dict[str, str]], user_message: str, summary: str = "") -> str:
        lines = [
            "You are a helpful assistant in a multi-turn chat.",
            "Use prior conversation as context and answer the latest user message.",
        ]
        if summary.strip():
            lines.append("")
            lines.append("Conversation summary of earlier messages:")
            lines.append(summary.strip())
        lines.append("")
        lines.append("Recent conversation history:")
        for item in history:
            role = item.get("role", "").strip().lower()
            content = item.get("content", "").strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        lines.append("")
        lines.append(f"User: {user_message.strip()}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        cleaned = text.strip()
        if not cleaned:
            return 0
        return max(1, (len(cleaned) // 4) + 1)

    def _estimate_history_tokens(self, history: list[dict]) -> int:
        total = 0
        for item in history:
            if not isinstance(item, dict):
                continue
            total += self._estimate_tokens(str(item.get("content", "")))
            total += 4
        return total

    def _infer_context_limit_tokens(self, model_id: str) -> int:
        model = model_id.lower()
        if "haiku" in model or "sonnet" in model or "opus" in model:
            return 200_000
        return 100_000

    def _from_overflow(
        self,
        model: str,
        current_request_tokens: int,
        history_tokens: int,
        context_tokens: int,
        context_limit: int,
        history_tokens_full: int,
        history_tokens_effective: int,
        summary_tokens: int,
        token_savings: int,
        compression_enabled: bool,
    ) -> AgentResponse:
        return AgentResponse(
            text=(
                "Error: estimated context exceeds model limit. "
                "Clear history, reduce keep_last_n, or shorten messages."
            ),
            used_model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            cost_text="N/A",
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=True,
            history_tokens_full=history_tokens_full,
            history_tokens_effective=history_tokens_effective,
            summary_tokens=summary_tokens,
            token_savings=token_savings,
            compression_enabled=compression_enabled,
        )

    def _default_state(self) -> dict:
        return {"summary": "", "recent_messages": [], "all_messages": []}

    def _load_state(self) -> dict:
        if not self.history_path.exists():
            return self._default_state()
        try:
            raw = self.history_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if isinstance(parsed, list):
            normalized = self._normalize_messages(parsed)
            return {"summary": "", "recent_messages": normalized[-60:], "all_messages": normalized}
        if not isinstance(parsed, dict):
            return self._default_state()
        summary = str(parsed.get("summary", "")).strip()
        recent = self._normalize_messages(parsed.get("recent_messages", []))
        all_messages = self._normalize_messages(parsed.get("all_messages", []))
        if not all_messages and recent:
            all_messages = list(recent)
        if not recent and all_messages:
            recent = all_messages[-60:]
        return {"summary": summary, "recent_messages": recent, "all_messages": all_messages}

    def _normalize_messages(self, messages: object) -> list[dict]:
        if not isinstance(messages, list):
            return []
        normalized: list[dict] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            out = {"role": role, "content": content}
            meta = item.get("meta")
            if isinstance(meta, dict):
                out["meta"] = meta
            normalized.append(out)
        return normalized

    def _save_state(self, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        self.history_path.write_text(payload, encoding="utf-8")

    def _compress_recent_messages(
        self,
        summary: str,
        recent_messages: list[dict],
        model_id: str,
        keep_last_n: int,
        summarize_every_n: int,
    ) -> tuple[str, list[dict]]:
        keep = max(2, keep_last_n)
        step = max(2, summarize_every_n)
        if len(recent_messages) <= keep:
            return summary, recent_messages
        compressed_summary = summary
        remaining = list(recent_messages)
        while len(remaining) > keep + step:
            chunk = remaining[:step]
            compressed_summary = self._merge_summary(compressed_summary, chunk, model_id)
            remaining = remaining[step:]
        return compressed_summary, remaining

    def _merge_summary(self, current_summary: str, chunk: list[dict], model_id: str) -> str:
        chunk_lines: list[str] = []
        for item in chunk:
            role = item.get("role", "")
            prefix = "User" if role == "user" else "Assistant"
            chunk_lines.append(f"{prefix}: {item.get('content', '')}")
        chunk_text = "\n".join(chunk_lines)
        prompt = (
            "Update conversation summary. Keep key facts, decisions, and constraints. "
            "Use concise bullet points in plain text.\n\n"
            f"Current summary:\n{current_summary or '(empty)'}\n\n"
            f"New messages:\n{chunk_text}\n\n"
            "Return only updated summary."
        )
        try:
            text, _, _ = ask_claude_with_meta(
                prompt=prompt,
                model_override=model_id,
                temperature=0.0,
                max_tokens=220,
            )
            candidate = text.strip()
            if candidate:
                return candidate
        except Exception:
            pass
        fallback = f"{current_summary}\n- {chunk_text[:220]}".strip()
        return fallback[:4000]
