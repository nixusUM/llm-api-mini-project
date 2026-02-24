import json
from pathlib import Path
from dataclasses import dataclass
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


class LLMAgent:
    def __init__(self, history_path: str = "data/chat_history.json") -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> list[dict[str, str]]:
        if not self.history_path.exists():
            return []
        try:
            raw = self.history_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        history: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            history.append({"role": role, "content": content})
        return history

    def clear_history(self) -> None:
        self._save_history([])

    def run_chat_persistent(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResponse:
        history = self.load_history()
        response = self.run_chat(history, user_message, model_id, temperature, max_tokens)
        history.append({"role": "user", "content": user_message.strip()})
        history.append({"role": "assistant", "content": response.text})
        self._save_history(history[-40:])
        return response

    def run_chat(
        self,
        history: list[dict[str, str]],
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResponse:
        prompt = self._build_chat_prompt(history, user_message)
        return self.run(prompt, model_id, temperature, max_tokens)

    def run(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResponse:
        started = perf_counter()
        try:
            text, used_model, usage = ask_claude_with_meta(
                prompt=prompt,
                model_override=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._from_success(text, used_model, usage, started)
        except Exception as exc:
            return self._from_error(str(exc), model_id, started)

    def _from_success(self, text: str, model: str, usage: dict, started: float) -> AgentResponse:
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
        )

    def _from_error(self, error_text: str, model: str, started: float) -> AgentResponse:
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

    def _build_chat_prompt(self, history: list[dict[str, str]], user_message: str) -> str:
        lines = [
            "You are a helpful assistant in a multi-turn chat.",
            "Use prior conversation as context and answer the latest user message.",
            "",
            "Conversation history:",
        ]
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

    def _save_history(self, history: list[dict[str, str]]) -> None:
        payload = json.dumps(history, ensure_ascii=False, indent=2)
        self.history_path.write_text(payload, encoding="utf-8")
