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
        return AgentResponse(
            text=f"Error: {error_text}",
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
