import json
import re
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
    strategy: str
    branch: str
    current_request_tokens: int
    history_tokens_full: int
    history_tokens_effective: int
    facts_tokens: int
    context_tokens_estimate: int
    context_limit_tokens: int
    overflowed: bool


class LLMAgent:
    def __init__(self, state_path: str = "data/agent_state.json") -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self, branch_id: str | None = None) -> list[dict]:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        branch = self._get_branch(state, active)
        return list(branch["messages"])

    def load_facts(self, branch_id: str | None = None) -> dict[str, str]:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        branch = self._get_branch(state, active)
        return dict(branch["facts"])

    def list_checkpoints(self, branch_id: str | None = None) -> list[dict]:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        branch = self._get_branch(state, active)
        return list(branch["checkpoints"])

    def list_branches(self) -> list[str]:
        state = self._load_state()
        return sorted(state["branches"].keys())

    def get_active_branch(self) -> str:
        return self._load_state()["active_branch"]

    def switch_branch(self, branch_id: str) -> bool:
        state = self._load_state()
        if branch_id not in state["branches"]:
            return False
        state["active_branch"] = branch_id
        self._save_state(state)
        return True

    def create_checkpoint(self, label: str, branch_id: str | None = None) -> str:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        branch = self._get_branch(state, active)
        checkpoint_id = f"cp_{len(branch['checkpoints']) + 1}"
        branch["checkpoints"].append(
            {
                "id": checkpoint_id,
                "label": label.strip() or checkpoint_id,
                "message_index": len(branch["messages"]),
            }
        )
        self._save_state(state)
        return checkpoint_id

    def create_branch_from_checkpoint(
        self,
        source_branch: str,
        checkpoint_id: str,
        new_branch: str,
    ) -> tuple[bool, str]:
        state = self._load_state()
        source = self._get_branch(state, source_branch)
        if new_branch in state["branches"]:
            return False, "Branch already exists."
        checkpoint = next((cp for cp in source["checkpoints"] if cp["id"] == checkpoint_id), None)
        if not checkpoint:
            return False, "Checkpoint not found."
        idx = int(checkpoint["message_index"])
        state["branches"][new_branch] = {
            "messages": list(source["messages"][:idx]),
            "facts": dict(source["facts"]),
            "checkpoints": [],
        }
        state["active_branch"] = new_branch
        self._save_state(state)
        return True, "Branch created."

    def clear_branch(self, branch_id: str | None = None) -> None:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        state["branches"][active] = {"messages": [], "facts": {}, "checkpoints": []}
        self._save_state(state)

    def clear_all(self) -> None:
        self._save_state(self._default_state())

    def run_chat_persistent(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        strategy: str,
        window_n: int,
        branch_id: str | None = None,
        context_limit_override: int | None = None,
    ) -> AgentResponse:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        if active not in state["branches"]:
            state["branches"][active] = {"messages": [], "facts": {}, "checkpoints": []}
        state["active_branch"] = active
        branch = self._get_branch(state, active)

        if strategy == "facts":
            self._update_facts(branch["facts"], user_message)

        full_history = list(branch["messages"])
        effective_history = self._select_history_for_strategy(full_history, strategy, window_n)
        facts_text = self._facts_to_text(branch["facts"]) if strategy == "facts" else ""

        response = self._run_with_context(
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            strategy=strategy,
            branch=active,
            full_history=full_history,
            effective_history=effective_history,
            facts_text=facts_text,
            context_limit_override=context_limit_override,
        )

        branch["messages"].append({"role": "user", "content": user_message.strip()})
        branch["messages"].append(
            {
                "role": "assistant",
                "content": response.text,
                "meta": {
                    "strategy": strategy,
                    "current_request_tokens": response.current_request_tokens,
                    "history_tokens_full": response.history_tokens_full,
                    "history_tokens_effective": response.history_tokens_effective,
                    "facts_tokens": response.facts_tokens,
                    "response_tokens": response.output_tokens,
                    "total_turn_tokens": (
                        response.current_request_tokens
                        + response.history_tokens_effective
                        + response.output_tokens
                    ),
                },
            }
        )
        branch["messages"] = branch["messages"][-220:]
        self._save_state(state)
        return response

    def _run_with_context(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        strategy: str,
        branch: str,
        full_history: list[dict],
        effective_history: list[dict],
        facts_text: str,
        context_limit_override: int | None,
    ) -> AgentResponse:
        current_request_tokens = self._estimate_tokens(user_message)
        full_history_tokens = self._estimate_history_tokens(full_history)
        effective_history_tokens = self._estimate_history_tokens(effective_history)
        facts_tokens = self._estimate_tokens(facts_text)
        context_tokens = current_request_tokens + effective_history_tokens + facts_tokens
        context_limit = context_limit_override or self._infer_context_limit_tokens(model_id)

        if context_tokens + max_tokens > context_limit:
            return self._overflow_response(
                model=model_id,
                strategy=strategy,
                branch=branch,
                current_request_tokens=current_request_tokens,
                full_history_tokens=full_history_tokens,
                effective_history_tokens=effective_history_tokens,
                facts_tokens=facts_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
            )

        prompt = self._build_chat_prompt(
            strategy=strategy,
            history=effective_history,
            user_message=user_message,
            facts_text=facts_text,
        )
        started = perf_counter()
        try:
            text, used_model, usage = ask_claude_with_meta(
                prompt=prompt,
                model_override=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return self._error_response(
                error_text=str(exc),
                model=model_id,
                strategy=strategy,
                branch=branch,
                started=started,
                current_request_tokens=current_request_tokens,
                full_history_tokens=full_history_tokens,
                effective_history_tokens=effective_history_tokens,
                facts_tokens=facts_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
            )
        return self._success_response(
            text=text,
            model=used_model,
            usage=usage,
            strategy=strategy,
            branch=branch,
            started=started,
            current_request_tokens=current_request_tokens,
            full_history_tokens=full_history_tokens,
            effective_history_tokens=effective_history_tokens,
            facts_tokens=facts_tokens,
            context_tokens=context_tokens,
            context_limit=context_limit,
        )

    def _success_response(
        self,
        text: str,
        model: str,
        usage: dict,
        strategy: str,
        branch: str,
        started: float,
        current_request_tokens: int,
        full_history_tokens: int,
        effective_history_tokens: int,
        facts_tokens: int,
        context_tokens: int,
        context_limit: int,
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
            strategy=strategy,
            branch=branch,
            current_request_tokens=current_request_tokens,
            history_tokens_full=full_history_tokens,
            history_tokens_effective=effective_history_tokens,
            facts_tokens=facts_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=False,
        )

    def _error_response(
        self,
        error_text: str,
        model: str,
        strategy: str,
        branch: str,
        started: float,
        current_request_tokens: int,
        full_history_tokens: int,
        effective_history_tokens: int,
        facts_tokens: int,
        context_tokens: int,
        context_limit: int,
    ) -> AgentResponse:
        latency_ms = int((perf_counter() - started) * 1000)
        return AgentResponse(
            text=f"Error: {self._friendly_error_text(error_text)}",
            used_model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            cost_text="N/A",
            strategy=strategy,
            branch=branch,
            current_request_tokens=current_request_tokens,
            history_tokens_full=full_history_tokens,
            history_tokens_effective=effective_history_tokens,
            facts_tokens=facts_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=False,
        )

    def _overflow_response(
        self,
        model: str,
        strategy: str,
        branch: str,
        current_request_tokens: int,
        full_history_tokens: int,
        effective_history_tokens: int,
        facts_tokens: int,
        context_tokens: int,
        context_limit: int,
    ) -> AgentResponse:
        return AgentResponse(
            text=(
                "Error: estimated context exceeds model limit. "
                "Use smaller window or switch strategy."
            ),
            used_model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            cost_text="N/A",
            strategy=strategy,
            branch=branch,
            current_request_tokens=current_request_tokens,
            history_tokens_full=full_history_tokens,
            history_tokens_effective=effective_history_tokens,
            facts_tokens=facts_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            overflowed=True,
        )

    def _select_history_for_strategy(self, history: list[dict], strategy: str, window_n: int) -> list[dict]:
        if strategy in {"sliding", "facts"}:
            n = max(2, window_n)
            return history[-n:]
        return history

    def _build_chat_prompt(
        self,
        strategy: str,
        history: list[dict],
        user_message: str,
        facts_text: str,
    ) -> str:
        lines = [
            "You are a helpful assistant in a multi-turn chat.",
            "Respect existing decisions and constraints from context.",
            f"Context strategy: {strategy}",
        ]
        if facts_text:
            lines.extend(["", "Sticky facts:", facts_text])
        lines.extend(["", "Conversation history:"])
        for item in history:
            role = item.get("role", "").strip().lower()
            content = item.get("content", "").strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        lines.extend(["", f"User: {user_message.strip()}", "Assistant:"])
        return "\n".join(lines)

    def _update_facts(self, facts: dict[str, str], user_message: str) -> None:
        text = user_message.strip()
        if not text:
            return
        lowered = text.lower()
        self._update_fact_if_contains(facts, lowered, text, "goal", ["цель", "goal"])
        self._update_fact_if_contains(facts, lowered, text, "constraints", ["огранич", "constraint"])
        self._update_fact_if_contains(facts, lowered, text, "preferences", ["предпоч", "prefer"])
        self._update_fact_if_contains(facts, lowered, text, "decisions", ["решени", "decision", "договор"])
        self._update_fact_if_contains(facts, lowered, text, "budget", ["бюджет", "budget"])
        self._update_fact_if_contains(facts, lowered, text, "kpi", ["kpi", "метрик"])
        kv_matches = re.findall(r"([A-Za-zА-Яа-я0-9_ ]{2,30})\s*:\s*([^,\n]{2,80})", text)
        for key, value in kv_matches:
            normalized = key.strip().lower().replace(" ", "_")
            if len(normalized) > 30:
                continue
            facts[normalized] = value.strip()
        facts["last_user_intent"] = text[:140]
        if len(facts) > 15:
            keys = list(facts.keys())
            for key in keys[:-15]:
                facts.pop(key, None)

    def _update_fact_if_contains(
        self,
        facts: dict[str, str],
        lowered: str,
        original: str,
        key: str,
        needles: list[str],
    ) -> None:
        if any(needle in lowered for needle in needles):
            facts[key] = original[:180]

    def _facts_to_text(self, facts: dict[str, str]) -> str:
        if not facts:
            return ""
        lines = [f"- {key}: {value}" for key, value in facts.items()]
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

    def _default_state(self) -> dict:
        return {
            "active_branch": "main",
            "branches": {"main": {"messages": [], "facts": {}, "checkpoints": []}},
        }

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return self._default_state()
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if not isinstance(parsed, dict):
            return self._default_state()
        active = str(parsed.get("active_branch", "main")).strip() or "main"
        branches = parsed.get("branches", {})
        if not isinstance(branches, dict):
            return self._default_state()
        normalized_branches: dict[str, dict] = {}
        for name, branch in branches.items():
            if not isinstance(name, str) or not isinstance(branch, dict):
                continue
            messages = self._normalize_messages(branch.get("messages", []))
            facts = self._normalize_facts(branch.get("facts", {}))
            checkpoints = self._normalize_checkpoints(branch.get("checkpoints", []), len(messages))
            normalized_branches[name] = {
                "messages": messages[-220:],
                "facts": facts,
                "checkpoints": checkpoints[-40:],
            }
        if not normalized_branches:
            return self._default_state()
        if active not in normalized_branches:
            active = sorted(normalized_branches.keys())[0]
        return {"active_branch": active, "branches": normalized_branches}

    def _save_state(self, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        self.state_path.write_text(payload, encoding="utf-8")

    def _get_branch(self, state: dict, branch_id: str) -> dict:
        if branch_id not in state["branches"]:
            state["branches"][branch_id] = {"messages": [], "facts": {}, "checkpoints": []}
        return state["branches"][branch_id]

    def _normalize_messages(self, messages: object) -> list[dict]:
        if not isinstance(messages, list):
            return []
        result: list[dict] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized = {"role": role, "content": content}
            meta = item.get("meta")
            if isinstance(meta, dict):
                normalized["meta"] = meta
            result.append(normalized)
        return result

    def _normalize_facts(self, facts: object) -> dict[str, str]:
        if not isinstance(facts, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in facts.items():
            k = str(key).strip()
            v = str(value).strip()
            if not k or not v:
                continue
            out[k[:32]] = v[:200]
        return out

    def _normalize_checkpoints(self, checkpoints: object, max_messages: int) -> list[dict]:
        if not isinstance(checkpoints, list):
            return []
        out: list[dict] = []
        for item in checkpoints:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "")).strip()
            label = str(item.get("label", "")).strip() or cid
            idx_raw = item.get("message_index", 0)
            idx = int(idx_raw) if isinstance(idx_raw, int) else 0
            idx = max(0, min(idx, max_messages))
            if not cid:
                continue
            out.append({"id": cid, "label": label[:60], "message_index": idx})
        return out
