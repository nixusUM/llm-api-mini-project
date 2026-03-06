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
    working_tokens: int
    long_term_tokens: int
    profile_tokens: int
    context_tokens_estimate: int
    context_limit_tokens: int
    include_memory_layers: bool
    profile_id: str
    overflowed: bool
    invariant_tokens: int = 0
    blocked_by_invariants: bool = False
    invariant_report: str = ""


class LLMAgent:
    def __init__(self, state_path: str = "data/agent_state.json") -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self, branch_id: str | None = None) -> list[dict]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        return list(branch["messages"])

    def load_facts(self, branch_id: str | None = None) -> dict[str, str]:
        return self.load_working_memory(branch_id)

    def load_working_memory(self, branch_id: str | None = None) -> dict[str, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        return dict(branch["working_memory"])

    def load_long_term_memory(self) -> dict[str, str]:
        state = self._load_state()
        return dict(state["long_term_memory"])

    def load_invariants(self) -> list[dict[str, str | bool]]:
        state = self._load_state()
        invariants = state.get("invariants", {})
        if not isinstance(invariants, dict):
            return []
        rows: list[dict[str, str | bool]] = []
        for inv_id in sorted(invariants.keys()):
            item = invariants.get(inv_id, {})
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "id": inv_id,
                    "category": str(item.get("category", "general")),
                    "text": str(item.get("text", "")),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        return rows

    def save_invariant(
        self,
        invariant_id: str,
        category: str,
        text: str,
        enabled: bool = True,
    ) -> tuple[bool, str]:
        normalized = self._normalize_memory_key(invariant_id)
        if not normalized:
            return False, "Invariant id is empty."
        cleaned_text = text.strip()[:240]
        if not cleaned_text:
            return False, "Invariant text is empty."
        state = self._load_state()
        if "invariants" not in state or not isinstance(state["invariants"], dict):
            state["invariants"] = {}
        state["invariants"][normalized] = {
            "category": (category.strip().lower() or "general")[:40],
            "text": cleaned_text,
            "enabled": bool(enabled),
        }
        self._save_state(state)
        return True, f"Invariant saved: {normalized}"

    def delete_invariant(self, invariant_id: str) -> tuple[bool, str]:
        normalized = self._normalize_memory_key(invariant_id)
        if not normalized:
            return False, "Invariant id is empty."
        state = self._load_state()
        invariants = state.get("invariants", {})
        if not isinstance(invariants, dict) or normalized not in invariants:
            return False, f"Invariant not found: {normalized}"
        invariants.pop(normalized, None)
        self._save_state(state)
        return True, f"Invariant deleted: {normalized}"

    def short_term_memory(self, window_n: int = 8, branch_id: str | None = None) -> list[dict]:
        history = self.load_history(branch_id)
        n = max(2, window_n)
        return history[-n:]

    def list_checkpoints(self, branch_id: str | None = None) -> list[dict]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        return list(branch["checkpoints"])

    def list_branches(self) -> list[str]:
        state = self._load_state()
        return sorted(state["branches"].keys())

    def get_active_branch(self) -> str:
        return self._load_state()["active_branch"]

    def list_profiles(self) -> list[str]:
        state = self._load_state()
        return sorted(state["profiles"].keys())

    def get_active_profile(self) -> str:
        return self._load_state()["active_profile"]

    def load_profile(self, profile_id: str | None = None) -> dict[str, str]:
        state = self._load_state()
        active = profile_id or state["active_profile"]
        profile = state["profiles"].get(active, {})
        return dict(profile) if isinstance(profile, dict) else self._empty_profile()

    def switch_profile(self, profile_id: str) -> bool:
        state = self._load_state()
        if profile_id not in state["profiles"]:
            return False
        state["active_profile"] = profile_id
        self._save_state(state)
        return True

    def save_profile(
        self,
        profile_id: str,
        style: str,
        output_format: str,
        constraints: str,
        preferences: str,
    ) -> tuple[bool, str]:
        normalized = self._normalize_memory_key(profile_id)
        if not normalized:
            return False, "Profile id is empty."
        state = self._load_state()
        state["profiles"][normalized] = {
            "style": style.strip()[:220],
            "format": output_format.strip()[:220],
            "constraints": constraints.strip()[:220],
            "preferences": preferences.strip()[:220],
        }
        state["active_profile"] = normalized
        self._save_state(state)
        return True, f"Profile saved: {normalized}"

    def delete_profile(self, profile_id: str) -> tuple[bool, str]:
        normalized = self._normalize_memory_key(profile_id)
        if not normalized:
            return False, "Profile id is empty."
        state = self._load_state()
        if normalized == "default":
            return False, "Default profile cannot be deleted."
        if normalized not in state["profiles"]:
            return False, f"Profile not found: {normalized}"
        state["profiles"].pop(normalized, None)
        if state["active_profile"] == normalized:
            state["active_profile"] = "default"
        self._save_state(state)
        return True, f"Profile deleted: {normalized}"

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
            {"id": checkpoint_id, "label": label.strip() or checkpoint_id, "message_index": len(branch["messages"])}
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
            "working_memory": dict(source["working_memory"]),
            "checkpoints": [],
        }
        state["active_branch"] = new_branch
        self._save_state(state)
        return True, "Branch created."

    def load_task_state(self, branch_id: str | None = None) -> dict[str, object]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = branch.get("task_state", {})
        return dict(task_state) if isinstance(task_state, dict) else self._default_task_state()

    def set_task_state(
        self,
        stage: str,
        current_step: str,
        expected_action: str,
        branch_id: str | None = None,
    ) -> tuple[bool, str]:
        normalized_stage = self._normalize_stage(stage)
        if not normalized_stage:
            return False, f"Unknown stage: {stage}"
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        current_stage = str(task_state.get("stage", "planning")).strip().lower()
        if normalized_stage != current_stage:
            ok, reason = self._can_transition(task_state, current_stage, normalized_stage)
            if not ok:
                return False, reason
            self._apply_stage_defaults(task_state, normalized_stage)
            task_state["stage"] = normalized_stage
        cleaned_step = current_step.strip()[:220]
        cleaned_action = expected_action.strip()[:220]
        if cleaned_step:
            task_state["current_step"] = cleaned_step
        if cleaned_action:
            task_state["expected_action"] = cleaned_action
        self._save_state(state)
        return True, f"Task state updated: {task_state['stage']}"

    def pause_task(self, branch_id: str | None = None) -> tuple[bool, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        task_state["paused"] = True
        self._save_state(state)
        return True, "Task paused."

    def resume_task(self, branch_id: str | None = None) -> tuple[bool, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        task_state["paused"] = False
        self._save_state(state)
        return True, "Task resumed."

    def advance_task_stage(self, branch_id: str | None = None) -> tuple[bool, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        current = str(task_state.get("stage", "planning")).strip().lower()
        flow = self._stage_flow()
        if current not in flow:
            current = "planning"
        idx = flow.index(current)
        if idx == len(flow) - 1:
            return False, "Task already at final stage: done"
        next_stage = flow[idx + 1]
        ok, reason = self._can_transition(task_state, current, next_stage)
        if not ok:
            return False, reason
        self._apply_stage_defaults(task_state, next_stage)
        task_state["stage"] = next_stage
        self._save_state(state)
        return True, f"Task moved to: {next_stage}"

    def approve_plan(self, branch_id: str | None = None) -> tuple[bool, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        stage = str(task_state.get("stage", "planning")).strip().lower()
        if stage != "planning":
            return False, "Plan approval is allowed only at planning stage."
        task_state["plan_approved"] = True
        self._save_state(state)
        return True, "Plan approved. Transition to execution is now allowed."

    def pass_validation(self, branch_id: str | None = None) -> tuple[bool, str]:
        state = self._load_state()
        branch = self._get_branch(state, branch_id or state["active_branch"])
        task_state = self._ensure_task_state(branch)
        stage = str(task_state.get("stage", "planning")).strip().lower()
        if stage != "validation":
            return False, "Validation pass can be marked only at validation stage."
        task_state["validation_passed"] = True
        self._save_state(state)
        return True, "Validation marked as passed. Transition to done is now allowed."

    def set_memory_item(
        self,
        layer: str,
        key: str,
        value: str,
        branch_id: str | None = None,
    ) -> tuple[bool, str]:
        memory_key = self._normalize_memory_key(key)
        memory_value = value.strip()[:200]
        if not memory_key:
            return False, "Memory key is empty."
        if not memory_value:
            return False, "Memory value is empty."
        state = self._load_state()
        if layer == "working":
            branch = self._get_branch(state, branch_id or state["active_branch"])
            branch["working_memory"][memory_key] = memory_value
            self._trim_memory(branch["working_memory"], 40)
            self._save_state(state)
            return True, f"Saved to working memory: {memory_key}"
        if layer == "long_term":
            state["long_term_memory"][memory_key] = memory_value
            self._trim_memory(state["long_term_memory"], 80)
            self._save_state(state)
            return True, f"Saved to long-term memory: {memory_key}"
        return False, f"Unknown memory layer: {layer}"

    def delete_memory_item(
        self,
        layer: str,
        key: str,
        branch_id: str | None = None,
    ) -> tuple[bool, str]:
        memory_key = self._normalize_memory_key(key)
        if not memory_key:
            return False, "Memory key is empty."
        state = self._load_state()
        if layer == "working":
            branch = self._get_branch(state, branch_id or state["active_branch"])
            if memory_key not in branch["working_memory"]:
                return False, f"Working memory key not found: {memory_key}"
            branch["working_memory"].pop(memory_key, None)
            self._save_state(state)
            return True, f"Deleted from working memory: {memory_key}"
        if layer == "long_term":
            if memory_key not in state["long_term_memory"]:
                return False, f"Long-term memory key not found: {memory_key}"
            state["long_term_memory"].pop(memory_key, None)
            self._save_state(state)
            return True, f"Deleted from long-term memory: {memory_key}"
        return False, f"Unknown memory layer: {layer}"

    def clear_branch(self, branch_id: str | None = None) -> None:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        state["branches"][active] = self._empty_branch()
        self._save_state(state)

    def clear_all(self) -> None:
        self._save_state(self._default_state())

    def run_chat_preview(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        strategy: str,
        window_n: int,
        branch_id: str | None = None,
        profile_id: str | None = None,
        context_limit_override: int | None = None,
        include_memory_layers: bool = True,
    ) -> AgentResponse:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        selected_profile = profile_id or state["active_profile"]
        profile = self._get_profile(state, selected_profile)
        branch = self._get_branch(state, active)
        task_state = self._ensure_task_state(branch)
        invariants = self._get_enabled_invariants(state)
        working_memory = dict(branch["working_memory"])
        if strategy == "facts":
            self._update_facts(working_memory, user_message)
        full_history = list(branch["messages"])
        effective_history = self._select_history_for_strategy(full_history, strategy, window_n)
        long_term = dict(state["long_term_memory"])
        return self._run_with_context(
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            strategy=strategy,
            branch=active,
            full_history=full_history,
            effective_history=effective_history,
            working_memory=working_memory if include_memory_layers else {},
            long_term_memory=long_term if include_memory_layers else {},
            profile=profile,
            profile_id=selected_profile,
            task_state=task_state,
            invariants=invariants,
            context_limit_override=context_limit_override,
            include_memory_layers=include_memory_layers,
        )

    def run_chat_persistent(
        self,
        user_message: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        strategy: str,
        window_n: int,
        branch_id: str | None = None,
        profile_id: str | None = None,
        context_limit_override: int | None = None,
        include_memory_layers: bool = True,
    ) -> AgentResponse:
        state = self._load_state()
        active = branch_id or state["active_branch"]
        selected_profile = profile_id or state["active_profile"]
        state["active_profile"] = selected_profile
        state["active_branch"] = active
        profile = self._get_profile(state, selected_profile)
        branch = self._get_branch(state, active)
        task_state = self._ensure_task_state(branch)
        invariants = self._get_enabled_invariants(state)
        if strategy == "facts":
            self._update_facts(branch["working_memory"], user_message)
        full_history = list(branch["messages"])
        effective_history = self._select_history_for_strategy(full_history, strategy, window_n)
        response = self._run_with_context(
            user_message=user_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            strategy=strategy,
            branch=active,
            full_history=full_history,
            effective_history=effective_history,
            working_memory=branch["working_memory"] if include_memory_layers else {},
            long_term_memory=state["long_term_memory"] if include_memory_layers else {},
            profile=profile,
            profile_id=selected_profile,
            task_state=task_state,
            invariants=invariants,
            context_limit_override=context_limit_override,
            include_memory_layers=include_memory_layers,
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
                    "working_tokens": response.working_tokens,
                    "long_term_tokens": response.long_term_tokens,
                    "profile_tokens": response.profile_tokens,
                    "invariant_tokens": response.invariant_tokens,
                    "blocked_by_invariants": response.blocked_by_invariants,
                    "invariant_report": response.invariant_report,
                    "task_stage": task_state.get("stage", "planning"),
                    "task_paused": bool(task_state.get("paused", False)),
                    "response_tokens": response.output_tokens,
                    "total_turn_tokens": (
                        response.current_request_tokens
                        + response.history_tokens_effective
                        + response.working_tokens
                        + response.long_term_tokens
                        + response.profile_tokens
                        + response.invariant_tokens
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
        working_memory: dict[str, str],
        long_term_memory: dict[str, str],
        profile: dict[str, str],
        profile_id: str,
        task_state: dict[str, object],
        invariants: list[dict[str, str | bool]],
        context_limit_override: int | None,
        include_memory_layers: bool,
    ) -> AgentResponse:
        violations = self._detect_invariant_violations(user_message, invariants)
        if violations:
            return self._invariant_blocked_response(
                model=model_id,
                strategy=strategy,
                branch=branch,
                current_request_tokens=self._estimate_tokens(user_message),
                full_history_tokens=self._estimate_history_tokens(full_history),
                effective_history_tokens=self._estimate_history_tokens(effective_history),
                working_tokens=self._estimate_tokens(self._memory_to_text(working_memory)),
                long_term_tokens=self._estimate_tokens(self._memory_to_text(long_term_memory)),
                profile_tokens=self._estimate_tokens(self._profile_to_text(profile)),
                context_limit=context_limit_override or self._infer_context_limit_tokens(model_id),
                include_memory_layers=include_memory_layers,
                profile_id=profile_id,
                violations=violations,
            )
        current_request_tokens = self._estimate_tokens(user_message)
        full_history_tokens = self._estimate_history_tokens(full_history)
        effective_history_tokens = self._estimate_history_tokens(effective_history)
        working_text = self._memory_to_text(working_memory)
        long_term_text = self._memory_to_text(long_term_memory)
        profile_text = self._profile_to_text(profile)
        task_state_text = self._task_state_to_text(task_state)
        invariants_text = self._invariants_to_text(invariants)
        working_tokens = self._estimate_tokens(working_text)
        long_term_tokens = self._estimate_tokens(long_term_text)
        profile_tokens = self._estimate_tokens(profile_text)
        task_state_tokens = self._estimate_tokens(task_state_text)
        invariants_tokens = self._estimate_tokens(invariants_text)
        context_tokens = (
            current_request_tokens
            + effective_history_tokens
            + working_tokens
            + long_term_tokens
            + profile_tokens
            + task_state_tokens
            + invariants_tokens
        )
        context_limit = context_limit_override or self._infer_context_limit_tokens(model_id)
        if context_tokens + max_tokens > context_limit:
            return self._overflow_response(
                model=model_id,
                strategy=strategy,
                branch=branch,
                current_request_tokens=current_request_tokens,
                full_history_tokens=full_history_tokens,
                effective_history_tokens=effective_history_tokens,
                working_tokens=working_tokens,
                long_term_tokens=long_term_tokens,
                profile_tokens=profile_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
                include_memory_layers=include_memory_layers,
                profile_id=profile_id,
                invariant_tokens=invariants_tokens,
            )
        prompt = self._build_chat_prompt(
            strategy=strategy,
            history=effective_history,
            user_message=user_message,
            working_memory_text=working_text,
            long_term_memory_text=long_term_text,
            profile_id=profile_id,
            profile_text=profile_text,
            task_state_text=task_state_text,
            invariants_text=invariants_text,
            include_memory_layers=include_memory_layers,
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
                working_tokens=working_tokens,
                long_term_tokens=long_term_tokens,
                profile_tokens=profile_tokens,
                context_tokens=context_tokens,
                context_limit=context_limit,
                include_memory_layers=include_memory_layers,
                profile_id=profile_id,
                invariant_tokens=invariants_tokens,
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
            working_tokens=working_tokens,
            long_term_tokens=long_term_tokens,
            profile_tokens=profile_tokens,
            context_tokens=context_tokens,
            context_limit=context_limit,
            include_memory_layers=include_memory_layers,
            profile_id=profile_id,
            invariant_tokens=invariants_tokens,
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
        working_tokens: int,
        long_term_tokens: int,
        profile_tokens: int,
        context_tokens: int,
        context_limit: int,
        include_memory_layers: bool,
        profile_id: str,
        invariant_tokens: int = 0,
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
            facts_tokens=working_tokens,
            working_tokens=working_tokens,
            long_term_tokens=long_term_tokens,
            profile_tokens=profile_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            include_memory_layers=include_memory_layers,
            profile_id=profile_id,
            overflowed=False,
            invariant_tokens=invariant_tokens,
            blocked_by_invariants=False,
            invariant_report="",
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
        working_tokens: int,
        long_term_tokens: int,
        profile_tokens: int,
        context_tokens: int,
        context_limit: int,
        include_memory_layers: bool,
        profile_id: str,
        invariant_tokens: int = 0,
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
            facts_tokens=working_tokens,
            working_tokens=working_tokens,
            long_term_tokens=long_term_tokens,
            profile_tokens=profile_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            include_memory_layers=include_memory_layers,
            profile_id=profile_id,
            overflowed=False,
            invariant_tokens=invariant_tokens,
            blocked_by_invariants=False,
            invariant_report="",
        )

    def _overflow_response(
        self,
        model: str,
        strategy: str,
        branch: str,
        current_request_tokens: int,
        full_history_tokens: int,
        effective_history_tokens: int,
        working_tokens: int,
        long_term_tokens: int,
        profile_tokens: int,
        context_tokens: int,
        context_limit: int,
        include_memory_layers: bool,
        profile_id: str,
        invariant_tokens: int = 0,
    ) -> AgentResponse:
        return AgentResponse(
            text="Error: estimated context exceeds model limit. Use smaller window or reduce memory.",
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
            facts_tokens=working_tokens,
            working_tokens=working_tokens,
            long_term_tokens=long_term_tokens,
            profile_tokens=profile_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            include_memory_layers=include_memory_layers,
            profile_id=profile_id,
            overflowed=True,
            invariant_tokens=invariant_tokens,
            blocked_by_invariants=False,
            invariant_report="",
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
        working_memory_text: str,
        long_term_memory_text: str,
        profile_id: str,
        profile_text: str,
        task_state_text: str,
        invariants_text: str,
        include_memory_layers: bool,
    ) -> str:
        lines = [
            "You are a helpful assistant in a multi-turn chat.",
            "Respect existing decisions and constraints from context.",
            "You are NOT allowed to violate active invariants.",
            "If user request conflicts with invariants, refuse and explain exactly which invariant is violated.",
            "Always include a short 'Invariant Check:' line before your final answer.",
            f"Context strategy: {strategy}",
            f"Memory layers enabled: {'yes' if include_memory_layers else 'no'}",
            f"Active user profile: {profile_id}",
            "Personalization profile:",
            profile_text or "- empty",
            "Task state machine:",
            task_state_text or "- empty",
            "Active invariants:",
            invariants_text or "- none",
        ]
        if include_memory_layers:
            lines.extend(["", "Working memory (current task):", working_memory_text or "- empty"])
            lines.extend(["", "Long-term memory (profile/knowledge):", long_term_memory_text or "- empty"])
        lines.extend(["", "Conversation history (short-term):"])
        for item in history:
            role = item.get("role", "").strip().lower()
            content = item.get("content", "").strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        lines.extend(["", f"User: {user_message.strip()}", "Assistant:"])
        return "\n".join(lines)

    def _invariants_to_text(self, invariants: list[dict[str, str | bool]]) -> str:
        if not invariants:
            return ""
        lines: list[str] = []
        for item in invariants:
            inv_id = str(item.get("id", "")).strip()
            category = str(item.get("category", "general")).strip()
            text = str(item.get("text", "")).strip()
            if inv_id and text:
                lines.append(f"- [{inv_id}] ({category}) {text}")
        return "\n".join(lines)

    def _detect_invariant_violations(
        self,
        user_message: str,
        invariants: list[dict[str, str | bool]],
    ) -> list[dict[str, str]]:
        if not invariants:
            return []
        message = user_message.strip().lower()
        if not message:
            return []
        violations: list[dict[str, str]] = []
        bypass_words = ("игнорируй", "обойди", "наруш", "ignore", "bypass", "override")
        if "инвариант" in message and any(word in message for word in bypass_words):
            for item in invariants[:3]:
                violations.append(
                    {
                        "id": str(item.get("id", "")),
                        "category": str(item.get("category", "general")),
                        "text": str(item.get("text", "")),
                    }
                )
            return violations
        for item in invariants:
            text = str(item.get("text", "")).lower()
            if not text:
                continue
            if self._message_conflicts_with_invariant(message, text):
                violations.append(
                    {
                        "id": str(item.get("id", "")),
                        "category": str(item.get("category", "general")),
                        "text": str(item.get("text", "")),
                    }
                )
        return violations[:3]

    def _message_conflicts_with_invariant(self, message: str, invariant_text: str) -> bool:
        tech_words = (
            "python", "flask", "fastapi", "django", "node", "nodejs", "express",
            "react", "vue", "angular", "java", "kotlin", "go", "rust",
            "postgres", "mysql", "mongodb", "redis",
        )
        action_words = ("использ", "добав", "перепиш", "мигрир", "use", "add", "rewrite", "migrate")
        if not any(word in message for word in action_words):
            return False
        forbidden_markers = ("запрещ", "нельзя", "не использовать", "без ", "do not use", "forbidden")
        for marker in forbidden_markers:
            if marker in invariant_text:
                for tech in tech_words:
                    if tech in invariant_text and tech in message:
                        return True
        only_markers = ("только", "only", "используем", "стек")
        if any(marker in invariant_text for marker in only_markers):
            allowed = [tech for tech in tech_words if tech in invariant_text]
            requested = [tech for tech in tech_words if tech in message]
            if allowed and requested and any(tech not in allowed for tech in requested):
                return True
        return False

    def _invariant_blocked_response(
        self,
        model: str,
        strategy: str,
        branch: str,
        current_request_tokens: int,
        full_history_tokens: int,
        effective_history_tokens: int,
        working_tokens: int,
        long_term_tokens: int,
        profile_tokens: int,
        context_limit: int,
        include_memory_layers: bool,
        profile_id: str,
        violations: list[dict[str, str]],
    ) -> AgentResponse:
        violation_lines = []
        for item in violations:
            inv_id = item.get("id", "")
            category = item.get("category", "general")
            text = item.get("text", "")
            violation_lines.append(f"- [{inv_id}] ({category}) {text}")
        report = "\n".join(violation_lines) if violation_lines else "- No details."
        text = (
            "Invariant Check: conflict detected.\n"
            "I cannot propose this solution because it violates active invariants.\n"
            "Violated invariants:\n"
            f"{report}\n\n"
            "Please rephrase the request so it stays within these constraints."
        )
        invariant_tokens = self._estimate_tokens(report)
        context_tokens = (
            current_request_tokens
            + effective_history_tokens
            + working_tokens
            + long_term_tokens
            + profile_tokens
            + invariant_tokens
        )
        return AgentResponse(
            text=text,
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
            facts_tokens=working_tokens,
            working_tokens=working_tokens,
            long_term_tokens=long_term_tokens,
            profile_tokens=profile_tokens,
            context_tokens_estimate=context_tokens,
            context_limit_tokens=context_limit,
            include_memory_layers=include_memory_layers,
            profile_id=profile_id,
            overflowed=False,
            invariant_tokens=invariant_tokens,
            blocked_by_invariants=True,
            invariant_report=report,
        )

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
            normalized = self._normalize_memory_key(key)
            if normalized:
                facts[normalized] = value.strip()[:120]
        facts["last_user_intent"] = text[:140]
        self._trim_memory(facts, 40)

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

    def _memory_to_text(self, memory: dict[str, str]) -> str:
        if not memory:
            return ""
        return "\n".join(f"- {key}: {value}" for key, value in memory.items())

    def _profile_to_text(self, profile: dict[str, str]) -> str:
        if not profile:
            return ""
        lines = []
        for key in ("style", "format", "constraints", "preferences"):
            value = profile.get(key, "").strip()
            if value:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _task_state_to_text(self, task_state: dict[str, object]) -> str:
        stage = str(task_state.get("stage", "planning")).strip().lower()
        current_step = str(task_state.get("current_step", "")).strip()
        expected_action = str(task_state.get("expected_action", "")).strip()
        paused = bool(task_state.get("paused", False))
        plan_approved = bool(task_state.get("plan_approved", False))
        validation_passed = bool(task_state.get("validation_passed", False))
        lines = [
            f"- stage: {stage}",
            f"- current_step: {current_step or 'not set'}",
            f"- expected_action: {expected_action or 'not set'}",
            f"- paused: {'yes' if paused else 'no'}",
            f"- plan_approved: {'yes' if plan_approved else 'no'}",
            f"- validation_passed: {'yes' if validation_passed else 'no'}",
            "- workflow: planning -> execution -> validation -> done",
            "- transition rule: no stage jumps; only one-step forward transitions",
            "- gate rule: planning->execution requires plan_approved=yes",
            "- gate rule: validation->done requires validation_passed=yes",
        ]
        return "\n".join(lines)

    def _normalize_memory_key(self, key: str) -> str:
        normalized = key.strip().lower().replace(" ", "_")
        normalized = re.sub(r"[^a-zа-я0-9_]+", "", normalized, flags=re.IGNORECASE)
        return normalized[:40]

    def _trim_memory(self, memory: dict[str, str], limit: int) -> None:
        if len(memory) <= limit:
            return
        keys = list(memory.keys())
        for key in keys[:-limit]:
            memory.pop(key, None)

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
        return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate

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
        return "N/A" if cost is None else f"${cost:.6f}"

    def _friendly_error_text(self, error_text: str) -> str:
        normalized = error_text.lower()
        if "overloaded" in normalized or "(529)" in normalized:
            return "LLM provider is temporarily overloaded (529). Please retry in a few seconds."
        return error_text

    def _default_state(self) -> dict:
        return {
            "active_branch": "main",
            "active_profile": "default",
            "long_term_memory": {},
            "invariants": {},
            "profiles": {"default": self._empty_profile()},
            "branches": {"main": self._empty_branch()},
        }

    def _empty_branch(self) -> dict:
        return {
            "messages": [],
            "working_memory": {},
            "checkpoints": [],
            "task_state": self._default_task_state(),
        }

    def _empty_profile(self) -> dict[str, str]:
        return {"style": "", "format": "", "constraints": "", "preferences": ""}

    def _default_task_state(self) -> dict[str, object]:
        return {
            "stage": "planning",
            "current_step": "Collect requirements and outline plan.",
            "expected_action": "Provide or confirm plan details.",
            "paused": False,
            "plan_approved": False,
            "validation_passed": False,
        }

    def _stage_flow(self) -> tuple[str, ...]:
        return ("planning", "execution", "validation", "done")

    def _can_transition(self, task_state: dict[str, object], current: str, target: str) -> tuple[bool, str]:
        if target == current:
            return True, "No stage change."
        if bool(task_state.get("paused", False)):
            return False, "Cannot change stage while task is paused. Resume first."
        flow = self._stage_flow()
        if current not in flow or target not in flow:
            return False, f"Unsupported transition: {current} -> {target}"
        current_idx = flow.index(current)
        target_idx = flow.index(target)
        if target_idx != current_idx + 1:
            return False, f"Stage jump is not allowed: {current} -> {target}"
        if current == "planning" and target == "execution":
            if not bool(task_state.get("plan_approved", False)):
                return False, "Transition blocked: approve plan before execution."
        if current == "validation" and target == "done":
            if not bool(task_state.get("validation_passed", False)):
                return False, "Transition blocked: mark validation as passed before done."
        return True, "Transition allowed."

    def _apply_stage_defaults(self, task_state: dict[str, object], stage: str) -> None:
        if stage == "execution":
            task_state["expected_action"] = "Implement planned solution."
            task_state["current_step"] = "Start implementation based on approved plan."
        elif stage == "validation":
            task_state["expected_action"] = "Validate output and compare with requirements."
            task_state["current_step"] = "Run checks and review implementation quality."
            task_state["validation_passed"] = False
        elif stage == "done":
            task_state["expected_action"] = "Finalize result and summarize."
            task_state["current_step"] = "Prepare final summary and handoff."

    def _normalize_stage(self, stage: str) -> str:
        normalized = stage.strip().lower()
        return normalized if normalized in self._stage_flow() else ""

    def _normalize_task_state(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, dict):
            return self._default_task_state()
        stage = self._normalize_stage(str(raw.get("stage", "planning"))) or "planning"
        current_step = str(raw.get("current_step", "")).strip()[:220]
        expected_action = str(raw.get("expected_action", "")).strip()[:220]
        paused = bool(raw.get("paused", False))
        plan_approved = bool(raw.get("plan_approved", False))
        validation_passed = bool(raw.get("validation_passed", False))
        if stage in {"execution", "validation", "done"}:
            plan_approved = True
        if stage == "done":
            validation_passed = True
        if not current_step:
            current_step = self._default_task_state()["current_step"]
        if not expected_action:
            expected_action = self._default_task_state()["expected_action"]
        return {
            "stage": stage,
            "current_step": current_step,
            "expected_action": expected_action,
            "paused": paused,
            "plan_approved": plan_approved,
            "validation_passed": validation_passed,
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
        active_profile = str(parsed.get("active_profile", "default")).strip() or "default"
        branches = parsed.get("branches", {})
        if not isinstance(branches, dict):
            return self._default_state()
        long_term_memory = self._normalize_memory_map(parsed.get("long_term_memory", {}), 80)
        profiles = self._normalize_profiles(parsed.get("profiles", {}))
        invariants = self._normalize_invariants(parsed.get("invariants", {}))
        normalized_branches: dict[str, dict] = {}
        for name, branch in branches.items():
            if not isinstance(name, str) or not isinstance(branch, dict):
                continue
            messages = self._normalize_messages(branch.get("messages", []))
            legacy_facts = self._normalize_memory_map(branch.get("facts", {}), 40)
            working_memory = self._normalize_memory_map(
                branch.get("working_memory", branch.get("facts", {})),
                40,
            )
            if not working_memory and legacy_facts:
                working_memory = legacy_facts
            checkpoints = self._normalize_checkpoints(branch.get("checkpoints", []), len(messages))
            task_state = self._normalize_task_state(branch.get("task_state", {}))
            normalized_branches[name] = {
                "messages": messages[-220:],
                "working_memory": working_memory,
                "checkpoints": checkpoints[-40:],
                "task_state": task_state,
            }
        if not normalized_branches:
            return self._default_state()
        if active not in normalized_branches:
            active = sorted(normalized_branches.keys())[0]
        if active_profile not in profiles:
            active_profile = "default"
        return {
            "active_branch": active,
            "active_profile": active_profile,
            "long_term_memory": long_term_memory,
            "invariants": invariants,
            "profiles": profiles,
            "branches": normalized_branches,
        }

    def _save_state(self, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        self.state_path.write_text(payload, encoding="utf-8")

    def _get_branch(self, state: dict, branch_id: str) -> dict:
        if branch_id not in state["branches"]:
            state["branches"][branch_id] = self._empty_branch()
        branch = state["branches"][branch_id]
        self._ensure_task_state(branch)
        return branch

    def _get_profile(self, state: dict, profile_id: str) -> dict[str, str]:
        if profile_id not in state["profiles"]:
            state["profiles"][profile_id] = self._empty_profile()
        profile = state["profiles"][profile_id]
        return profile if isinstance(profile, dict) else self._empty_profile()

    def _get_enabled_invariants(self, state: dict) -> list[dict[str, str | bool]]:
        raw = state.get("invariants", {})
        if not isinstance(raw, dict):
            return []
        rows: list[dict[str, str | bool]] = []
        for inv_id in sorted(raw.keys()):
            item = raw.get(inv_id, {})
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "id": inv_id,
                    "category": str(item.get("category", "general")).strip() or "general",
                    "text": text,
                    "enabled": True,
                }
            )
        return rows

    def _ensure_task_state(self, branch: dict) -> dict[str, object]:
        task_state = self._normalize_task_state(branch.get("task_state", {}))
        branch["task_state"] = task_state
        return task_state

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

    def _normalize_memory_map(self, raw: object, limit: int) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            normalized_key = self._normalize_memory_key(str(key))
            normalized_value = str(value).strip()[:200]
            if normalized_key and normalized_value:
                out[normalized_key] = normalized_value
        self._trim_memory(out, limit)
        return out

    def _normalize_profiles(self, raw: object) -> dict[str, dict[str, str]]:
        if not isinstance(raw, dict):
            return {"default": self._empty_profile()}
        normalized: dict[str, dict[str, str]] = {}
        for profile_id, profile in raw.items():
            pid = self._normalize_memory_key(str(profile_id))
            if not pid:
                continue
            if isinstance(profile, dict):
                normalized[pid] = {
                    "style": str(profile.get("style", "")).strip()[:220],
                    "format": str(profile.get("format", "")).strip()[:220],
                    "constraints": str(profile.get("constraints", "")).strip()[:220],
                    "preferences": str(profile.get("preferences", "")).strip()[:220],
                }
        if "default" not in normalized:
            normalized["default"] = self._empty_profile()
        return normalized

    def _normalize_invariants(self, raw: object) -> dict[str, dict[str, str | bool]]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, dict[str, str | bool]] = {}
        for inv_id, invariant in raw.items():
            iid = self._normalize_memory_key(str(inv_id))
            if not iid or not isinstance(invariant, dict):
                continue
            text = str(invariant.get("text", "")).strip()[:240]
            if not text:
                continue
            normalized[iid] = {
                "category": str(invariant.get("category", "general")).strip()[:40] or "general",
                "text": text,
                "enabled": bool(invariant.get("enabled", True)),
            }
        return normalized

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
            if cid:
                out.append({"id": cid, "label": label[:60], "message_index": idx})
        return out
