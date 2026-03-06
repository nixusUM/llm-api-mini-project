import os

from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import get_available_models
from anthropic_client import get_model_override
from llm_agent import LLMAgent

app = Flask(__name__)
agent = LLMAgent()
DEFAULT_MODEL_ID = "claude-3-haiku-20240307"
STRATEGIES = ("sliding", "facts", "branching")


def parse_temperature(raw_value: str, fallback: float) -> float:
    try:
        return float(raw_value)
    except ValueError:
        return fallback


def parse_max_tokens(raw_value: str, fallback: int) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        return fallback
    return max(100, min(value, 2000))


def parse_context_limit(raw_value: str, fallback: int) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        return fallback
    return max(200, min(value, 500000))


def parse_window(raw_value: str, fallback: int) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        return fallback
    return max(2, min(value, 60))


def as_result_view(response) -> dict:
    return {
        "text": response.text,
        "used_model": response.used_model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "latency_ms": response.latency_ms,
        "cost_text": response.cost_text,
        "strategy": response.strategy,
        "branch": response.branch,
        "current_request_tokens": response.current_request_tokens,
        "history_tokens_full": response.history_tokens_full,
        "history_tokens_effective": response.history_tokens_effective,
        "facts_tokens": response.facts_tokens,
        "working_tokens": response.working_tokens,
        "long_term_tokens": response.long_term_tokens,
        "profile_tokens": response.profile_tokens,
        "context_tokens_estimate": response.context_tokens_estimate,
        "context_limit_tokens": response.context_limit_tokens,
        "include_memory_layers": response.include_memory_layers,
        "profile_id": response.profile_id,
        "overflowed": response.overflowed,
        "invariant_tokens": response.invariant_tokens,
        "blocked_by_invariants": response.blocked_by_invariants,
        "invariant_report": response.invariant_report,
    }


def build_token_growth(history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    turn = 1
    cumulative = 0
    for item in history:
        if item.get("role") != "assistant":
            continue
        meta = item.get("meta", {})
        if not isinstance(meta, dict):
            continue
        req = int(meta.get("current_request_tokens", 0) or 0)
        full_hist = int(meta.get("history_tokens_full", 0) or 0)
        eff_hist = int(meta.get("history_tokens_effective", 0) or 0)
        facts_tokens = int(meta.get("facts_tokens", 0) or 0)
        working_tokens = int(meta.get("working_tokens", facts_tokens) or 0)
        long_term_tokens = int(meta.get("long_term_tokens", 0) or 0)
        profile_tokens = int(meta.get("profile_tokens", 0) or 0)
        resp = int(meta.get("response_tokens", 0) or 0)
        total_turn = int(
            meta.get("total_turn_tokens", req + eff_hist + working_tokens + long_term_tokens + profile_tokens + resp)
        )
        cumulative += total_turn
        rows.append(
            {
                "turn": turn,
                "strategy": str(meta.get("strategy", "")),
                "task_stage": str(meta.get("task_stage", "")),
                "task_paused": bool(meta.get("task_paused", False)),
                "req": req,
                "hist_full": full_hist,
                "hist_effective": eff_hist,
                "working_tokens": working_tokens,
                "long_term_tokens": long_term_tokens,
                "profile_tokens": profile_tokens,
                "resp": resp,
                "total": total_turn,
                "cumulative": cumulative,
            }
        )
        turn += 1
    return rows


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    model_options = get_available_models()
    env_model = get_model_override()
    if env_model:
        default_model = env_model
    elif DEFAULT_MODEL_ID in model_options:
        default_model = DEFAULT_MODEL_ID
    else:
        default_model = model_options[0] if model_options else ""

    selected_model = default_model
    strategy = "sliding"
    temperature = "0.7"
    max_tokens = "600"
    context_limit = "200000"
    window_n = "8"
    include_memory_layers = True
    selected_profile = agent.get_active_profile()
    profiles = agent.list_profiles()
    profile_id = selected_profile
    profile_style = ""
    profile_format = ""
    profile_constraints = ""
    profile_preferences = ""
    compare_profile_a = selected_profile
    compare_profile_b = selected_profile
    task_stage = "planning"
    task_step = ""
    expected_action = ""
    memory_layer = "working"
    memory_key = ""
    memory_value = ""
    invariant_id = ""
    invariant_category = "general"
    invariant_text = ""
    checkpoint_label = ""
    new_branch_name = ""
    source_checkpoint_id = ""
    status = ""
    result = {}
    compare_result = {}
    compare_memory_result = {}
    compare_profiles_result = {}

    active_branch = agent.get_active_branch()
    selected_branch = active_branch
    branches = agent.list_branches()
    history = agent.load_history(active_branch)
    short_term_memory = agent.short_term_memory(parse_window(window_n, 8), active_branch)
    working_memory = agent.load_working_memory(active_branch)
    long_term_memory = agent.load_long_term_memory()
    invariants = agent.load_invariants()
    active_profile_data = agent.load_profile(selected_profile)
    profile_style = active_profile_data.get("style", "")
    profile_format = active_profile_data.get("format", "")
    profile_constraints = active_profile_data.get("constraints", "")
    profile_preferences = active_profile_data.get("preferences", "")
    checkpoints = agent.list_checkpoints(active_branch)
    task_state = agent.load_task_state(active_branch)
    task_stage = str(task_state.get("stage", "planning"))
    task_step = str(task_state.get("current_step", ""))
    expected_action = str(task_state.get("expected_action", ""))
    token_growth = build_token_growth(history)
    state_path = str(agent.state_path)

    if request.method == "POST":
        action = request.form.get("action", "send").strip().lower()
        prompt = request.form.get("prompt", "").strip()
        selected_model = request.form.get("selected_model", default_model).strip()
        if model_options and selected_model not in model_options:
            selected_model = default_model
        strategy = request.form.get("strategy", "sliding").strip().lower()
        if strategy not in STRATEGIES:
            strategy = "sliding"
        temperature = request.form.get("temperature", "0.7").strip()
        max_tokens = request.form.get("max_tokens", "600").strip()
        context_limit = request.form.get("context_limit", "200000").strip()
        window_n = request.form.get("window_n", "8").strip()
        include_memory_layers = request.form.get("include_memory_layers", "") == "on"
        selected_profile = request.form.get("selected_profile", selected_profile).strip() or selected_profile
        profile_id = request.form.get("profile_id", selected_profile).strip()
        profile_style = request.form.get("profile_style", "").strip()
        profile_format = request.form.get("profile_format", "").strip()
        profile_constraints = request.form.get("profile_constraints", "").strip()
        profile_preferences = request.form.get("profile_preferences", "").strip()
        compare_profile_a = request.form.get("compare_profile_a", selected_profile).strip() or selected_profile
        compare_profile_b = request.form.get("compare_profile_b", selected_profile).strip() or selected_profile
        task_stage = request.form.get("task_stage", task_stage).strip().lower()
        task_step = request.form.get("task_step", task_step).strip()
        expected_action = request.form.get("expected_action", expected_action).strip()
        memory_layer = request.form.get("memory_layer", "working").strip().lower()
        memory_key = request.form.get("memory_key", "").strip()
        memory_value = request.form.get("memory_value", "").strip()
        invariant_id = request.form.get("invariant_id", "").strip()
        invariant_category = request.form.get("invariant_category", "general").strip().lower()
        invariant_text = request.form.get("invariant_text", "").strip()
        checkpoint_label = request.form.get("checkpoint_label", "").strip()
        new_branch_name = request.form.get("new_branch_name", "").strip()
        source_checkpoint_id = request.form.get("source_checkpoint_id", "").strip()
        selected_branch = request.form.get("selected_branch", active_branch).strip() or active_branch

        parsed_temp = parse_temperature(temperature, 0.7)
        parsed_max_tokens = parse_max_tokens(max_tokens, 600)
        parsed_context_limit = parse_context_limit(context_limit, 200000)
        parsed_window = parse_window(window_n, 8)
        available_profiles = agent.list_profiles()
        if selected_profile not in available_profiles:
            selected_profile = agent.get_active_profile()
        if compare_profile_a not in available_profiles:
            compare_profile_a = selected_profile
        if compare_profile_b not in available_profiles:
            compare_profile_b = selected_profile

        if action == "clear_all":
            agent.clear_all()
            status = "All branches and history cleared."
        elif action == "set_task_state":
            ok, message = agent.set_task_state(
                stage=task_stage,
                current_step=task_step,
                expected_action=expected_action,
                branch_id=selected_branch,
            )
            status = message if ok else f"Set task state failed: {message}"
        elif action == "advance_task_stage":
            ok, message = agent.advance_task_stage(selected_branch)
            status = message if ok else f"Advance stage failed: {message}"
        elif action == "approve_plan":
            ok, message = agent.approve_plan(selected_branch)
            status = message if ok else f"Approve plan failed: {message}"
        elif action == "pass_validation":
            ok, message = agent.pass_validation(selected_branch)
            status = message if ok else f"Pass validation failed: {message}"
        elif action == "pause_task":
            ok, message = agent.pause_task(selected_branch)
            status = message if ok else f"Pause failed: {message}"
        elif action == "resume_task":
            ok, message = agent.resume_task(selected_branch)
            status = message if ok else f"Resume failed: {message}"
        elif action == "save_profile":
            ok, message = agent.save_profile(
                profile_id=profile_id,
                style=profile_style,
                output_format=profile_format,
                constraints=profile_constraints,
                preferences=profile_preferences,
            )
            status = message if ok else f"Save profile failed: {message}"
        elif action == "switch_profile":
            if agent.switch_profile(selected_profile):
                status = f"Switched profile: {selected_profile}"
            else:
                status = f"Profile not found: {selected_profile}"
        elif action == "delete_profile":
            ok, message = agent.delete_profile(selected_profile)
            status = message if ok else f"Delete profile failed: {message}"
        elif action == "save_memory":
            ok, message = agent.set_memory_item(
                layer=memory_layer,
                key=memory_key,
                value=memory_value,
                branch_id=selected_branch,
            )
            status = message if ok else f"Save memory failed: {message}"
        elif action == "delete_memory":
            ok, message = agent.delete_memory_item(
                layer=memory_layer,
                key=memory_key,
                branch_id=selected_branch,
            )
            status = message if ok else f"Delete memory failed: {message}"
        elif action == "save_invariant":
            ok, message = agent.save_invariant(
                invariant_id=invariant_id,
                category=invariant_category,
                text=invariant_text,
                enabled=True,
            )
            status = message if ok else f"Save invariant failed: {message}"
        elif action == "delete_invariant":
            ok, message = agent.delete_invariant(invariant_id)
            status = message if ok else f"Delete invariant failed: {message}"
        elif action == "switch_branch":
            if agent.switch_branch(selected_branch):
                status = f"Switched to branch: {selected_branch}"
            else:
                status = f"Branch not found: {selected_branch}"
        elif action == "create_checkpoint":
            checkpoint_id = agent.create_checkpoint(checkpoint_label, selected_branch)
            status = f"Checkpoint created: {checkpoint_id}"
        elif action == "create_branch":
            ok, message = agent.create_branch_from_checkpoint(
                source_branch=selected_branch,
                checkpoint_id=source_checkpoint_id,
                new_branch=new_branch_name,
            )
            status = message if ok else f"Create branch failed: {message}"
        elif action == "compare_strategies":
            if prompt:
                compared: dict[str, dict] = {}
                for strat in STRATEGIES:
                    response = agent.run_chat_persistent(
                        user_message=prompt,
                        model_id=selected_model,
                        temperature=parsed_temp,
                        max_tokens=parsed_max_tokens,
                        strategy=strat,
                        window_n=parsed_window,
                        branch_id=selected_branch,
                        profile_id=selected_profile,
                        context_limit_override=parsed_context_limit,
                        include_memory_layers=include_memory_layers,
                    )
                    compared[strat] = as_result_view(response)
                compare_result = compared
                status = "Compared all strategies on the same prompt."
                prompt = ""
            else:
                status = "Prompt is empty."
        elif action == "compare_memory":
            if prompt:
                with_memory = agent.run_chat_preview(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    strategy=strategy,
                    window_n=parsed_window,
                    branch_id=selected_branch,
                    profile_id=selected_profile,
                    context_limit_override=parsed_context_limit,
                    include_memory_layers=True,
                )
                without_memory = agent.run_chat_preview(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    strategy=strategy,
                    window_n=parsed_window,
                    branch_id=selected_branch,
                    profile_id=selected_profile,
                    context_limit_override=parsed_context_limit,
                    include_memory_layers=False,
                )
                compare_memory_result = {
                    "with_memory": as_result_view(with_memory),
                    "without_memory": as_result_view(without_memory),
                }
                status = "Compared the same prompt with memory layers ON/OFF."
            else:
                status = "Prompt is empty."
        elif action == "compare_profiles":
            if prompt:
                profile_a_resp = agent.run_chat_preview(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    strategy=strategy,
                    window_n=parsed_window,
                    branch_id=selected_branch,
                    profile_id=compare_profile_a,
                    context_limit_override=parsed_context_limit,
                    include_memory_layers=include_memory_layers,
                )
                profile_b_resp = agent.run_chat_preview(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    strategy=strategy,
                    window_n=parsed_window,
                    branch_id=selected_branch,
                    profile_id=compare_profile_b,
                    context_limit_override=parsed_context_limit,
                    include_memory_layers=include_memory_layers,
                )
                compare_profiles_result = {
                    compare_profile_a: as_result_view(profile_a_resp),
                    compare_profile_b: as_result_view(profile_b_resp),
                }
                status = "Compared answers for two profiles."
            else:
                status = "Prompt is empty."
        elif action == "send":
            if prompt:
                response = agent.run_chat_persistent(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    strategy=strategy,
                    window_n=parsed_window,
                    branch_id=selected_branch,
                    profile_id=selected_profile,
                    context_limit_override=parsed_context_limit,
                    include_memory_layers=include_memory_layers,
                )
                result = as_result_view(response)
                prompt = ""
            else:
                status = "Prompt is empty."

        active_branch = agent.get_active_branch()
        selected_profile = agent.get_active_profile()
        profiles = agent.list_profiles()
        branches = agent.list_branches()
        history = agent.load_history(active_branch)
        short_term_memory = agent.short_term_memory(parsed_window, active_branch)
        working_memory = agent.load_working_memory(active_branch)
        long_term_memory = agent.load_long_term_memory()
        invariants = agent.load_invariants()
        active_profile_data = agent.load_profile(selected_profile)
        profile_style = active_profile_data.get("style", "")
        profile_format = active_profile_data.get("format", "")
        profile_constraints = active_profile_data.get("constraints", "")
        profile_preferences = active_profile_data.get("preferences", "")
        checkpoints = agent.list_checkpoints(active_branch)
        task_state = agent.load_task_state(active_branch)
        task_stage = str(task_state.get("stage", "planning"))
        task_step = str(task_state.get("current_step", ""))
        expected_action = str(task_state.get("expected_action", ""))
        token_growth = build_token_growth(history)
        if active_branch not in branches and branches:
            active_branch = branches[0]

    return render_template(
        "index_modern.html",
        prompt=prompt,
        model_options=model_options,
        selected_model=selected_model,
        strategy=strategy,
        strategies=STRATEGIES,
        temperature=temperature,
        max_tokens=max_tokens,
        context_limit=context_limit,
        window_n=window_n,
        include_memory_layers=include_memory_layers,
        selected_profile=selected_profile,
        profiles=profiles,
        profile_id=profile_id,
        profile_style=profile_style,
        profile_format=profile_format,
        profile_constraints=profile_constraints,
        profile_preferences=profile_preferences,
        compare_profile_a=compare_profile_a,
        compare_profile_b=compare_profile_b,
        task_state=task_state,
        task_stage=task_stage,
        task_step=task_step,
        expected_action=expected_action,
        memory_layer=memory_layer,
        memory_key=memory_key,
        memory_value=memory_value,
        invariant_id=invariant_id,
        invariant_category=invariant_category,
        invariant_text=invariant_text,
        checkpoint_label=checkpoint_label,
        new_branch_name=new_branch_name,
        source_checkpoint_id=source_checkpoint_id,
        status=status,
        result=result,
        compare_result=compare_result,
        compare_memory_result=compare_memory_result,
        compare_profiles_result=compare_profiles_result,
        active_branch=active_branch,
        selected_branch=selected_branch,
        branches=branches,
        short_term_memory=short_term_memory,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        invariants=invariants,
        checkpoints=checkpoints,
        history=history,
        token_growth=token_growth,
        state_path=state_path,
    )


if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", "5051"))
    app.run(debug=True, host="127.0.0.1", port=port)
