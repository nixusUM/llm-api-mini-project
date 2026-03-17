import os
import json
import threading
from datetime import datetime
from datetime import timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request

from anthropic_client import get_available_models
from anthropic_client import get_model_override
from llm_agent import LLMAgent
from mcp_list_tools import call_mcp_tool_sync
from mcp_list_tools import list_mcp_tools_sync
from mcp_orchestrator import get_tool_to_server_map
from mcp_orchestrator import run_long_flow
from mcp_orchestrator import SERVERS
from rag_service import RAGService

app = Flask(__name__)
agent = LLMAgent()
rag_service = RAGService()
DEFAULT_MODEL_ID = "claude-3-haiku-20240307"
STRATEGIES = ("sliding", "facts", "branching")
SCHEDULER_LOCK = threading.Lock()
SCHEDULER_STOP_EVENT = threading.Event()
SCHEDULER_THREAD: threading.Thread | None = None
SCHEDULER_STATE = {
    "running": False,
    "poll_seconds": 30,
    "last_tick_at": "",
    "last_status": "idle",
    "last_report": "",
}


def _now_utc_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _scheduler_loop(poll_seconds: int) -> None:
    interval = max(5, poll_seconds)
    while not SCHEDULER_STOP_EVENT.is_set():
        try:
            run_due = call_mcp_tool_sync("run_due_summaries", {})
            report = call_mcp_tool_sync("get_summary_report", {"limit": 5})
            report_text = str(report.get("structured", report.get("text", "")))
            status_text = "ok" if run_due.get("ok") else "error"
        except Exception as exc:
            report_text = f"Scheduler error: {exc}"
            status_text = "error"
        with SCHEDULER_LOCK:
            SCHEDULER_STATE["last_tick_at"] = _now_utc_text()
            SCHEDULER_STATE["last_status"] = status_text
            SCHEDULER_STATE["last_report"] = report_text[:1200]
        if SCHEDULER_STOP_EVENT.wait(interval):
            break


def start_scheduler(poll_seconds: int) -> str:
    global SCHEDULER_THREAD
    with SCHEDULER_LOCK:
        if SCHEDULER_STATE["running"]:
            return "Scheduler is already running."
        SCHEDULER_STOP_EVENT.clear()
        SCHEDULER_STATE["running"] = True
        SCHEDULER_STATE["poll_seconds"] = max(5, poll_seconds)
        SCHEDULER_STATE["last_status"] = "starting"
        thread = threading.Thread(
            target=_scheduler_loop,
            args=(SCHEDULER_STATE["poll_seconds"],),
            daemon=True,
        )
        SCHEDULER_THREAD = thread
        thread.start()
    return f"Scheduler started (every {SCHEDULER_STATE['poll_seconds']}s)."


def stop_scheduler() -> str:
    global SCHEDULER_THREAD
    with SCHEDULER_LOCK:
        if not SCHEDULER_STATE["running"]:
            return "Scheduler is not running."
        SCHEDULER_STATE["running"] = False
        SCHEDULER_STATE["last_status"] = "stopped"
        SCHEDULER_STOP_EVENT.set()
        SCHEDULER_THREAD = None
    return "Scheduler stopped."


def scheduler_snapshot() -> dict[str, object]:
    with SCHEDULER_LOCK:
        return dict(SCHEDULER_STATE)


def get_periodic_summary_text(limit: int = 5) -> str:
    try:
        report = call_mcp_tool_sync(tool_name="get_summary_report", arguments={"limit": limit})
    except Exception as exc:
        return f"Scheduler summary read failed: {exc}"
    return str(report.get("structured", report.get("text", "")))


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


def _format_mcp_extra_result(tool_result: dict, tool_name: str) -> str:
    if tool_result.get("structured"):
        return json.dumps(tool_result["structured"], ensure_ascii=False, indent=2)
    text = (tool_result.get("text") or "").strip()
    return text or f"{tool_name}: no output"


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
    assignment_mode = request.args.get("full", "0") != "1"
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
    mcp_status = ""
    mcp_tools: list[dict[str, str]] = []
    public_mcp_status = ""
    public_mcp_tools: list[dict[str, str]] = []
    mcp_todo_id = "1"
    mcp_tool_result = ""
    weather_city = "London"
    rate_from = "USD"
    rate_to = "EUR"
    mcp_extra_result = ""
    periodic_job_id = "todo_summary_main"
    periodic_interval_seconds = "60"
    periodic_user_id = "1"
    scheduler_poll_seconds = "30"
    scheduler_state = scheduler_snapshot()
    periodic_summary_result = str(scheduler_state.get("last_report", ""))
    pipeline_query = "qui"
    pipeline_limit = "5"
    pipeline_file_name = "pipeline_summary.txt"
    pipeline_result = ""
    orchestration_query = "qui"
    orchestration_limit = "5"
    orchestration_file_name = "orchestration_summary.txt"
    orchestration_result = ""
    tool_to_server_map: dict[str, str] = {}
    try:
        tool_to_server_map = get_tool_to_server_map()
    except Exception:
        pass

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
        mcp_todo_id = request.form.get("mcp_todo_id", "1").strip() or "1"
        weather_city = request.form.get("weather_city", weather_city).strip()
        rate_from = request.form.get("rate_from", rate_from).strip()
        rate_to = request.form.get("rate_to", rate_to).strip()
        periodic_job_id = request.form.get("periodic_job_id", periodic_job_id).strip() or periodic_job_id
        periodic_interval_seconds = request.form.get("periodic_interval_seconds", periodic_interval_seconds).strip()
        periodic_user_id = request.form.get("periodic_user_id", periodic_user_id).strip()
        scheduler_poll_seconds = request.form.get("scheduler_poll_seconds", scheduler_poll_seconds).strip()
        pipeline_query = request.form.get("pipeline_query", pipeline_query).strip()
        pipeline_limit = request.form.get("pipeline_limit", pipeline_limit).strip()
        pipeline_file_name = request.form.get("pipeline_file_name", pipeline_file_name).strip()
        orchestration_query = request.form.get("orchestration_query", orchestration_query).strip()
        orchestration_limit = request.form.get("orchestration_limit", orchestration_limit).strip()
        orchestration_file_name = request.form.get("orchestration_file_name", orchestration_file_name).strip()
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
            stop_scheduler()
            agent.clear_all()
            try:
                call_mcp_tool_sync(tool_name="clear_periodic_state", arguments={})
            except Exception:
                pass
            with SCHEDULER_LOCK:
                SCHEDULER_STATE["running"] = False
                SCHEDULER_STATE["last_tick_at"] = ""
                SCHEDULER_STATE["last_status"] = "cleared"
                SCHEDULER_STATE["last_report"] = ""
            periodic_summary_result = ""
            status = "All branches/history and scheduler state cleared."
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
        elif action == "test_mcp":
            try:
                tools = list_mcp_tools_sync()
                mcp_tools = [{"name": name, "description": description} for name, description in tools]
                mcp_status = "MCP connection: OK"
                status = f"MCP tools loaded: {len(mcp_tools)}"
            except Exception as exc:
                mcp_status = f"MCP connection failed: {exc}"
                status = mcp_status
        elif action == "test_public_mcp":
            try:
                tools = list_mcp_tools_sync(["npx", "-y", "@modelcontextprotocol/server-everything"])
                public_mcp_tools = [{"name": name, "description": description} for name, description in tools]
                public_mcp_status = "Public MCP connection: OK"
                status = f"Public MCP tools loaded: {len(public_mcp_tools)}"
            except Exception as exc:
                public_mcp_status = f"Public MCP connection failed: {exc}"
                status = public_mcp_status
        elif action == "run_mcp_todo_tool":
            try:
                todo_id = int(mcp_todo_id)
            except ValueError:
                todo_id = 1
            tool_result = call_mcp_tool_sync(
                tool_name="get_todo_from_mock_api",
                arguments={"todo_id": todo_id},
            )
            mcp_tool_result = str(tool_result.get("text", "")).strip()
            if not mcp_tool_result and tool_result.get("structured"):
                mcp_tool_result = str(tool_result.get("structured"))
            if tool_result.get("ok"):
                status = "MCP tool get_todo_from_mock_api executed successfully."
                if mcp_tool_result:
                    prompt = (
                        "Use this MCP tool result as factual input:\n"
                        f"{mcp_tool_result}\n\n"
                        "Summarize key details in 3 bullet points."
                    )
            else:
                status = "MCP tool execution failed."
        elif action == "run_weather":
            tool_result = call_mcp_tool_sync(
                tool_name="get_weather",
                arguments={"city": weather_city or "London", "units": "celsius"},
            )
            mcp_extra_result = _format_mcp_extra_result(tool_result, "get_weather")
            status = "Weather fetched." if tool_result.get("ok") else "Weather request failed."
            if tool_result.get("ok") and tool_result.get("structured"):
                prompt = (
                    "Use this weather data and suggest what to wear or do:\n"
                    f"{mcp_extra_result}"
                )
        elif action == "run_exchange_rate":
            tool_result = call_mcp_tool_sync(
                tool_name="get_exchange_rate",
                arguments={"from_currency": rate_from or "USD", "to_currency": rate_to or "EUR"},
            )
            mcp_extra_result = _format_mcp_extra_result(tool_result, "get_exchange_rate")
            status = "Exchange rate fetched." if tool_result.get("ok") else "Rate request failed."
        elif action == "run_quote":
            tool_result = call_mcp_tool_sync(tool_name="get_random_quote", arguments={})
            mcp_extra_result = _format_mcp_extra_result(tool_result, "get_random_quote")
            status = "Quote fetched." if tool_result.get("ok") else "Quote request failed."
            if tool_result.get("ok") and tool_result.get("structured"):
                prompt = (
                    "Here is a random quote. Comment on it in one sentence:\n"
                    f"{mcp_extra_result}"
                )
        elif action == "configure_periodic_summary":
            try:
                interval_int = int(periodic_interval_seconds)
            except ValueError:
                interval_int = 60
            try:
                user_int = int(periodic_user_id)
            except ValueError:
                user_int = 1
            result_obj = call_mcp_tool_sync(
                tool_name="configure_periodic_summary",
                arguments={
                    "job_id": periodic_job_id,
                    "interval_seconds": interval_int,
                    "user_id": user_int,
                    "enabled": True,
                },
            )
            periodic_summary_result = str(result_obj.get("structured", result_obj.get("text", "")))
            status = "Periodic summary job configured."
        elif action == "run_periodic_tick":
            run_due = call_mcp_tool_sync(tool_name="run_due_summaries", arguments={})
            report = call_mcp_tool_sync(tool_name="get_summary_report", arguments={"limit": 5})
            periodic_summary_result = str(report.get("structured", report.get("text", "")))
            if run_due.get("ok"):
                status = "Periodic tick executed. Summary updated."
                if periodic_summary_result:
                    prompt = (
                        "Use this periodic summary as background context:\n"
                        f"{periodic_summary_result}\n\n"
                        "Provide a short operational digest."
                    )
            else:
                status = "Periodic tick failed."
        elif action == "start_scheduler":
            try:
                poll_int = int(scheduler_poll_seconds)
            except ValueError:
                poll_int = 30
            status = start_scheduler(poll_int)
        elif action == "stop_scheduler":
            status = stop_scheduler()
        elif action == "refresh_scheduler":
            periodic_summary_result = get_periodic_summary_text(limit=5)
            with SCHEDULER_LOCK:
                SCHEDULER_STATE["last_report"] = periodic_summary_result[:1200]
                SCHEDULER_STATE["last_tick_at"] = _now_utc_text()
                SCHEDULER_STATE["last_status"] = "manual_refresh"
            status = "Scheduler summary refreshed."
        elif action == "run_mcp_pipeline":
            try:
                limit_int = int(pipeline_limit)
            except ValueError:
                limit_int = 5
            search_step = call_mcp_tool_sync(
                tool_name="search_data",
                arguments={"query": pipeline_query, "limit": limit_int},
            )
            search_struct = search_step.get("structured", {})
            if not isinstance(search_struct, dict):
                search_struct = {}
            summarize_step = call_mcp_tool_sync(
                tool_name="summarize_data",
                arguments={"search_payload_json": json.dumps(search_struct, ensure_ascii=False)},
            )
            summarize_struct = summarize_step.get("structured", {})
            if not isinstance(summarize_struct, dict):
                summarize_struct = {}
            summary_text = str(summarize_struct.get("summary_text", "")).strip()
            save_step = call_mcp_tool_sync(
                tool_name="save_to_file",
                arguments={"file_name": pipeline_file_name, "content": summary_text},
            )
            save_struct = save_step.get("structured", {})
            if not isinstance(save_struct, dict):
                save_struct = {}
            pipeline_payload = {
                "ok": bool(search_step.get("ok")) and bool(summarize_step.get("ok")) and bool(save_step.get("ok")),
                "steps": {
                    "search": search_struct,
                    "summarize": summarize_struct,
                    "save_to_file": save_struct,
                },
            }
            pipeline_result = json.dumps(pipeline_payload, ensure_ascii=False, indent=2)
            status = "MCP pipeline executed: search -> summarize -> save_to_file."
            if summary_text:
                prompt = (
                    "Use this MCP pipeline summary:\n"
                    f"{summary_text}\n\n"
                    "Provide 3 concise insights."
                )
        elif action == "run_orchestration_flow":
            try:
                limit_int = int(orchestration_limit)
            except ValueError:
                limit_int = 5
            try:
                steps = run_long_flow(
                    query=orchestration_query or "qui",
                    limit=limit_int,
                    output_file=orchestration_file_name or "orchestration_summary.txt",
                )
                all_ok = all(s.get("result", {}).get("ok", False) for s in steps)
                payload = {
                    "ok": all_ok,
                    "steps": [
                        {
                            "server_id": s["server_id"],
                            "tool_name": s["tool_name"],
                            "arguments": s["arguments"],
                            "ok": s.get("result", {}).get("ok", False),
                            "text": (s.get("result") or {}).get("text", "")[:500],
                        }
                        for s in steps
                    ],
                }
                orchestration_result = json.dumps(payload, ensure_ascii=False, indent=2)
                status = "Orchestration flow ran across local + public MCP servers."
                if all_ok and len(steps) >= 2:
                    sum_result = steps[1].get("result", {})
                    sum_struct = sum_result.get("structured", {})
                    if isinstance(sum_struct, dict):
                        summary_text = str(sum_struct.get("summary_text", "")).strip()
                        if summary_text:
                            prompt = (
                                "Ниже результат длинного флоу оркестрации MCP (поиск → суммаризация → echo → сохранение в файл). "
                                "Используй эти данные по заданию: сформулируй краткие выводы или ответь на вопрос пользователя.\n\n"
                                f"{summary_text}"
                            )
            except Exception as exc:
                orchestration_result = json.dumps(
                    {"ok": False, "error": str(exc), "steps": []},
                    ensure_ascii=False,
                    indent=2,
                )
                status = f"Orchestration flow failed: {exc}"
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
        scheduler_state = scheduler_snapshot()

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
        mcp_status=mcp_status,
        mcp_tools=mcp_tools,
        public_mcp_status=public_mcp_status,
        public_mcp_tools=public_mcp_tools,
        mcp_todo_id=mcp_todo_id,
        mcp_tool_result=mcp_tool_result,
        weather_city=weather_city,
        rate_from=rate_from,
        rate_to=rate_to,
        mcp_extra_result=mcp_extra_result,
        periodic_job_id=periodic_job_id,
        periodic_interval_seconds=periodic_interval_seconds,
        periodic_user_id=periodic_user_id,
        scheduler_poll_seconds=scheduler_poll_seconds,
        periodic_summary_result=periodic_summary_result,
        scheduler_state=scheduler_state,
        pipeline_query=pipeline_query,
        pipeline_limit=pipeline_limit,
        pipeline_file_name=pipeline_file_name,
        pipeline_result=pipeline_result,
        orchestration_query=orchestration_query,
        orchestration_limit=orchestration_limit,
        orchestration_file_name=orchestration_file_name,
        orchestration_result=orchestration_result,
        tool_to_server_map=tool_to_server_map,
        registered_servers=list(SERVERS.keys()),
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
        rag_questions=rag_service.control_questions(),
        assignment_mode=assignment_mode,
    )


@app.route("/scheduler_status", methods=["GET"])
def scheduler_status():
    state = scheduler_snapshot()
    report = str(state.get("last_report", "")).strip()
    if not report:
        report = get_periodic_summary_text(limit=5)
    return jsonify(
        {
            "running": bool(state.get("running", False)),
            "poll_seconds": int(state.get("poll_seconds", 30) or 30),
            "last_tick_at": str(state.get("last_tick_at", "")),
            "last_status": str(state.get("last_status", "")),
            "last_report": report,
        }
    )


@app.route("/api/rag_query", methods=["POST"])
def api_rag_query():
    payload = request.get_json(force=True, silent=True) or {}
    question = str(payload.get("question", "")).strip()
    top_k = int(payload.get("top_k", rag_service.top_k))
    if not question:
        return jsonify({"error": "Question is empty."}), 400
    result = rag_service.answer_question(question, top_k=top_k)
    return jsonify(result)


# Document Indexer routes
@app.route("/document_indexer")
def document_indexer():
    """Start standalone indexer if needed and redirect."""
    import subprocess
    import socket

    indexer_port = int(os.getenv("INDEXER_PORT", "5052"))

    # Check if indexer is running
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_running = sock.connect_ex(("127.0.0.1", indexer_port)) == 0
    sock.close()

    if not is_running:
        subprocess.Popen(
            ["python", "document_indexer_app.py"],
            cwd=os.path.dirname(__file__),
        )
        import time
        time.sleep(2)

    return redirect(f"http://127.0.0.1:{indexer_port}/document_indexer")


if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", "5051"))
    app.run(debug=True, host="127.0.0.1", port=port)
