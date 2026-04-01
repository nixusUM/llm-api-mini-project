import os
import json
import time
import threading
from datetime import datetime
from datetime import timezone
from urllib import error as urlerror
from urllib import request as urlrequest

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
from dev_assistant_rag import (
    build_dev_assistant_local_llm_prompt,
    build_project_context_block,
)
from support_assistant_rag import build_support_assistant_local_prompt
from rag_service import RAGService

app = Flask(__name__)
agent = LLMAgent()
rag_service = RAGService()
DEFAULT_MODEL_ID = "claude-3-5-haiku-latest"
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
LOCAL_LLM_DEFAULT_ENDPOINT = "http://127.0.0.1:8088"
LOCAL_LLM_DEFAULT_MODEL = "qwen-local"


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


def _normalize_local_llm_endpoint(raw_value: str) -> str:
    endpoint = raw_value.strip() or LOCAL_LLM_DEFAULT_ENDPOINT
    return endpoint.rstrip("/")


def _http_get_json(url: str, timeout_sec: float = 8.0) -> dict:
    req = urlrequest.Request(url=url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _local_llm_chat_once(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 180,
    temperature: float = 0.2,
    system_instruction: str | None = None,
) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if system_instruction and system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url=f"{endpoint}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urlrequest.urlopen(req, timeout=20.0) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        choices = parsed.get("choices", [])
        if not choices:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {"ok": False, "error": "No choices in response.", "raw": parsed, "latency_ms": elapsed_ms}
        message = choices[0].get("message", {})
        text = str(message.get("content", "")).strip()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {"ok": True, "text": text, "raw": parsed, "latency_ms": elapsed_ms}
    except urlerror.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        return {"ok": False, "error": f"HTTP {exc.code}: {details}", "latency_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {"ok": False, "error": str(exc), "latency_ms": elapsed_ms}


def run_local_llm_checks(endpoint: str, model: str, prompts: list[str]) -> dict[str, object]:
    health_result: dict[str, object]
    health_started = time.perf_counter()
    try:
        health_payload = _http_get_json(f"{endpoint}/health")
        health_ok = str(health_payload.get("status", "")).lower() == "ok"
        health_latency_ms = int((time.perf_counter() - health_started) * 1000)
        health_result = {
            "ok": health_ok,
            "payload": health_payload,
            "latency_ms": health_latency_ms,
        }
    except Exception as exc:
        health_latency_ms = int((time.perf_counter() - health_started) * 1000)
        health_result = {
            "ok": False,
            "error": str(exc),
            "latency_ms": health_latency_ms,
        }

    prompt_results: list[dict[str, object]] = []
    for idx, prompt in enumerate(prompts, start=1):
        prompt_text = prompt.strip()
        if not prompt_text:
            continue
        chat = _local_llm_chat_once(
            endpoint=endpoint,
            model=model,
            prompt=prompt_text,
        )
        prompt_results.append(
            {
                "id": idx,
                "prompt": prompt_text,
                "ok": bool(chat.get("ok", False)),
                "answer": str(chat.get("text", "")).strip(),
                "error": str(chat.get("error", "")).strip(),
                "latency_ms": int(chat.get("latency_ms", 0) or 0),
            }
        )

    all_prompts_ok = all(item["ok"] for item in prompt_results) if prompt_results else False
    return {
        "endpoint": endpoint,
        "model": model,
        "health": health_result,
        "prompt_results": prompt_results,
        "summary": {
            "total_prompts": len(prompt_results),
            "ok_prompts": sum(1 for item in prompt_results if item["ok"]),
            "all_ok": bool(health_result.get("ok", False)) and all_prompts_ok,
        },
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len((text or "").strip()) // 4)


def _build_prompt_from_template(template: str, prompt: str) -> str:
    raw = (template or "").strip()
    if not raw:
        return prompt
    if "{prompt}" in raw:
        return raw.replace("{prompt}", prompt)
    return f"{raw}\n\n{prompt}"


def _apply_context_limit(prompt: str, context_window: int, max_tokens: int) -> tuple[str, int, bool]:
    budget = max(200, context_window - max_tokens - 64)
    estimated = _estimate_tokens(prompt)
    if estimated <= budget:
        return prompt, estimated, False
    max_chars = max(240, budget * 4)
    trimmed = prompt[:max_chars]
    return trimmed, _estimate_tokens(trimmed), True


def _split_keywords(raw_keywords: str) -> list[str]:
    return [item.strip().lower() for item in (raw_keywords or "").split(",") if item.strip()]


def _keyword_quality(answer: str, expected_keywords: list[str]) -> dict[str, object]:
    if not expected_keywords:
        return {"matched": 0, "total": 0, "score": 1.0, "hits": []}
    low = (answer or "").lower()
    hits = [k for k in expected_keywords if k in low]
    score = len(hits) / max(1, len(expected_keywords))
    return {"matched": len(hits), "total": len(expected_keywords), "score": round(score, 3), "hits": hits}


def _run_local_suite(
    endpoint: str,
    model: str,
    quant_label: str,
    prompts: list[dict[str, object]],
    temperature: float,
    max_tokens: int,
    context_window: int,
    prompt_template: str,
    repeats: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    latencies: list[int] = []
    token_outputs: list[int] = []
    ok_runs = 0
    total_runs = 0
    for item in prompts:
        prompt = str(item.get("prompt", "")).strip()
        expected = list(item.get("expected_keywords", []))
        if not prompt:
            continue
        for run_idx in range(1, repeats + 1):
            full_prompt = _build_prompt_from_template(prompt_template, prompt)
            prepared, ctx_tokens, truncated = _apply_context_limit(full_prompt, context_window, max_tokens)
            chat = _local_llm_chat_once(
                endpoint=endpoint,
                model=model,
                prompt=prepared,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            total_runs += 1
            if chat.get("ok"):
                ok_runs += 1
            latency_ms = int(chat.get("latency_ms", 0) or 0)
            answer = str(chat.get("text", "")).strip()
            out_tokens = _estimate_tokens(answer)
            quality = _keyword_quality(answer, expected)
            latencies.append(latency_ms)
            token_outputs.append(out_tokens)
            rows.append(
                {
                    "prompt_id": item.get("id"),
                    "prompt": prompt,
                    "run": run_idx,
                    "ok": bool(chat.get("ok", False)),
                    "latency_ms": latency_ms,
                    "answer": answer,
                    "error": str(chat.get("error", "")),
                    "ctx_tokens_est": ctx_tokens,
                    "ctx_window": context_window,
                    "ctx_truncated": truncated,
                    "output_tokens_est": out_tokens,
                    "quality": quality,
                }
            )
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0
    avg_output_tokens = int(sum(token_outputs) / len(token_outputs)) if token_outputs else 0
    quality_scores = [float(r.get("quality", {}).get("score", 0.0)) for r in rows if r.get("ok")]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0
    return {
        "model": model,
        "quantization": quant_label,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "context_window": context_window,
        "prompt_template": prompt_template,
        "rows": rows,
        "summary": {
            "ok_runs": ok_runs,
            "total_runs": total_runs,
            "success_rate": round(ok_runs / max(1, total_runs), 3),
            "avg_latency_ms": avg_latency,
            "avg_output_tokens_est": avg_output_tokens,
            "avg_quality_score": avg_quality,
        },
    }


def run_local_llm_optimization(
    endpoint: str,
    prompts: list[dict[str, object]],
    baseline: dict[str, object],
    optimized: dict[str, object],
    repeats: int,
) -> dict[str, object]:
    base_report = _run_local_suite(
        endpoint=endpoint,
        model=str(baseline["model"]),
        quant_label=str(baseline["quantization"]),
        prompts=prompts,
        temperature=float(baseline["temperature"]),
        max_tokens=int(baseline["max_tokens"]),
        context_window=int(baseline["context_window"]),
        prompt_template=str(baseline["prompt_template"]),
        repeats=repeats,
    )
    opt_report = _run_local_suite(
        endpoint=endpoint,
        model=str(optimized["model"]),
        quant_label=str(optimized["quantization"]),
        prompts=prompts,
        temperature=float(optimized["temperature"]),
        max_tokens=int(optimized["max_tokens"]),
        context_window=int(optimized["context_window"]),
        prompt_template=str(optimized["prompt_template"]),
        repeats=repeats,
    )
    base_summary = base_report["summary"]
    opt_summary = opt_report["summary"]
    return {
        "endpoint": endpoint,
        "repeats": repeats,
        "prompts_count": len(prompts),
        "baseline": base_report,
        "optimized": opt_report,
        "delta": {
            "quality": round(float(opt_summary["avg_quality_score"]) - float(base_summary["avg_quality_score"]), 3),
            "latency_ms": int(opt_summary["avg_latency_ms"]) - int(base_summary["avg_latency_ms"]),
            "success_rate": round(float(opt_summary["success_rate"]) - float(base_summary["success_rate"]), 3),
            "output_tokens_est": int(opt_summary["avg_output_tokens_est"]) - int(base_summary["avg_output_tokens_est"]),
        },
    }


def build_local_result(
    model: str,
    strategy: str,
    branch: str,
    profile_id: str,
    include_memory_layers: bool,
    latency_ms: int,
    text: str,
) -> dict[str, object]:
    return {
        "text": text,
        "used_model": f"local:{model}",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": int(latency_ms),
        "cost_text": "local",
        "strategy": strategy,
        "branch": branch,
        "current_request_tokens": 0,
        "history_tokens_full": 0,
        "history_tokens_effective": 0,
        "facts_tokens": 0,
        "working_tokens": 0,
        "long_term_tokens": 0,
        "profile_tokens": 0,
        "context_tokens_estimate": 0,
        "context_limit_tokens": 0,
        "include_memory_layers": include_memory_layers,
        "profile_id": profile_id,
        "overflowed": False,
        "invariant_tokens": 0,
        "blocked_by_invariants": False,
        "invariant_report": "",
    }


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
    full_param = request.args.get("full")
    if full_param is not None:
        assignment_mode = full_param != "1"
    else:
        assignment_mode = request.args.get("assignment", "0") == "1"
    prompt = ""
    try:
        model_options = get_available_models()
    except Exception:
        model_options = []
    env_model = get_model_override()
    if env_model:
        default_model = env_model
    elif DEFAULT_MODEL_ID in model_options:
        default_model = DEFAULT_MODEL_ID
    else:
        default_model = model_options[0] if model_options else ""

    selected_model = default_model
    llm_backend = "local"
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
    local_llm_endpoint = LOCAL_LLM_DEFAULT_ENDPOINT
    local_llm_model = LOCAL_LLM_DEFAULT_MODEL
    local_llm_prompt_1 = "Ответь кратко: 2+2=?"
    local_llm_prompt_2 = "Объясни в 3 пунктах, что такое REST API простыми словами."
    local_llm_prompt_3 = (
        "Сделай мини-план запуска pet-проекта трекер привычек на 4 недели: "
        "цели, риски, метрики, стек."
    )
    local_llm_status = ""
    local_llm_result = ""
    local_llm_report: dict[str, object] = {}
    local_llm_opt_case = "RAG QA assistant"
    local_llm_opt_repeats = "1"
    local_llm_opt_model_baseline = LOCAL_LLM_DEFAULT_MODEL
    local_llm_opt_model_optimized = LOCAL_LLM_DEFAULT_MODEL
    local_llm_opt_quant_baseline = "q4_k_m (baseline)"
    local_llm_opt_quant_optimized = "q4_k_m (optimized prompt)"
    local_llm_opt_temperature_baseline = "0.2"
    local_llm_opt_temperature_optimized = "0.35"
    local_llm_opt_max_tokens_baseline = "220"
    local_llm_opt_max_tokens_optimized = "420"
    local_llm_opt_context_baseline = "4096"
    local_llm_opt_context_optimized = "8192"
    local_llm_opt_template_baseline = "Ответь кратко по фактам.\nВопрос:\n{prompt}\nОтвет:"
    local_llm_opt_template_optimized = (
        "Ты помощник по локальному RAG. Дай структурированный ответ:\n"
        "1) Короткий вывод\n2) Детали\n3) Ограничения/риски\n\nВопрос:\n{prompt}\nОтвет:"
    )
    local_llm_opt_expected_1 = "safe call, elvis, !!"
    local_llm_opt_expected_2 = "источники, цитаты, релевантность"
    local_llm_opt_expected_3 = "скорость, стабильность, качество"
    local_llm_opt_status = ""
    local_llm_opt_result = ""
    local_llm_optimization_report: dict[str, object] = {}
    dev_help_question = (
        "Какие точки входа в проекте и что нужно для запуска Telegram-бота и веба?"
    )
    dev_help_answer = ""
    dev_help_context_display = ""
    dev_help_status = ""
    dev_help_use_mcp = False
    support_ticket_id = "TCK-1001"
    support_question = "Почему не работает авторизация?"
    support_answer = ""
    support_context_display = ""
    support_status = ""
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
        action_override = request.form.get("action_override", "").strip().lower()
        if action_override:
            action = action_override
        prompt = request.form.get("prompt", "").strip()
        selected_model = request.form.get("selected_model", default_model).strip()
        llm_backend = request.form.get("llm_backend", llm_backend).strip().lower()
        if llm_backend not in {"local", "cloud"}:
            llm_backend = "local"
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
        local_llm_endpoint = _normalize_local_llm_endpoint(
            request.form.get("local_llm_endpoint", local_llm_endpoint)
        )
        local_llm_model = request.form.get("local_llm_model", local_llm_model).strip() or LOCAL_LLM_DEFAULT_MODEL
        local_llm_prompt_1 = request.form.get("local_llm_prompt_1", local_llm_prompt_1).strip()
        local_llm_prompt_2 = request.form.get("local_llm_prompt_2", local_llm_prompt_2).strip()
        local_llm_prompt_3 = request.form.get("local_llm_prompt_3", local_llm_prompt_3).strip()
        local_llm_opt_case = request.form.get("local_llm_opt_case", local_llm_opt_case).strip()
        local_llm_opt_repeats = request.form.get("local_llm_opt_repeats", local_llm_opt_repeats).strip()
        local_llm_opt_model_baseline = request.form.get(
            "local_llm_opt_model_baseline", local_llm_opt_model_baseline
        ).strip() or local_llm_model
        local_llm_opt_model_optimized = request.form.get(
            "local_llm_opt_model_optimized", local_llm_opt_model_optimized
        ).strip() or local_llm_model
        local_llm_opt_quant_baseline = request.form.get(
            "local_llm_opt_quant_baseline", local_llm_opt_quant_baseline
        ).strip()
        local_llm_opt_quant_optimized = request.form.get(
            "local_llm_opt_quant_optimized", local_llm_opt_quant_optimized
        ).strip()
        local_llm_opt_temperature_baseline = request.form.get(
            "local_llm_opt_temperature_baseline", local_llm_opt_temperature_baseline
        ).strip()
        local_llm_opt_temperature_optimized = request.form.get(
            "local_llm_opt_temperature_optimized", local_llm_opt_temperature_optimized
        ).strip()
        local_llm_opt_max_tokens_baseline = request.form.get(
            "local_llm_opt_max_tokens_baseline", local_llm_opt_max_tokens_baseline
        ).strip()
        local_llm_opt_max_tokens_optimized = request.form.get(
            "local_llm_opt_max_tokens_optimized", local_llm_opt_max_tokens_optimized
        ).strip()
        local_llm_opt_context_baseline = request.form.get(
            "local_llm_opt_context_baseline", local_llm_opt_context_baseline
        ).strip()
        local_llm_opt_context_optimized = request.form.get(
            "local_llm_opt_context_optimized", local_llm_opt_context_optimized
        ).strip()
        local_llm_opt_template_baseline = request.form.get(
            "local_llm_opt_template_baseline", local_llm_opt_template_baseline
        ).strip()
        local_llm_opt_template_optimized = request.form.get(
            "local_llm_opt_template_optimized", local_llm_opt_template_optimized
        ).strip()
        local_llm_opt_expected_1 = request.form.get("local_llm_opt_expected_1", local_llm_opt_expected_1).strip()
        local_llm_opt_expected_2 = request.form.get("local_llm_opt_expected_2", local_llm_opt_expected_2).strip()
        local_llm_opt_expected_3 = request.form.get("local_llm_opt_expected_3", local_llm_opt_expected_3).strip()
        dev_help_question = request.form.get("dev_help_question", dev_help_question).strip()
        dev_help_use_mcp = request.form.get("dev_help_use_mcp", "") == "on"
        support_ticket_id = request.form.get("support_ticket_id", support_ticket_id).strip()
        support_question = request.form.get("support_question", support_question).strip()
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
        elif action == "run_dev_assistant_help":
            q = (request.form.get("dev_help_question") or "").strip()
            use_mcp_ctx = request.form.get("dev_help_use_mcp", "") == "on"
            if not q:
                status = "Введите вопрос ассистенту разработчика."
            else:
                try:
                    prompt, ctx_dict = build_dev_assistant_local_llm_prompt(
                        q, use_mcp_context=use_mcp_ctx
                    )
                    chat = _local_llm_chat_once(
                        endpoint=local_llm_endpoint,
                        model=local_llm_model,
                        prompt=prompt,
                        max_tokens=900,
                        temperature=0.15,
                        system_instruction=None,
                    )
                    dev_help_answer = str(chat.get("text", "")).strip()
                    dev_help_context_display = build_project_context_block(ctx_dict)
                    src = "MCP" if use_mcp_ctx else "git (как в project_context)"
                    if chat.get("ok"):
                        dev_help_status = f"Ответ получен ({chat.get('latency_ms', 0)} ms). Контекст: {src}."
                        status = dev_help_status
                    else:
                        dev_help_status = f"Ошибка локальной LLM: {chat.get('error', '')}"
                        status = dev_help_status
                except Exception as exc:
                    dev_help_status = f"Ошибка ассистента: {exc}"
                    status = dev_help_status
        elif action == "run_support_assistant_help":
            ticket_id = (request.form.get("support_ticket_id") or "").strip()
            q = (request.form.get("support_question") or "").strip()
            if not ticket_id or not q:
                support_status = "Введите ticket_id и вопрос для ассистента поддержки."
                status = support_status
            else:
                try:
                    prompt, ctx_dict = build_support_assistant_local_prompt(
                        ticket_id=ticket_id,
                        question=q,
                    )
                    chat = _local_llm_chat_once(
                        endpoint=local_llm_endpoint,
                        model=local_llm_model,
                        prompt=prompt,
                        max_tokens=900,
                        temperature=0.15,
                        system_instruction=None,
                    )
                    support_answer = str(chat.get("text", "")).strip()
                    support_context_display = json.dumps(ctx_dict, ensure_ascii=False, indent=2)
                    if chat.get("ok"):
                        support_status = (
                            f"Ответ поддержки получен ({chat.get('latency_ms', 0)} ms). "
                            f"Тикет: {ticket_id}."
                        )
                        status = support_status
                    else:
                        support_status = f"Ошибка локальной LLM: {chat.get('error', '')}"
                        status = support_status
                except Exception as exc:
                    support_status = f"Ошибка ассистента поддержки: {exc}"
                    status = support_status
        elif action == "run_local_llm_checks":
            prompts = [local_llm_prompt_1, local_llm_prompt_2, local_llm_prompt_3]
            report = run_local_llm_checks(
                endpoint=local_llm_endpoint,
                model=local_llm_model,
                prompts=prompts,
            )
            local_llm_report = report
            local_llm_result = json.dumps(report, ensure_ascii=False, indent=2)
            health_ok = bool(report.get("health", {}).get("ok", False))
            ok_prompts = int(report.get("summary", {}).get("ok_prompts", 0) or 0)
            total_prompts = int(report.get("summary", {}).get("total_prompts", 0) or 0)
            if health_ok and ok_prompts == total_prompts and total_prompts > 0:
                local_llm_status = f"Local LLM check: OK ({ok_prompts}/{total_prompts} prompts)."
            else:
                local_llm_status = f"Local LLM check: issues found ({ok_prompts}/{total_prompts} prompts)."
            status = local_llm_status
        elif action == "run_local_llm_optimization":
            try:
                repeats = max(1, min(5, int(local_llm_opt_repeats or "1")))
            except ValueError:
                repeats = 1
            prompts_eval = [
                {
                    "id": 1,
                    "prompt": local_llm_prompt_1,
                    "expected_keywords": _split_keywords(local_llm_opt_expected_1),
                },
                {
                    "id": 2,
                    "prompt": local_llm_prompt_2,
                    "expected_keywords": _split_keywords(local_llm_opt_expected_2),
                },
                {
                    "id": 3,
                    "prompt": local_llm_prompt_3,
                    "expected_keywords": _split_keywords(local_llm_opt_expected_3),
                },
            ]
            baseline_cfg = {
                "model": local_llm_opt_model_baseline,
                "quantization": local_llm_opt_quant_baseline,
                "temperature": parse_temperature(local_llm_opt_temperature_baseline, 0.2),
                "max_tokens": parse_max_tokens(local_llm_opt_max_tokens_baseline, 220),
                "context_window": parse_context_limit(local_llm_opt_context_baseline, 4096),
                "prompt_template": local_llm_opt_template_baseline,
            }
            optimized_cfg = {
                "model": local_llm_opt_model_optimized,
                "quantization": local_llm_opt_quant_optimized,
                "temperature": parse_temperature(local_llm_opt_temperature_optimized, 0.35),
                "max_tokens": parse_max_tokens(local_llm_opt_max_tokens_optimized, 420),
                "context_window": parse_context_limit(local_llm_opt_context_optimized, 8192),
                "prompt_template": local_llm_opt_template_optimized,
            }
            report = run_local_llm_optimization(
                endpoint=local_llm_endpoint,
                prompts=prompts_eval,
                baseline=baseline_cfg,
                optimized=optimized_cfg,
                repeats=repeats,
            )
            local_llm_optimization_report = report
            local_llm_opt_result = json.dumps(report, ensure_ascii=False, indent=2)
            delta = report.get("delta", {})
            local_llm_opt_status = (
                f"Optimization run done ({local_llm_opt_case or 'case'}). "
                f"Δquality={delta.get('quality', 0)}, "
                f"Δlatency_ms={delta.get('latency_ms', 0)}, "
                f"Δsuccess={delta.get('success_rate', 0)}"
            )
            status = local_llm_opt_status
        elif action == "send":
            if prompt:
                if llm_backend == "local":
                    chat = _local_llm_chat_once(
                        endpoint=local_llm_endpoint,
                        model=local_llm_model,
                        prompt=prompt,
                        max_tokens=parsed_max_tokens,
                        temperature=parsed_temp,
                    )
                    if chat.get("ok"):
                        answer_text = str(chat.get("text", "")).strip()
                        result = build_local_result(
                            model=local_llm_model,
                            strategy=strategy,
                            branch=selected_branch,
                            profile_id=selected_profile,
                            include_memory_layers=include_memory_layers,
                            latency_ms=int(chat.get("latency_ms", 0) or 0),
                            text=answer_text,
                        )
                        agent.append_external_turn(
                            user_message=prompt,
                            assistant_message=answer_text,
                            branch_id=selected_branch,
                            strategy=f"local_{strategy}",
                            model_id=local_llm_model,
                            latency_ms=int(chat.get("latency_ms", 0) or 0),
                        )
                        status = f"Local LLM response received ({result['latency_ms']} ms)."
                    else:
                        status = f"Local LLM error: {chat.get('error', 'unknown error')}"
                else:
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
        llm_backend=llm_backend,
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
        local_llm_endpoint=local_llm_endpoint,
        local_llm_model=local_llm_model,
        local_llm_prompt_1=local_llm_prompt_1,
        local_llm_prompt_2=local_llm_prompt_2,
        local_llm_prompt_3=local_llm_prompt_3,
        local_llm_status=local_llm_status,
        local_llm_result=local_llm_result,
        local_llm_report=local_llm_report,
        local_llm_opt_case=local_llm_opt_case,
        local_llm_opt_repeats=local_llm_opt_repeats,
        local_llm_opt_model_baseline=local_llm_opt_model_baseline,
        local_llm_opt_model_optimized=local_llm_opt_model_optimized,
        local_llm_opt_quant_baseline=local_llm_opt_quant_baseline,
        local_llm_opt_quant_optimized=local_llm_opt_quant_optimized,
        local_llm_opt_temperature_baseline=local_llm_opt_temperature_baseline,
        local_llm_opt_temperature_optimized=local_llm_opt_temperature_optimized,
        local_llm_opt_max_tokens_baseline=local_llm_opt_max_tokens_baseline,
        local_llm_opt_max_tokens_optimized=local_llm_opt_max_tokens_optimized,
        local_llm_opt_context_baseline=local_llm_opt_context_baseline,
        local_llm_opt_context_optimized=local_llm_opt_context_optimized,
        local_llm_opt_template_baseline=local_llm_opt_template_baseline,
        local_llm_opt_template_optimized=local_llm_opt_template_optimized,
        local_llm_opt_expected_1=local_llm_opt_expected_1,
        local_llm_opt_expected_2=local_llm_opt_expected_2,
        local_llm_opt_expected_3=local_llm_opt_expected_3,
        local_llm_opt_status=local_llm_opt_status,
        local_llm_opt_result=local_llm_opt_result,
        local_llm_optimization_report=local_llm_optimization_report,
        dev_help_question=dev_help_question,
        dev_help_answer=dev_help_answer,
        dev_help_context_display=dev_help_context_display,
        dev_help_status=dev_help_status,
        dev_help_use_mcp=dev_help_use_mcp,
        support_ticket_id=support_ticket_id,
        support_question=support_question,
        support_answer=support_answer,
        support_context_display=support_context_display,
        support_status=support_status,
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
        rag_chat_scenarios=rag_service.chat_scenarios(),
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
    top_k_before = int(payload.get("top_k_before", rag_service.default_top_k_before))
    top_k_after = int(payload.get("top_k_after", rag_service.default_top_k_after))
    threshold = float(payload.get("threshold", rag_service.default_threshold))
    enable_query_rewrite = bool(payload.get("enable_query_rewrite", True))
    if not question:
        return jsonify({"error": "Question is empty."}), 400
    result = rag_service.answer_question(
        question=question,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        threshold=threshold,
        enable_query_rewrite=enable_query_rewrite,
    )
    return jsonify(result)


@app.route("/api/rag_audit", methods=["POST"])
def api_rag_audit():
    payload = request.get_json(force=True, silent=True) or {}
    top_k_before = int(payload.get("top_k_before", rag_service.default_top_k_before))
    top_k_after = int(payload.get("top_k_after", rag_service.default_top_k_after))
    threshold = float(payload.get("threshold", rag_service.default_threshold))
    enable_query_rewrite = bool(payload.get("enable_query_rewrite", True))
    result = rag_service.evaluate_control_questions(
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        threshold=threshold,
        enable_query_rewrite=enable_query_rewrite,
    )
    return jsonify(result)


@app.route("/api/rag_chat", methods=["POST"])
def api_rag_chat():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = str(payload.get("session_id", "default")).strip()
    question = str(payload.get("question", "")).strip()
    top_k_before = int(payload.get("top_k_before", rag_service.default_top_k_before))
    top_k_after = int(payload.get("top_k_after", rag_service.default_top_k_after))
    threshold = float(payload.get("threshold", rag_service.default_threshold))
    if not question:
        return jsonify({"error": "Question is empty."}), 400
    result = rag_service.chat_turn(
        session_id=session_id,
        question=question,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        threshold=threshold,
    )
    return jsonify(result)


@app.route("/api/rag_chat_reset", methods=["POST"])
def api_rag_chat_reset():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = str(payload.get("session_id", "default")).strip()
    result = rag_service.reset_chat_session(session_id)
    return jsonify(
        {
            "session_id": result.get("id", session_id),
            "task_state": result.get("task_state", {}),
            "history": result.get("history", []),
        }
    )


@app.route("/api/rag_chat_scenarios", methods=["POST"])
def api_rag_chat_scenarios():
    payload = request.get_json(force=True, silent=True) or {}
    top_k_before = int(payload.get("top_k_before", rag_service.default_top_k_before))
    top_k_after = int(payload.get("top_k_after", rag_service.default_top_k_after))
    threshold = float(payload.get("threshold", rag_service.default_threshold))
    result = rag_service.evaluate_chat_scenarios(
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        threshold=threshold,
    )
    return jsonify(result)


@app.route("/api/rag_compare", methods=["POST"])
def api_rag_compare():
    payload = request.get_json(force=True, silent=True) or {}
    repeats = int(payload.get("repeats", 2))
    question_limit = int(payload.get("question_limit", 4))
    top_k_before = int(payload.get("top_k_before", rag_service.default_top_k_before))
    top_k_after = int(payload.get("top_k_after", rag_service.default_top_k_after))
    threshold = float(payload.get("threshold", rag_service.default_threshold))
    enable_query_rewrite = bool(payload.get("enable_query_rewrite", True))
    result = rag_service.compare_generation_backends(
        repeats=max(1, min(repeats, 5)),
        question_limit=max(1, min(question_limit, 10)),
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        threshold=threshold,
        enable_query_rewrite=enable_query_rewrite,
    )
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
    web_debug = os.getenv("WEB_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    web_reloader = os.getenv("WEB_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(
        debug=web_debug,
        use_reloader=web_reloader if web_debug else False,
        host="127.0.0.1",
        port=port,
    )
