from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import get_available_models
from anthropic_client import get_model_override
from llm_agent import LLMAgent

app = Flask(__name__)
agent = LLMAgent()
DEFAULT_MODEL_ID = "claude-3-haiku-20240307"


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


def build_token_growth(history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    cumulative = 0
    turn = 1
    for item in history:
        if item.get("role") != "assistant":
            continue
        meta = item.get("meta", {})
        if not isinstance(meta, dict):
            continue
        current_request = int(meta.get("current_request_tokens", 0) or 0)
        history_tokens = int(meta.get("history_tokens_effective", meta.get("history_tokens", 0)) or 0)
        history_tokens_full = int(meta.get("history_tokens_full", history_tokens) or 0)
        response_tokens = int(meta.get("response_tokens", 0) or 0)
        total_turn = current_request + history_tokens + response_tokens
        cumulative += total_turn
        rows.append(
            {
                "turn": turn,
                "current_request_tokens": current_request,
                "history_tokens": history_tokens,
                "history_tokens_full": history_tokens_full,
                "response_tokens": response_tokens,
                "total_turn_tokens": total_turn,
                "cumulative_tokens": cumulative,
                "overflowed": bool(meta.get("overflowed", False)),
                "token_savings": int(meta.get("token_savings", 0) or 0),
                "compression_enabled": bool(meta.get("compression_enabled", False)),
            }
        )
        turn += 1
    return rows


def response_to_view(result) -> dict:
    return {
        "text": result.text,
        "used_model": result.used_model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "cost_text": result.cost_text,
        "current_request_tokens": result.current_request_tokens,
        "history_tokens": result.history_tokens,
        "context_tokens_estimate": result.context_tokens_estimate,
        "context_limit_tokens": result.context_limit_tokens,
        "overflowed": result.overflowed,
        "history_tokens_full": result.history_tokens_full,
        "history_tokens_effective": result.history_tokens_effective,
        "summary_tokens": result.summary_tokens,
        "token_savings": result.token_savings,
        "compression_enabled": result.compression_enabled,
    }


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
    temperature = "0.7"
    max_tokens = "600"
    context_limit = "200000"
    compression_enabled = True
    keep_last_n = "8"
    summarize_every_n = "10"
    result = {}
    ab_result = {}
    history = agent.load_history()
    history_path = str(agent.history_path)
    token_growth = build_token_growth(history)

    if request.method == "POST":
        action = request.form.get("action", "send").strip().lower()
        if action in {"clear", "clear_all"}:
            agent.clear_history()
            return render_template(
                "index.html",
                prompt="",
                model_options=model_options,
                selected_model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                context_limit=context_limit,
                compression_enabled=compression_enabled,
                keep_last_n=keep_last_n,
                summarize_every_n=summarize_every_n,
                result={},
                ab_result={},
                history=[],
                history_path=history_path,
                token_growth=[],
            )

        prompt = request.form.get("prompt", "").strip()
        selected_model = request.form.get("selected_model", default_model).strip()
        if model_options and selected_model not in model_options:
            selected_model = default_model
        temperature = request.form.get("temperature", "0.7").strip()
        max_tokens = request.form.get("max_tokens", "600").strip()
        context_limit = request.form.get("context_limit", "200000").strip()
        compression_enabled = request.form.get("compression_enabled") == "on"
        keep_last_n = request.form.get("keep_last_n", "8").strip()
        summarize_every_n = request.form.get("summarize_every_n", "10").strip()
        parsed_temp = parse_temperature(temperature, 0.7)
        parsed_max_tokens = parse_max_tokens(max_tokens, 600)
        parsed_context_limit = parse_context_limit(context_limit, 200000)
        parsed_keep_last_n = max(2, min(int(keep_last_n) if keep_last_n.isdigit() else 8, 60))
        parsed_summarize_every_n = max(2, min(int(summarize_every_n) if summarize_every_n.isdigit() else 10, 50))
        if prompt:
            if action == "compare_ab":
                compared = agent.compare_compression(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    context_limit_override=parsed_context_limit,
                    keep_last_n=parsed_keep_last_n,
                    summarize_every_n=parsed_summarize_every_n,
                )
                ab_result = {
                    "no_compression": response_to_view(compared["no_compression"]),
                    "with_compression": response_to_view(compared["with_compression"]),
                }
            else:
                agent_result = agent.run_chat_persistent(
                    user_message=prompt,
                    model_id=selected_model,
                    temperature=parsed_temp,
                    max_tokens=parsed_max_tokens,
                    context_limit_override=parsed_context_limit,
                    compression_enabled=compression_enabled,
                    keep_last_n=parsed_keep_last_n,
                    summarize_every_n=parsed_summarize_every_n,
                )
                result = response_to_view(agent_result)
                history = agent.load_history()
                token_growth = build_token_growth(history)
                prompt = ""
        else:
            result = {"text": "Prompt is empty."}

    return render_template(
        "index.html",
        prompt=prompt,
        model_options=model_options,
        selected_model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
        context_limit=context_limit,
        compression_enabled=compression_enabled,
        keep_last_n=keep_last_n,
        summarize_every_n=summarize_every_n,
        result=result,
        ab_result=ab_result,
        history=history,
        history_path=history_path,
        token_growth=token_growth,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
