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
        "context_tokens_estimate": response.context_tokens_estimate,
        "context_limit_tokens": response.context_limit_tokens,
        "overflowed": response.overflowed,
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
        resp = int(meta.get("response_tokens", 0) or 0)
        total_turn = int(meta.get("total_turn_tokens", req + eff_hist + resp))
        cumulative += total_turn
        rows.append(
            {
                "turn": turn,
                "strategy": str(meta.get("strategy", "")),
                "req": req,
                "hist_full": full_hist,
                "hist_effective": eff_hist,
                "facts_tokens": facts_tokens,
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
    checkpoint_label = ""
    new_branch_name = ""
    source_checkpoint_id = ""
    status = ""
    result = {}
    compare_result = {}

    active_branch = agent.get_active_branch()
    branches = agent.list_branches()
    history = agent.load_history(active_branch)
    facts = agent.load_facts(active_branch)
    checkpoints = agent.list_checkpoints(active_branch)
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
        checkpoint_label = request.form.get("checkpoint_label", "").strip()
        new_branch_name = request.form.get("new_branch_name", "").strip()
        source_checkpoint_id = request.form.get("source_checkpoint_id", "").strip()
        selected_branch = request.form.get("selected_branch", active_branch).strip() or active_branch

        parsed_temp = parse_temperature(temperature, 0.7)
        parsed_max_tokens = parse_max_tokens(max_tokens, 600)
        parsed_context_limit = parse_context_limit(context_limit, 200000)
        parsed_window = parse_window(window_n, 8)

        if action == "clear_all":
            agent.clear_all()
            status = "All branches and history cleared."
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
                        context_limit_override=parsed_context_limit,
                    )
                    compared[strat] = as_result_view(response)
                compare_result = compared
                status = "Compared all strategies on the same prompt."
                prompt = ""
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
                    context_limit_override=parsed_context_limit,
                )
                result = as_result_view(response)
                prompt = ""
            else:
                status = "Prompt is empty."

        active_branch = agent.get_active_branch()
        branches = agent.list_branches()
        history = agent.load_history(active_branch)
        facts = agent.load_facts(active_branch)
        checkpoints = agent.list_checkpoints(active_branch)
        token_growth = build_token_growth(history)
        if active_branch not in branches and branches:
            active_branch = branches[0]

    return render_template(
        "index.html",
        prompt=prompt,
        model_options=model_options,
        selected_model=selected_model,
        strategy=strategy,
        strategies=STRATEGIES,
        temperature=temperature,
        max_tokens=max_tokens,
        context_limit=context_limit,
        window_n=window_n,
        checkpoint_label=checkpoint_label,
        new_branch_name=new_branch_name,
        source_checkpoint_id=source_checkpoint_id,
        status=status,
        result=result,
        compare_result=compare_result,
        active_branch=active_branch,
        branches=branches,
        facts=facts,
        checkpoints=checkpoints,
        history=history,
        token_growth=token_growth,
        state_path=state_path,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
