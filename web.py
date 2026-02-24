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
    result = {}
    history = agent.load_history()
    history_path = str(agent.history_path)

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
                result={},
                history=[],
                history_path=history_path,
            )

        prompt = request.form.get("prompt", "").strip()
        selected_model = request.form.get("selected_model", default_model).strip()
        if model_options and selected_model not in model_options:
            selected_model = default_model
        temperature = request.form.get("temperature", "0.7").strip()
        max_tokens = request.form.get("max_tokens", "600").strip()
        parsed_temp = parse_temperature(temperature, 0.7)
        parsed_max_tokens = parse_max_tokens(max_tokens, 600)
        if prompt:
            agent_result = agent.run_chat_persistent(
                user_message=prompt,
                model_id=selected_model,
                temperature=parsed_temp,
                max_tokens=parsed_max_tokens,
            )
            result = {
                "text": agent_result.text,
                "used_model": agent_result.used_model,
                "input_tokens": agent_result.input_tokens,
                "output_tokens": agent_result.output_tokens,
                "total_tokens": agent_result.total_tokens,
                "latency_ms": agent_result.latency_ms,
                "cost_text": agent_result.cost_text,
            }
            history = agent.load_history()
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
        result=result,
        history=history,
        history_path=history_path,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
