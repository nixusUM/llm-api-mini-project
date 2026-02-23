import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, session

from anthropic_client import get_available_models
from anthropic_client import get_model_override
from llm_agent import LLMAgent

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
agent = LLMAgent()


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


def get_chat_history() -> list[dict[str, str]]:
    history = session.get("chat_history", [])
    return history if isinstance(history, list) else []


def save_chat_history(history: list[dict[str, str]]) -> None:
    session["chat_history"] = history[-20:]


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    model_options = get_available_models()
    default_model = get_model_override() or (model_options[0] if model_options else "")
    selected_model = default_model
    temperature = "0.7"
    max_tokens = "600"
    result = {}
    history = get_chat_history()

    if request.method == "POST":
        action = request.form.get("action", "send").strip().lower()
        if action == "clear":
            save_chat_history([])
            return render_template(
                "index.html",
                prompt="",
                model_options=model_options,
                selected_model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                result={},
                history=[],
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
            agent_result = agent.run_chat(
                history=history,
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
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": agent_result.text})
            save_chat_history(history)
            history = get_chat_history()
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
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
