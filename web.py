from time import perf_counter

from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import ask_claude_with_meta
from anthropic_client import get_available_models
from anthropic_client import get_model_override

app = Flask(__name__)


def parse_temperature(raw_value: str, fallback: float) -> float:
    try:
        return float(raw_value)
    except ValueError:
        return fallback


def parse_max_tokens(raw_value: str, fallback: int) -> int:
    try:
        parsed = int(raw_value)
    except ValueError:
        return fallback
    return max(100, min(parsed, 2000))


def infer_rates_usd_per_mtok(model_id: str) -> tuple[float, float] | None:
    model = model_id.lower()
    if "haiku" in model:
        return (0.80, 4.00)
    if "sonnet" in model:
        return (3.00, 15.00)
    if "opus" in model:
        return (15.00, 75.00)
    return None


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = infer_rates_usd_per_mtok(model_id)
    if not rates:
        return None
    input_rate, output_rate = rates
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "N/A"
    return f"${cost:.6f}"


def build_model_summary_prompt(
    prompt: str,
    weak: dict,
    medium: dict,
    strong: dict,
    language: str,
) -> str:
    rows = []
    rows.append(f"Weak model ({weak['used_model']}): {weak['text']}")
    rows.append(f"Medium model ({medium['used_model']}): {medium['text']}")
    rows.append(f"Strong model ({strong['used_model']}): {strong['text']}")
    body = "\n\n".join(rows)
    is_ru = language == "ru"
    if is_ru:
        header = (
            "Сравни три ответа моделей на один и тот же запрос. "
            "Оцени: качество, скорость и ресурсоемкость. "
            "Выведи краткий markdown до 900 символов с разделами: "
            "Качество, Скорость, Ресурсоемкость, Где лучше использовать, Итог."
        )
        metrics = (
            "Метрики:\n"
            f"Слабая: задержка {weak['latency_ms']} мс, токены {weak['total_tokens']}, стоимость {weak['cost_text']}\n"
            f"Средняя: задержка {medium['latency_ms']} мс, токены {medium['total_tokens']}, стоимость {medium['cost_text']}\n"
            f"Сильная: задержка {strong['latency_ms']} мс, токены {strong['total_tokens']}, стоимость {strong['cost_text']}"
        )
        return f"{header}\n\nИсходный запрос:\n{prompt}\n\n{metrics}\n\nОтветы:\n{body}"
    return (
        "Compare these three model outputs for one prompt. "
        "Evaluate: quality, speed, and resource usage. "
        "Use concise markdown under 900 characters with sections: "
        "Quality, Speed, Resource usage, Best fit by model tier, Final verdict.\n\n"
        f"Original prompt:\n{prompt}\n\n"
        "Metrics:\n"
        f"Weak latency {weak['latency_ms']} ms, tokens {weak['total_tokens']}, cost {weak['cost_text']}\n"
        f"Medium latency {medium['latency_ms']} ms, tokens {medium['total_tokens']}, cost {medium['cost_text']}\n"
        f"Strong latency {strong['latency_ms']} ms, tokens {strong['total_tokens']}, cost {strong['cost_text']}\n\n"
        f"Outputs:\n{body}"
    )


def pick_default_models(model_options: list[str]) -> tuple[str, str, str]:
    if not model_options:
        return "", "", ""
    weak = model_options[-1]
    medium = model_options[len(model_options) // 2]
    strong = model_options[0]
    return weak, medium, strong


def request_model_result(prompt: str, model_id: str, temperature: float, max_tokens: int) -> dict:
    started = perf_counter()
    try:
        text, used_model, usage = ask_claude_with_meta(
            prompt=prompt,
            model_override=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        elapsed_ms = int((perf_counter() - started) * 1000)
        return {
            "text": f"Error: {exc}",
            "used_model": model_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": elapsed_ms,
            "cost_text": "N/A",
        }
    elapsed_ms = int((perf_counter() - started) * 1000)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    cost = estimate_cost_usd(used_model, input_tokens, output_tokens)
    return {
        "text": text,
        "used_model": used_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": elapsed_ms,
        "cost_text": format_cost(cost),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    model_options = get_available_models()
    weak_model, medium_model, strong_model = pick_default_models(model_options)
    temperature = "0.7"
    max_tokens = "500"
    max_tokens_summary = "600"
    summary_language = "ru"
    weak = {}
    medium = {}
    strong = {}
    summary = ""
    summary_model = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        weak_model = request.form.get("weak_model", weak_model).strip()
        medium_model = request.form.get("medium_model", medium_model).strip()
        strong_model = request.form.get("strong_model", strong_model).strip()
        temperature = request.form.get("temperature", "0.7").strip()
        max_tokens = request.form.get("max_tokens", "500").strip()
        max_tokens_summary = request.form.get("max_tokens_summary", "600").strip()
        summary_language = request.form.get("summary_language", "ru").strip().lower()
        if summary_language not in {"ru", "en"}:
            summary_language = "ru"

        parsed_temp = parse_temperature(temperature, 0.7)
        parsed_max_tokens = parse_max_tokens(max_tokens, 500)
        parsed_summary_tokens = parse_max_tokens(max_tokens_summary, 600)

        if prompt:
            weak = request_model_result(prompt, weak_model, parsed_temp, parsed_max_tokens)
            medium = request_model_result(prompt, medium_model, parsed_temp, parsed_max_tokens)
            strong = request_model_result(prompt, strong_model, parsed_temp, parsed_max_tokens)

            env_model = get_model_override() or strong.get("used_model", strong_model)
            summary_prompt = build_model_summary_prompt(
                prompt,
                weak,
                medium,
                strong,
                summary_language,
            )
            summary_data = request_model_result(
                summary_prompt,
                env_model,
                0.0,
                parsed_summary_tokens,
            )
            summary = summary_data["text"]
            summary_model = summary_data["used_model"]
        else:
            message = "Prompt is empty."
            weak = {"text": message}
            medium = {"text": message}
            strong = {"text": message}

    return render_template(
        "index.html",
        prompt=prompt,
        model_options=model_options,
        weak_model=weak_model,
        medium_model=medium_model,
        strong_model=strong_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_tokens_summary=max_tokens_summary,
        summary_language=summary_language,
        weak=weak,
        medium=medium,
        strong=strong,
        summary=summary,
        summary_model=summary_model,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
