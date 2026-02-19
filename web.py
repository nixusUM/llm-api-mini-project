from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import ask_claude_with_meta
from anthropic_client import get_available_models
from anthropic_client import get_model_override

app = Flask(__name__)


def get_answer(
    prompt: str,
    model_override: str,
    max_tokens: int = 900,
    stop_sequences: list[str] | None = None,
    temperature: float | None = None,
) -> tuple[str, str]:
    try:
        return ask_claude_with_meta(
            prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            temperature=temperature,
            model_override=model_override,
        )
    except Exception as exc:
        return f"Error: {exc}", model_override


def parse_temperature(raw_value: str, fallback: float) -> float:
    try:
        return float(raw_value)
    except ValueError:
        return fallback


def format_temperature(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_max_tokens(raw_value: str, fallback: int) -> int:
    try:
        parsed = int(raw_value)
    except ValueError:
        return fallback
    return max(100, min(parsed, 2000))


def build_comparison_prompt(
    prompt: str,
    t0: float,
    a0: str,
    t1: float,
    a1: str,
    t2: float,
    a2: str,
) -> str:
    parts = [
        "Compare three model outputs for the same prompt.",
        "Assess: accuracy, creativity, diversity, and best use-cases.",
        "Return short markdown without tables.",
        "Keep total response under 900 characters.",
        "Use 3-5 bullet points per section.",
        "Do not quote full model outputs.",
        "Return sections:",
        "1) Accuracy",
        "2) Creativity",
        "3) Diversity",
        "4) When to use each temperature",
        "5) Final recommendation",
        "",
        f"Original prompt:\n{prompt}",
        "",
        f"Temperature {format_temperature(t0)} output:\n{a0}",
        "",
        f"Temperature {format_temperature(t1)} output:\n{a1}",
        "",
        f"Temperature {format_temperature(t2)} output:\n{a2}",
    ]
    return "\n".join(parts)


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    model_options = get_available_models()
    default_model = get_model_override() or (model_options[0] if model_options else "")
    selected_model = default_model
    temp_0 = "0"
    temp_07 = "0.7"
    temp_12 = "1.0"
    max_tokens_answer = "650"
    max_tokens_summary = "320"
    temperature_answer_0 = ""
    temperature_answer_07 = ""
    temperature_answer_12 = ""
    used_model_0 = ""
    used_model_07 = ""
    used_model_12 = ""
    effective_temp_0 = temp_0
    effective_temp_07 = temp_07
    effective_temp_12 = temp_12
    comparison_summary = ""
    summary_model = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        selected_model = request.form.get("selected_model", default_model).strip()
        if model_options and selected_model not in model_options:
            selected_model = default_model
        temp_0 = request.form.get("temp_0", "0").strip()
        temp_07 = request.form.get("temp_07", "0.7").strip()
        temp_12 = request.form.get("temp_12", "1.0").strip()
        max_tokens_answer = request.form.get("max_tokens_answer", "650").strip()
        max_tokens_summary = request.form.get("max_tokens_summary", "320").strip()
        answer_tokens = parse_max_tokens(max_tokens_answer, 650)
        summary_tokens = parse_max_tokens(max_tokens_summary, 320)
        parsed_temp_0 = parse_temperature(temp_0, 0.0)
        parsed_temp_07 = parse_temperature(temp_07, 0.7)
        parsed_temp_12 = parse_temperature(temp_12, 1.0)
        effective_temp_0 = format_temperature(parsed_temp_0)
        effective_temp_07 = format_temperature(parsed_temp_07)
        effective_temp_12 = format_temperature(parsed_temp_12)

        if prompt:
            temperature_answer_0, used_model_0 = get_answer(
                prompt,
                model_override=selected_model,
                temperature=parsed_temp_0,
                max_tokens=answer_tokens,
            )
            temperature_answer_07, used_model_07 = get_answer(
                prompt,
                model_override=selected_model,
                temperature=parsed_temp_07,
                max_tokens=answer_tokens,
            )
            temperature_answer_12, used_model_12 = get_answer(
                prompt,
                model_override=selected_model,
                temperature=parsed_temp_12,
                max_tokens=answer_tokens,
            )
            summary_prompt = build_comparison_prompt(
                prompt,
                parsed_temp_0,
                temperature_answer_0,
                parsed_temp_07,
                temperature_answer_07,
                parsed_temp_12,
                temperature_answer_12,
            )
            comparison_summary, summary_model = get_answer(
                summary_prompt,
                model_override=selected_model,
                temperature=0.0,
                max_tokens=summary_tokens,
            )
        else:
            temperature_answer_0 = "Prompt is empty."
            temperature_answer_07 = "Prompt is empty."
            temperature_answer_12 = "Prompt is empty."
            used_model_0 = selected_model
            used_model_07 = selected_model
            used_model_12 = selected_model

    return render_template(
        "index.html",
        prompt=prompt,
        model_options=model_options,
        selected_model=selected_model,
        temp_0=temp_0,
        temp_07=temp_07,
        temp_12=temp_12,
        max_tokens_answer=max_tokens_answer,
        max_tokens_summary=max_tokens_summary,
        temperature_answer_0=temperature_answer_0,
        temperature_answer_07=temperature_answer_07,
        temperature_answer_12=temperature_answer_12,
        used_model_0=used_model_0,
        used_model_07=used_model_07,
        used_model_12=used_model_12,
        effective_temp_0=effective_temp_0,
        effective_temp_07=effective_temp_07,
        effective_temp_12=effective_temp_12,
        comparison_summary=comparison_summary,
        summary_model=summary_model,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
