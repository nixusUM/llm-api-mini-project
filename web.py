from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import ask_claude

app = Flask(__name__)


def get_answer(
    prompt: str,
    max_tokens: int = 250,
    stop_sequences: list[str] | None = None,
) -> str:
    try:
        return ask_claude(
            prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
    except Exception as exc:
        return f"Error: {exc}"


def parse_positive_int(raw_value: str, fallback: int) -> int:
    try:
        parsed_value = int(raw_value)
        return parsed_value if parsed_value > 0 else fallback
    except ValueError:
        return fallback


def as_stop_sequences(raw_value: str) -> list[str] | None:
    cleaned_value = raw_value.strip()
    if not cleaned_value:
        return None
    return [cleaned_value]


def token_limit_from_char_limit(char_limit: int) -> int:
    estimated_tokens = (char_limit // 4) + 20
    return max(40, min(estimated_tokens, 400))


def build_constrained_prompt(
    prompt: str,
    format_instruction: str,
    char_limit: int,
    ending_instruction: str,
) -> str:
    rules = [
        f"Format: {format_instruction.strip()}",
        f"Length limit: {char_limit} characters max.",
        f"Finish condition: {ending_instruction.strip()}",
    ]
    instructions = "\n".join(f"- {rule}" for rule in rules)
    return f"{prompt}\n\nResponse rules:\n{instructions}"


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    unconstrained_answer = ""
    constrained_answer = ""
    format_instruction = "Use exactly 3 bullet points."
    length_limit = "220"
    ending_instruction = "Finish with [END]."
    stop_sequence = "[END]"

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        format_instruction = request.form.get("format_instruction", "").strip()
        length_limit = request.form.get("length_limit", "").strip()
        ending_instruction = request.form.get("ending_instruction", "").strip()
        stop_sequence = request.form.get("stop_sequence", "").strip()

        if prompt:
            unconstrained_answer = get_answer(prompt)
            char_limit = parse_positive_int(length_limit, 220)
            token_limit = token_limit_from_char_limit(char_limit)
            constrained_prompt = build_constrained_prompt(
                prompt,
                format_instruction or "Use exactly 3 bullet points.",
                char_limit,
                ending_instruction or "Finish with [END].",
            )
            constrained_answer = get_answer(
                constrained_prompt,
                max_tokens=token_limit,
                stop_sequences=as_stop_sequences(stop_sequence),
            )
        else:
            unconstrained_answer = "Prompt is empty."
            constrained_answer = "Prompt is empty."

    return render_template(
        "index.html",
        prompt=prompt,
        unconstrained_answer=unconstrained_answer,
        constrained_answer=constrained_answer,
        format_instruction=format_instruction,
        length_limit=length_limit,
        ending_instruction=ending_instruction,
        stop_sequence=stop_sequence,
    )


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
