from dotenv import load_dotenv
from flask import Flask, render_template, request

from anthropic_client import ask_claude

app = Flask(__name__)


def get_answer(prompt: str) -> str:
    try:
        return ask_claude(prompt)
    except Exception as exc:
        return f"Error: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    answer = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if prompt:
            answer = get_answer(prompt)
        else:
            answer = "Prompt is empty."
    return render_template("index.html", prompt=prompt, answer=answer)


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
