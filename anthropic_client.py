import os
from anthropic import Anthropic
from anthropic import APIStatusError

PREFERRED_MODELS = (
    "claude-sonnet-4-5",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
)


def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to .env or env vars.")


def get_model_override() -> str:
    return os.getenv("ANTHROPIC_MODEL", "").strip()


def get_error_text(exc: APIStatusError) -> str:
    return exc.response.text if exc.response else str(exc)


def message_has_text(message: object) -> bool:
    content = getattr(message, "content", [])
    return bool(content)


def extract_text(message: object) -> str:
    if not message_has_text(message):
        return "Empty response from model."
    first_block = message.content[0]
    return getattr(first_block, "text", "No text in model response.")


def send_message(client: Anthropic, prompt: str, max_tokens: int, model: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_text(message)


def list_model_ids(client: Anthropic) -> list[str]:
    return [model.id for model in client.models.list()]


def resolve_model(client: Anthropic, override: str) -> str:
    model_ids = list_model_ids(client)
    if override and override in model_ids:
        return override
    for model in PREFERRED_MODELS:
        if model in model_ids:
            return model
    if model_ids:
        return model_ids[0]
    if override:
        return override
    return PREFERRED_MODELS[0]


def is_model_not_found(exc: APIStatusError) -> bool:
    text = get_error_text(exc).lower()
    return exc.status_code == 404 and "model" in text


def ask_claude(prompt: str, max_tokens: int = 250) -> str:
    client = Anthropic(api_key=get_api_key())
    model = get_model_override()
    selected_model = model or PREFERRED_MODELS[0]
    try:
        return send_message(client, prompt, max_tokens, selected_model)
    except APIStatusError as exc:
        if is_model_not_found(exc):
            fallback_model = resolve_model(client, model)
            if fallback_model != selected_model:
                return send_message(client, prompt, max_tokens, fallback_model)
        code = exc.status_code
        raise RuntimeError(f"Anthropic API error ({code}): {get_error_text(exc)}") from exc
