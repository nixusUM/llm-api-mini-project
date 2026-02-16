import os
from typing import Any

import requests

API_URL = "https://api.anthropic.com/v1/messages"


def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to .env or env vars.")


def get_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def extract_text(data: dict[str, Any]) -> str:
    content = data.get("content", [])
    if not content:
        return "Empty response from model."
    return content[0].get("text", "No text in model response.")


def ask_claude(prompt: str, max_tokens: int = 250) -> str:
    headers = {
        "x-api-key": get_api_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": get_model(),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
    response.raise_for_status()
    return extract_text(response.json())
