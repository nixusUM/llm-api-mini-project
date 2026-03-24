import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv


def get_env(name: str, fallback: str) -> str:
    return os.getenv(name, fallback).strip()


def http_get_json(url: str, timeout_sec: float = 8.0) -> dict:
    req = urlrequest.Request(url=url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def http_post_json(url: str, payload: dict, timeout_sec: float = 20.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    load_dotenv()
    endpoint = get_env("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8088").rstrip("/")
    model = get_env("LOCAL_LLM_MODEL", "qwen-local")

    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")

    try:
        health = http_get_json(f"{endpoint}/health")
        print("\n[health]")
        print(json.dumps(health, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"\n[health] failed: {exc}")
        return

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ответь одним словом: ok"}],
        "temperature": 0.2,
        "max_tokens": 30,
    }
    try:
        response = http_post_json(f"{endpoint}/v1/chat/completions", payload)
        print("\n[chat]")
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            print(str(message.get("content", "")).strip() or "<empty>")
        else:
            print(json.dumps(response, ensure_ascii=False, indent=2))
    except urlerror.HTTPError as exc:
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        print(f"\n[chat] failed: HTTP {exc.code} {details}")
    except Exception as exc:
        print(f"\n[chat] failed: {exc}")


if __name__ == "__main__":
    main()
