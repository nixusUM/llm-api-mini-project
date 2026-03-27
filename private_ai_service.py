import json
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()
app = Flask(__name__)


def _env_text(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env_text(name, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_text(name, str(default))
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class ServiceConfig:
    host: str
    port: int
    local_endpoint: str
    default_model: str
    api_key: str
    max_requests_per_minute: int
    max_context_tokens: int
    default_max_tokens: int
    default_temperature: float
    max_completion_tokens: int
    allow_localhost_without_key: bool


def load_config() -> ServiceConfig:
    return ServiceConfig(
        host=_env_text("PRIVATE_AI_HOST", "0.0.0.0"),
        port=_env_int("PRIVATE_AI_PORT", 8099),
        local_endpoint=_env_text("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8088").rstrip("/"),
        default_model=_env_text("LOCAL_LLM_MODEL", "qwen-local"),
        api_key=_env_text("PRIVATE_AI_API_KEY", "change-me-private-key"),
        max_requests_per_minute=max(1, _env_int("PRIVATE_AI_RATE_LIMIT_PER_MIN", 30)),
        max_context_tokens=max(256, _env_int("PRIVATE_AI_MAX_CONTEXT_TOKENS", 8192)),
        default_max_tokens=max(64, _env_int("PRIVATE_AI_DEFAULT_MAX_TOKENS", 384)),
        default_temperature=max(0.0, min(1.5, _env_float("PRIVATE_AI_DEFAULT_TEMPERATURE", 0.3))),
        max_completion_tokens=max(64, _env_int("PRIVATE_AI_MAX_COMPLETION_TOKENS", 1024)),
        allow_localhost_without_key=_env_text("PRIVATE_AI_ALLOW_LOCALHOST_NO_AUTH", "1") in {"1", "true", "yes"},
    )


CONFIG = load_config()
_RATE_LOCK = Lock()
_RATE_BUCKETS: dict[str, deque[float]] = {}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4) if text else 1


def _client_id() -> str:
    header_key = request.headers.get("X-API-Key", "").strip()
    source = header_key or request.headers.get("Authorization", "").strip() or request.remote_addr or "unknown"
    return source[-48:]


def _prune_bucket(bucket: deque[float], now: float) -> None:
    cutoff = now - 60.0
    while bucket and bucket[0] < cutoff:
        bucket.popleft()


def _check_rate_limit() -> tuple[bool, int]:
    now = time.time()
    key = _client_id()
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS.setdefault(key, deque())
        _prune_bucket(bucket, now)
        remaining = CONFIG.max_requests_per_minute - len(bucket)
        if remaining <= 0:
            return False, 0
        bucket.append(now)
        return True, max(0, remaining - 1)


def _auth_ok() -> bool:
    if not CONFIG.api_key:
        return True
    if CONFIG.allow_localhost_without_key:
        addr = (request.remote_addr or "").strip()
        if addr in {"127.0.0.1", "::1", "localhost"}:
            return True
    incoming = request.headers.get("X-API-Key", "").strip()
    if incoming == CONFIG.api_key:
        return True
    auth = request.headers.get("Authorization", "").strip()
    return auth == f"Bearer {CONFIG.api_key}"


def _prepare_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    used_tokens = 0
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        msg_tokens = _estimate_tokens(content) + 6
        if used_tokens + msg_tokens > CONFIG.max_context_tokens:
            break
        prepared.append({"role": role, "content": content})
        used_tokens += msg_tokens
    return prepared, used_tokens


def _post_local_chat(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url=f"{CONFIG.local_endpoint}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlrequest.urlopen(req, timeout=60.0) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


@app.route("/health", methods=["GET"])
def health() -> Any:
    local_ok = False
    local_error = ""
    started = time.perf_counter()
    try:
        req = urlrequest.Request(url=f"{CONFIG.local_endpoint}/health", method="GET")
        with urlrequest.urlopen(req, timeout=6.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        local_ok = str(payload.get("status", "")).lower() == "ok"
    except Exception as exc:
        local_error = str(exc)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return jsonify(
        {
            "status": "ok" if local_ok else "degraded",
            "service": "private-local-llm",
            "local_llm_ok": local_ok,
            "local_llm_endpoint": CONFIG.local_endpoint,
            "latency_ms": latency_ms,
            "error": local_error,
        }
    )


def _chat_internal(payload: dict[str, Any], openai_compatible: bool) -> tuple[Any, int]:
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    allowed, remaining = _check_rate_limit()
    if not allowed:
        return jsonify({"error": "Rate limit exceeded", "limit_per_min": CONFIG.max_requests_per_minute}), 429

    model = str(payload.get("model", CONFIG.default_model)).strip() or CONFIG.default_model
    max_tokens = int(payload.get("max_tokens", CONFIG.default_max_tokens) or CONFIG.default_max_tokens)
    max_tokens = max(64, min(max_tokens, CONFIG.max_completion_tokens))
    temperature = float(payload.get("temperature", CONFIG.default_temperature) or CONFIG.default_temperature)
    temperature = max(0.0, min(temperature, 1.5))
    incoming_messages = payload.get("messages", [])
    if not isinstance(incoming_messages, list) or not incoming_messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    messages, context_tokens = _prepare_messages(incoming_messages)
    if not messages:
        return jsonify({"error": "context too large or empty", "max_context_tokens": CONFIG.max_context_tokens}), 400

    started = time.perf_counter()
    try:
        llm = _post_local_chat(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
    except urlerror.HTTPError as exc:
        return jsonify({"error": f"Local LLM HTTP {exc.code}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    latency_ms = int((time.perf_counter() - started) * 1000)

    choices = llm.get("choices", [])
    if not choices:
        return jsonify({"error": "Local LLM returned no choices"}), 502
    message = choices[0].get("message", {})
    text = str(message.get("content", "")).strip()
    out_tokens = _estimate_tokens(text)
    if openai_compatible:
        return (
            jsonify(
                {
                    "id": f"chatcmpl-private-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": context_tokens,
                        "completion_tokens": out_tokens,
                        "total_tokens": context_tokens + out_tokens,
                    },
                    "meta": {
                        "latency_ms": latency_ms,
                        "rate_limit_remaining": remaining,
                        "max_context_tokens": CONFIG.max_context_tokens,
                    },
                }
            ),
            200,
        )
    return jsonify(
        {
            "ok": True,
            "model": model,
            "answer": text,
            "latency_ms": latency_ms,
            "rate_limit_remaining": remaining,
            "usage_proxy": {
                "context_tokens_est": context_tokens,
                "output_tokens_est": out_tokens,
                "max_context_tokens": CONFIG.max_context_tokens,
            },
        }
    ), 200


@app.route("/v1/chat", methods=["POST"])
def chat() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    body, status_code = _chat_internal(payload, openai_compatible=False)
    return body, status_code


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    body, status_code = _chat_internal(payload, openai_compatible=True)
    return body, status_code


@app.route("/v1/config", methods=["GET"])
def config_view() -> Any:
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(
        {
            "service": "private-local-llm",
            "local_endpoint": CONFIG.local_endpoint,
            "default_model": CONFIG.default_model,
            "limits": {
                "rate_per_min": CONFIG.max_requests_per_minute,
                "max_context_tokens": CONFIG.max_context_tokens,
                "max_completion_tokens": CONFIG.max_completion_tokens,
            },
        }
    )


if __name__ == "__main__":
    app.run(host=CONFIG.host, port=CONFIG.port, debug=False, threaded=True)
