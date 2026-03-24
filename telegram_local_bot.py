import json
import os
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv

LOCAL_LLM_ENDPOINT = "LOCAL_LLM_ENDPOINT"
LOCAL_LLM_MODEL = "LOCAL_LLM_MODEL"
TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"


def get_env(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback).strip()


def normalize_endpoint(raw_endpoint: str) -> str:
    value = raw_endpoint.strip() or "http://127.0.0.1:8088"
    return value.rstrip("/")


def telegram_api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def http_post_json(url: str, payload: dict, timeout_sec: float = 25.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def send_tg_message(token: str, chat_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text[:4000]}
    try:
        http_post_json(telegram_api_url(token, "sendMessage"), payload, timeout_sec=15.0)
    except Exception:
        return


def fetch_tg_updates(token: str, offset: int) -> list[dict]:
    payload = {"timeout": 20, "offset": offset, "allowed_updates": ["message"]}
    response = http_post_json(telegram_api_url(token, "getUpdates"), payload)
    if not response.get("ok"):
        return []
    result = response.get("result", [])
    return result if isinstance(result, list) else []


def call_local_llm(endpoint: str, model: str, user_text: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": 0.3,
        "max_tokens": 700,
    }
    url = f"{endpoint}/v1/chat/completions"
    try:
        response = http_post_json(url, payload, timeout_sec=60.0)
        choices = response.get("choices", [])
        if not choices:
            return "Локальная LLM вернула пустой ответ."
        message = choices[0].get("message", {})
        answer = str(message.get("content", "")).strip()
        return answer or "Локальная LLM вернула пустой текст."
    except urlerror.HTTPError as exc:
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        return f"Ошибка локальной LLM: HTTP {exc.code} {details}"
    except Exception as exc:
        return f"Ошибка локальной LLM: {exc}"


def extract_message(update: dict) -> tuple[int, str] | None:
    message = update.get("message", {})
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = str(message.get("text", "")).strip()
    if not isinstance(chat_id, int) or not text:
        return None
    return chat_id, text


def handle_command(text: str) -> str | None:
    if text == "/start":
        return (
            "Привет! Я Telegram-бот на локальной LLM.\n"
            "Отправь любой текст, и я отвечу через локальную модель.\n"
            "Команды:\n"
            "/start - приветствие\n"
            "/health - проверить локальную LLM"
        )
    if text == "/health":
        return "Проверяю через следующий запрос..."
    return None


def bot_loop(token: str, endpoint: str, model: str) -> None:
    offset = 0
    while True:
        try:
            updates = fetch_tg_updates(token, offset)
            for update in updates:
                update_id = int(update.get("update_id", 0))
                offset = max(offset, update_id + 1)
                payload = extract_message(update)
                if payload is None:
                    continue
                chat_id, text = payload
                command_reply = handle_command(text)
                if command_reply is not None:
                    send_tg_message(token, chat_id, command_reply)
                    if text != "/health":
                        continue
                    text = "Ответь одним словом: ok"
                answer = call_local_llm(endpoint, model, text)
                send_tg_message(token, chat_id, answer)
        except Exception as exc:
            print(f"[bot] polling error: {exc}")
            time.sleep(2)


def validate_settings(token: str, endpoint: str, model: str) -> bool:
    if not token:
        print("Missing TELEGRAM_BOT_TOKEN in .env")
        return False
    if not endpoint:
        print("Missing LOCAL_LLM_ENDPOINT in .env")
        return False
    if not model:
        print("Missing LOCAL_LLM_MODEL in .env")
        return False
    return True


def main() -> None:
    load_dotenv()
    token = get_env(TELEGRAM_BOT_TOKEN)
    endpoint = normalize_endpoint(get_env(LOCAL_LLM_ENDPOINT, "http://127.0.0.1:8088"))
    model = get_env(LOCAL_LLM_MODEL, "qwen-local")
    if not validate_settings(token, endpoint, model):
        return
    print(f"Telegram bot started. Endpoint={endpoint}, model={model}")
    print("Cloud LLM is not used.")
    bot_loop(token, endpoint, model)


if __name__ == "__main__":
    main()
