import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv

from dev_assistant_rag import build_dev_assistant_local_llm_prompt

ROOT = Path(__file__).resolve().parent
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


def http_post_json(
    url: str,
    payload: dict,
    timeout_sec: float = 25.0,
    ignore_proxy: bool = False,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({})) if ignore_proxy else None
    open_fn = opener.open if opener else urlrequest.urlopen
    with open_fn(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def telegram_delete_webhook(token: str) -> dict:
    """Снять webhook — иначе getUpdates даёт HTTP 409 Conflict."""
    return http_post_json(
        telegram_api_url(token, "deleteWebhook"),
        {"drop_pending_updates": True},
        timeout_sec=15.0,
        ignore_proxy=True,
    )


def telegram_get_webhook_info(token: str) -> dict:
    return http_post_json(
        telegram_api_url(token, "getWebhookInfo"),
        {},
        timeout_sec=15.0,
        ignore_proxy=True,
    )


def _telegram_error_description(exc: urlerror.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
    except Exception:
        return str(exc)
    try:
        data = json.loads(raw)
        return str(data.get("description", raw))
    except Exception:
        return raw


def _webhook_info_summary(info: dict) -> str:
    if not info.get("ok"):
        return f"getWebhookInfo: {info}"
    res = info.get("result")
    if not isinstance(res, dict):
        return str(info)
    url = (res.get("url") or "").strip()
    pending = res.get("pending_update_count", 0)
    if url:
        return f"webhook_url={url!r}, pending_updates={pending}"
    return f"webhook не задан, pending_updates={pending}"


def poll_timeout_sec() -> int:
    raw = os.getenv("TELEGRAM_POLL_TIMEOUT", "20").strip()
    try:
        return max(0, min(int(raw), 50))
    except ValueError:
        return 20


def telegram_prepare_polling(token: str) -> None:
    try:
        r = telegram_delete_webhook(token)
        if r.get("ok"):
            print("[bot] deleteWebhook OK — long polling (getUpdates)")
        else:
            print(f"[bot] deleteWebhook: {r}")
        info = telegram_get_webhook_info(token)
        print(f"[bot] {_webhook_info_summary(info)}")
    except Exception as exc:
        print(f"[bot] deleteWebhook пропущен: {exc}")


def send_tg_message(token: str, chat_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text[:4000]}
    try:
        http_post_json(
            telegram_api_url(token, "sendMessage"),
            payload,
            timeout_sec=15.0,
            ignore_proxy=True,
        )
    except Exception:
        return


def fetch_tg_updates(token: str, offset: int, *, long_poll_timeout: int | None = None) -> list[dict]:
    tout = poll_timeout_sec() if long_poll_timeout is None else max(0, min(long_poll_timeout, 50))
    payload = {"timeout": tout, "offset": offset, "allowed_updates": ["message"]}
    response = http_post_json(
        telegram_api_url(token, "getUpdates"),
        payload,
        ignore_proxy=True,
    )
    if not response.get("ok"):
        return []
    result = response.get("result", [])
    return result if isinstance(result, list) else []


def call_local_llm(
    endpoint: str,
    model: str,
    user_text: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 700,
) -> str:
    messages: list[dict[str, str]] = []
    if system_instruction and system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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


def extract_help_question(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("/help"):
        return None
    rest = stripped[5:].strip()
    if not rest:
        return ""
    if rest.startswith("@"):
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return ""
        return parts[1].strip()
    return rest


def extract_review_target(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("/review_pr"):
        return None
    rest = stripped[len("/review_pr") :].strip()
    if not rest:
        return ""
    if rest.startswith("@"):
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return ""
        return parts[1].strip()
    return rest


def _repo_slug() -> str:
    value = os.getenv("GITHUB_REPO", "").strip()
    return value or "nixusUM/llm-api-mini-project"


def _pr_number(spec: str) -> str:
    cleaned = spec.strip()
    if cleaned.isdigit():
        return cleaned
    match = re.search(r"/pull/(\d+)", cleaned)
    if match:
        return match.group(1)
    raise ValueError("Укажите номер PR (например `1`) или ссылку `.../pull/<id>`.")


def _gh(*args: str) -> str:
    proc = subprocess.run(
        ["gh", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "gh failed").strip()
        raise RuntimeError(msg)
    return (proc.stdout or "").strip()


def _pr_range(spec: str) -> tuple[str, str, str]:
    number = _pr_number(spec)
    repo = _repo_slug()
    out = _gh(
        "pr",
        "view",
        number,
        "--repo",
        repo,
        "--json",
        "url,baseRefOid,headRefOid",
    )
    data = json.loads(out)
    return str(data["baseRefOid"]), str(data["headRefOid"]), str(data["url"])


def review_pr_reply(spec: str) -> str:
    try:
        base, head, pr_url = _pr_range(spec)
        with tempfile.NamedTemporaryFile(prefix="tg_pr_review_", suffix=".md", delete=False) as tmp:
            out_path = tmp.name
        _ = _gh(
            "api",
            "repos/" + _repo_slug(),
            "--method",
            "GET",
        )
        proc = subprocess.run(
            [
                "python3",
                "scripts/ai_pr_review.py",
                "--base",
                base,
                "--head",
                head,
                "--output",
                out_path,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "review script failed").strip()
            raise RuntimeError(err)
        review = Path(out_path).read_text(encoding="utf-8", errors="ignore").strip()
        review = review.replace("<!-- ai-pr-review -->", "").strip()
        return f"PR: {pr_url}\n\n{review[:3600]}"
    except Exception as exc:
        return (
            "Не удалось выполнить ревью PR.\n"
            f"Причина: {exc}\n\n"
            "Проверьте: `gh auth status`, наличие PR и что `scripts/ai_pr_review.py` доступен."
        )


def help_assistant_reply(endpoint: str, model: str, question: str) -> str:
    prompt, _ = build_dev_assistant_local_llm_prompt(question, use_mcp_context=None)
    return call_local_llm(
        endpoint,
        model,
        prompt,
        system_instruction=None,
        temperature=0.15,
        max_tokens=900,
    )


def handle_command(text: str) -> str | None:
    if text == "/start":
        return (
            "Привет! Я Telegram-бот на локальной LLM.\n"
            "Отправь любой текст, и я отвечу через локальную модель.\n"
            "Команды:\n"
            "/start — приветствие\n"
            "/health — проверить локальную LLM\n"
            "/help <вопрос> — ассистент по README, docs/ и контексту git репозитория\n"
            "/review_pr <id|url> — AI code review для PR (diff + RAG + рекомендации)"
        )
    if text == "/health":
        return "Проверяю через следующий запрос..."
    return None


def bot_loop(token: str, endpoint: str, model: str) -> None:
    offset = 0
    conflict_streak = 0
    while True:
        try:
            updates = fetch_tg_updates(token, offset)
            conflict_streak = 0
            for update in updates:
                update_id = int(update.get("update_id", 0))
                offset = max(offset, update_id + 1)
                payload = extract_message(update)
                if payload is None:
                    continue
                chat_id, text = payload
                review_target = extract_review_target(text)
                if review_target is not None:
                    if not review_target:
                        send_tg_message(
                            token,
                            chat_id,
                            "Укажите PR для ревью.\nПример: /review_pr 1\nИли: /review_pr https://github.com/<org>/<repo>/pull/1",
                        )
                        continue
                    send_tg_message(token, chat_id, "Запускаю AI-ревью PR…")
                    send_tg_message(token, chat_id, review_pr_reply(review_target))
                    continue
                help_q = extract_help_question(text)
                if help_q is not None:
                    if not help_q:
                        send_tg_message(
                            token,
                            chat_id,
                            "Ассистент разработчика: задайте вопрос об этом проекте.\n"
                            "Пример: /help Как запустить Telegram-бота?\n"
                            "Используются README, папка docs/, ветка git и список файлов.",
                        )
                        continue
                    send_tg_message(token, chat_id, "Ищу в документации и контексте репозитория…")
                    reply = help_assistant_reply(endpoint, model, help_q)
                    send_tg_message(token, chat_id, reply)
                    continue
                command_reply = handle_command(text)
                if command_reply is not None:
                    send_tg_message(token, chat_id, command_reply)
                    if text != "/health":
                        continue
                    text = "Ответь одним словом: ok"
                answer = call_local_llm(endpoint, model, text)
                send_tg_message(token, chat_id, answer)
        except urlerror.HTTPError as exc:
            if exc.code == 409:
                conflict_streak += 1
                detail = _telegram_error_description(exc)
                print(f"[bot] HTTP 409: {detail}")
                try:
                    telegram_delete_webhook(token)
                except Exception as del_exc:
                    print(f"[bot] deleteWebhook не удался: {del_exc}")
                try:
                    print(f"[bot] {_webhook_info_summary(telegram_get_webhook_info(token))}")
                except Exception:
                    pass
                if conflict_streak == 1:
                    print(
                        "[bot] Если только что был другой long poll (другой процесс или старый запуск), "
                        "Telegram держит слот до ~20–50 с. Жду 23 с…"
                    )
                    time.sleep(23)
                else:
                    time.sleep(min(5 + conflict_streak * 3, 45))
                if conflict_streak >= 4:
                    print(
                        "[bot] 409 не проходит: остановите ВСЕ другие экземпляры бота "
                        "(например: pgrep -fl telegram_local_bot) и не используйте этот токен на сервере с webhook."
                    )
                continue
            print(f"[bot] polling error: HTTP {exc.code}: {exc}")
            time.sleep(2)
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
    telegram_prepare_polling(token)
    print(f"Telegram bot started. Endpoint={endpoint}, model={model}")
    print("Cloud LLM is not used.")
    bot_loop(token, endpoint, model)


if __name__ == "__main__":
    main()
