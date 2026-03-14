import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-demo-server")
STATE_PATH = Path(__file__).with_name("data").joinpath("mcp_periodic_state.json")
PIPELINE_OUTPUTS_DIR = Path(__file__).with_name("data").joinpath("pipeline_outputs")


@mcp.tool()
def ping() -> str:
    return "pong"


@mcp.tool()
def sum_two_numbers(a: int, b: int) -> int:
    return a + b


@mcp.tool()
def get_todo_from_mock_api(todo_id: int) -> dict[str, object]:
    """Fetch one todo item from JSONPlaceholder mock API by todo_id."""
    if todo_id < 1:
        return {"ok": False, "error": "todo_id must be >= 1"}
    url = f"https://jsonplaceholder.typicode.com/todos/{todo_id}"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"request_failed: {exc}"}
    if not isinstance(payload, dict) or not payload.get("id"):
        return {"ok": False, "error": "todo_not_found"}
    return {
        "ok": True,
        "source": "jsonplaceholder",
        "todo": {
            "id": payload.get("id"),
            "title": payload.get("title", ""),
            "completed": bool(payload.get("completed", False)),
            "userId": payload.get("userId"),
        },
    }


@mcp.tool()
def search_data(query: str, limit: int = 5) -> dict[str, object]:
    """Search mock post data by query string (title/body)."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"ok": False, "error": "query is empty"}
    top_n = max(1, min(limit, 20))
    url = "https://jsonplaceholder.typicode.com/posts"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"request_failed: {exc}"}
    rows = payload if isinstance(payload, list) else []
    lowered = cleaned_query.lower()
    matches: list[dict[str, object]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", ""))
        body = str(item.get("body", ""))
        haystack = f"{title}\n{body}".lower()
        if lowered not in haystack:
            continue
        matches.append(
            {
                "id": int(item.get("id", 0) or 0),
                "title": title[:140],
                "body_excerpt": body[:220],
            }
        )
        if len(matches) >= top_n:
            break
    return {
        "ok": True,
        "query": cleaned_query,
        "returned_count": len(matches),
        "items": matches,
    }


@mcp.tool()
def summarize_data(search_payload_json: str) -> dict[str, object]:
    """Summarize search tool output (expects JSON string from search_data)."""
    try:
        payload = json.loads(search_payload_json)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid search_payload_json"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "search payload must be object"}
    items = payload.get("items", [])
    if not isinstance(items, list):
        return {"ok": False, "error": "items must be a list"}
    query = str(payload.get("query", "")).strip() or "-"
    count = len(items)
    titles = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        titles.append(str(item.get("title", "")).strip())
    if count == 0:
        summary = f"No matches found for query '{query}'."
    else:
        titles_text = "; ".join(title for title in titles if title) or "No titles."
        summary = (
            f"Search summary for '{query}': {count} matches. "
            f"Top titles: {titles_text}"
        )
    return {
        "ok": True,
        "query": query,
        "items_count": count,
        "summary_text": summary,
    }


@mcp.tool()
def save_to_file(file_name: str, content: str) -> dict[str, object]:
    """Save text content to data/pipeline_outputs and return file metadata."""
    base_name = file_name.strip().replace(" ", "_")
    if not base_name:
        base_name = "pipeline_summary.txt"
    if not base_name.endswith(".txt"):
        base_name += ".txt"
    safe_name = "".join(ch for ch in base_name if ch.isalnum() or ch in {"_", "-", "."})[:80]
    if not safe_name:
        safe_name = "pipeline_summary.txt"
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PIPELINE_OUTPUTS_DIR.joinpath(safe_name)
    text = content.strip()
    output_path.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "file_path": str(output_path),
        "bytes_written": len(text.encode("utf-8")),
    }


@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> dict[str, object]:
    """Get current weather for a city (Open-Meteo, no API key). Example: get_weather('London') or get_weather('Moscow', 'celsius')."""
    name = (city or "").strip()
    if not name:
        return {"ok": False, "error": "city is empty"}
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        geo = requests.get(geocode_url, params={"name": name, "count": 1}, timeout=8)
        geo.raise_for_status()
        data = geo.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results or not isinstance(results[0], dict):
            return {"ok": False, "error": f"city not found: {name}"}
        lat = results[0].get("latitude")
        lon = results[0].get("longitude")
        display_name = results[0].get("name", name)
        country = results[0].get("country_code", "")
    except requests.RequestException as exc:
        return {"ok": False, "error": f"geocoding failed: {exc}"}
    temp_unit = "fahrenheit" if (units or "").strip().lower() == "fahrenheit" else "celsius"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    try:
        w = requests.get(weather_url, params=params, timeout=8)
        w.raise_for_status()
        wdata = w.json()
        current = (wdata.get("current") or {}) if isinstance(wdata, dict) else {}
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        code = current.get("weather_code")
        wind = current.get("wind_speed_10m")
    except requests.RequestException as exc:
        return {"ok": False, "error": f"weather request failed: {exc}"}
    return {
        "ok": True,
        "source": "open-meteo",
        "place": f"{display_name}, {country}" if country else display_name,
        "units": temp_unit,
        "temperature": temp,
        "humidity_percent": humidity,
        "weather_code": code,
        "wind_speed_kmh": wind,
    }


@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str) -> dict[str, object]:
    """Get latest exchange rate (Frankfurter API, free). Example: get_exchange_rate('USD', 'EUR')."""
    fc = (from_currency or "USD").strip().upper()[:3]
    tc = (to_currency or "EUR").strip().upper()[:3]
    if not fc or not tc:
        return {"ok": False, "error": "from_currency and to_currency required"}
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": fc, "to": tc},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        rate = data.get("rates", {}).get(tc) if isinstance(data, dict) else None
        date = data.get("date", "") if isinstance(data, dict) else ""
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    if rate is None:
        return {"ok": False, "error": f"rate not found for {fc} -> {tc}"}
    return {
        "ok": True,
        "source": "frankfurter",
        "from_currency": fc,
        "to_currency": tc,
        "rate": rate,
        "date": date,
    }


@mcp.tool()
def get_random_quote() -> dict[str, object]:
    """Get a random quote (Quotable API, free)."""
    try:
        r = requests.get("https://api.quotable.io/random", timeout=6)
        r.raise_for_status()
        data = r.json()
        content = data.get("content", "") if isinstance(data, dict) else ""
        author = data.get("author", "") if isinstance(data, dict) else ""
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "source": "quotable",
        "quote": content,
        "author": author,
    }


def _default_state() -> dict[str, object]:
    return {"jobs": {}, "history": []}


def _load_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(parsed, dict):
        return _default_state()
    jobs = parsed.get("jobs", {})
    history = parsed.get("history", [])
    if not isinstance(jobs, dict) or not isinstance(history, list):
        return _default_state()
    return {"jobs": jobs, "history": history[-500:]}


def _save_state(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    STATE_PATH.write_text(payload, encoding="utf-8")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_text() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _fetch_todo_metrics(user_id: int) -> dict[str, object]:
    url = f"https://jsonplaceholder.typicode.com/todos?userId={user_id}"
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else []
    total = len(rows)
    completed = sum(1 for item in rows if isinstance(item, dict) and bool(item.get("completed", False)))
    pending = max(0, total - completed)
    rate = round((completed / total) * 100, 2) if total else 0.0
    return {
        "user_id": user_id,
        "total": total,
        "completed": completed,
        "pending": pending,
        "completion_rate": rate,
    }


def _aggregate_history(history: list[dict], limit: int) -> dict[str, object]:
    if not history:
        return {"runs": 0, "avg_completion_rate": 0.0}
    window = history[-max(1, min(limit, 100)) :]
    rates = [float(item.get("summary", {}).get("completion_rate", 0.0)) for item in window if isinstance(item, dict)]
    avg_rate = round(sum(rates) / len(rates), 2) if rates else 0.0
    return {"runs": len(window), "avg_completion_rate": avg_rate}


@mcp.tool()
def configure_periodic_summary(
    job_id: str,
    interval_seconds: int = 60,
    user_id: int = 1,
    enabled: bool = True,
) -> dict[str, object]:
    """Create or update periodic summary job with schedule settings."""
    normalized_id = job_id.strip().lower().replace(" ", "_")[:40]
    if not normalized_id:
        return {"ok": False, "error": "job_id is empty"}
    interval = max(10, min(interval_seconds, 86_400))
    user = max(1, min(user_id, 10))
    state = _load_state()
    jobs = state["jobs"] if isinstance(state["jobs"], dict) else {}
    now = _utc_now().timestamp()
    jobs[normalized_id] = {
        "job_id": normalized_id,
        "interval_seconds": interval,
        "user_id": user,
        "enabled": bool(enabled),
        "next_run_at": now,
        "updated_at": _utc_now_text(),
    }
    state["jobs"] = jobs
    _save_state(state)
    return {"ok": True, "job": jobs[normalized_id]}


@mcp.tool()
def run_due_summaries() -> dict[str, object]:
    """Execute all due periodic jobs and return aggregated summary."""
    state = _load_state()
    now = _utc_now().timestamp()
    jobs = state["jobs"] if isinstance(state["jobs"], dict) else {}
    history = state["history"] if isinstance(state["history"], list) else []
    executed: list[dict[str, object]] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict) or not bool(job.get("enabled", True)):
            continue
        next_run_at = float(job.get("next_run_at", 0.0) or 0.0)
        if now < next_run_at:
            continue
        try:
            summary = _fetch_todo_metrics(int(job.get("user_id", 1)))
        except requests.RequestException as exc:
            summary = {"error": str(exc)}
        run_item = {
            "job_id": job_id,
            "run_at": _utc_now_text(),
            "summary": summary,
        }
        history.append(run_item)
        executed.append(run_item)
        interval = int(job.get("interval_seconds", 60))
        job["next_run_at"] = now + max(10, interval)
    state["history"] = history[-500:]
    state["jobs"] = jobs
    _save_state(state)
    aggregate = _aggregate_history(state["history"], limit=20)
    return {"ok": True, "executed_jobs": executed, "aggregate": aggregate}


@mcp.tool()
def get_summary_report(limit: int = 10) -> dict[str, object]:
    """Return latest periodic runs and aggregated completion metrics."""
    state = _load_state()
    history = state["history"] if isinstance(state["history"], list) else []
    jobs = state["jobs"] if isinstance(state["jobs"], dict) else {}
    window = history[-max(1, min(limit, 100)) :]
    aggregate = _aggregate_history(history, limit=max(1, min(limit, 100)))
    return {
        "ok": True,
        "jobs_total": len(jobs),
        "history_total": len(history),
        "latest_runs": window,
        "aggregate": aggregate,
    }


@mcp.tool()
def clear_periodic_state() -> dict[str, object]:
    """Delete all periodic jobs/history and reset scheduler storage."""
    state = _default_state()
    _save_state(state)
    return {"ok": True, "jobs_total": 0, "history_total": 0}


if __name__ == "__main__":
    mcp.run(transport="stdio")
