import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlrequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test private local AI service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8099", help="Private service URL")
    parser.add_argument("--api-key", default="change-me-private-key", help="X-API-Key value")
    parser.add_argument("--requests", type=int, default=6, help="Total requests to send")
    parser.add_argument("--parallel", type=int, default=3, help="Concurrent workers")
    parser.add_argument("--max-tokens", type=int, default=220, help="max_tokens per request")
    parser.add_argument("--temperature", type=float, default=0.3, help="temperature per request")
    return parser.parse_args()


def post_json(url: str, payload: dict, headers: dict) -> tuple[int, dict, int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    started = time.perf_counter()
    with urlrequest.urlopen(req, timeout=60.0) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = json.loads(raw)
    return resp.status, parsed if isinstance(parsed, dict) else {}, elapsed_ms


def run_one(base_url: str, headers: dict, idx: int, max_tokens: int, temperature: float) -> dict:
    prompt = f"Запрос #{idx}: дай 3 пункта про приватный AI-сервис."
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        status, data, elapsed = post_json(f"{base_url.rstrip('/')}/v1/chat", payload, headers)
        return {
            "ok": status == 200 and bool(data.get("ok")),
            "status": status,
            "elapsed_ms": elapsed,
            "latency_ms": int(data.get("latency_ms", 0) or 0),
            "answer": str(data.get("answer", ""))[:120],
            "error": str(data.get("error", "")),
        }
    except Exception as exc:
        return {"ok": False, "status": 0, "elapsed_ms": 0, "latency_ms": 0, "answer": "", "error": str(exc)}


def main() -> None:
    args = parse_args()
    headers = {"Content-Type": "application/json", "X-API-Key": args.api_key}
    results = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [
            pool.submit(run_one, args.base_url, headers, idx + 1, args.max_tokens, args.temperature)
            for idx in range(max(1, args.requests))
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    total_ms = int((time.perf_counter() - started) * 1000)

    ok_rows = [r for r in results if r["ok"]]
    avg_latency = int(sum(r["latency_ms"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0
    print(json.dumps(
        {
            "base_url": args.base_url,
            "requests": args.requests,
            "parallel": args.parallel,
            "ok": len(ok_rows),
            "failed": len(results) - len(ok_rows),
            "success_rate": round(len(ok_rows) / max(1, len(results)), 3),
            "avg_model_latency_ms": avg_latency,
            "wall_time_ms": total_ms,
            "samples": results[:5],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
