from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("local-demo-server")


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


if __name__ == "__main__":
    mcp.run(transport="stdio")
