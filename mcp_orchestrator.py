"""Orchestration over multiple MCP servers: registration, tool routing, long flow execution."""

import json
import sys
from pathlib import Path

from mcp_list_tools import call_mcp_tool_sync
from mcp_list_tools import list_mcp_tools_sync

SERVER_ID_LOCAL = "local"
SERVER_ID_PUBLIC = "public"

# (server_id, cli_args for stdio connection)
SERVERS: dict[str, list[str]] = {
    SERVER_ID_LOCAL: [str(sys.executable), str(Path(__file__).with_name("mcp_local_server.py"))],
    SERVER_ID_PUBLIC: ["npx", "-y", "@modelcontextprotocol/server-everything"],
}


def get_tool_to_server_map() -> dict[str, str]:
    """List tools from each registered server and return tool_name -> server_id."""
    tool_to_server: dict[str, str] = {}
    for server_id, cli_args in SERVERS.items():
        try:
            tools = list_mcp_tools_sync(cli_args)
            for name, _ in tools:
                tool_to_server[name] = server_id
        except Exception:
            continue
    return tool_to_server


def run_long_flow(
    query: str,
    limit: int = 5,
    output_file: str = "orchestration_summary.txt",
) -> list[dict]:
    """
    Execute a multi-step flow using tools from different MCP servers.
    Steps: search_data (local) -> summarize_data (local) -> echo (public) -> save_to_file (local).
    Returns list of {server_id, tool_name, arguments, result}.
    """
    cli_local = SERVERS[SERVER_ID_LOCAL]
    cli_public = SERVERS[SERVER_ID_PUBLIC]
    steps: list[dict] = []

    # Step 1: search_data (local)
    step1 = call_mcp_tool_sync("search_data", {"query": query, "limit": limit}, cli_local)
    steps.append({
        "server_id": SERVER_ID_LOCAL,
        "tool_name": "search_data",
        "arguments": {"query": query, "limit": limit},
        "result": step1,
    })
    search_struct = step1.get("structured", {}) if isinstance(step1.get("structured"), dict) else {}

    # Step 2: summarize_data (local)
    step2 = call_mcp_tool_sync(
        "summarize_data",
        {"search_payload_json": json.dumps(search_struct, ensure_ascii=False)},
        cli_local,
    )
    steps.append({
        "server_id": SERVER_ID_LOCAL,
        "tool_name": "summarize_data",
        "arguments": {"search_payload_json": "<from step 1>"},
        "result": step2,
    })
    summary_text = ""
    sum_struct = step2.get("structured", {}) if isinstance(step2.get("structured"), dict) else {}
    summary_text = str(sum_struct.get("summary_text", "")).strip()

    # Step 3: echo (public) — cross-server: send summary to public MCP
    step3 = call_mcp_tool_sync(
        "echo",
        {"message": summary_text or "No summary"},
        cli_public,
    )
    steps.append({
        "server_id": SERVER_ID_PUBLIC,
        "tool_name": "echo",
        "arguments": {"message": summary_text or "No summary"},
        "result": step3,
    })

    # Step 4: save_to_file (local)
    step4 = call_mcp_tool_sync(
        "save_to_file",
        {"file_name": output_file, "content": summary_text or "No content"},
        cli_local,
    )
    steps.append({
        "server_id": SERVER_ID_LOCAL,
        "tool_name": "save_to_file",
        "arguments": {"file_name": output_file, "content": "<summary>"},
        "result": step4,
    })

    return steps
