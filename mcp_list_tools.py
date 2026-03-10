import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def build_server_params(cli_args: list[str] | None = None) -> StdioServerParameters:
    args_list = cli_args if cli_args is not None else sys.argv[1:]
    if args_list:
        command = args_list[0]
        args = args_list[1:]
        return StdioServerParameters(command=command, args=args)
    server_path = Path(__file__).with_name("mcp_local_server.py")
    return StdioServerParameters(command=sys.executable, args=[str(server_path)])


async def list_mcp_tools(cli_args: list[str] | None = None) -> list[tuple[str, str]]:
    params = build_server_params(cli_args)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
    return [(tool.name, tool.description or "-") for tool in tools_result.tools]


def list_mcp_tools_sync(cli_args: list[str] | None = None) -> list[tuple[str, str]]:
    return asyncio.run(list_mcp_tools(cli_args))


async def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, object] | None = None,
    cli_args: list[str] | None = None,
) -> dict[str, object]:
    params = build_server_params([] if cli_args is None else cli_args)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
    text_chunks: list[str] = []
    for item in result.content:
        text = getattr(item, "text", "")
        if text:
            text_chunks.append(str(text))
    text_output = "\n".join(text_chunks).strip()
    structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
    return {
        "ok": not bool(result.isError),
        "text": text_output,
        "structured": structured,
    }


def call_mcp_tool_sync(
    tool_name: str,
    arguments: dict[str, object] | None = None,
    cli_args: list[str] | None = None,
) -> dict[str, object]:
    return asyncio.run(call_mcp_tool(tool_name, arguments, cli_args))


async def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--call":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: mcp_list_tools.py --call <tool_name> [json_args]")
        tool_name = sys.argv[2]
        args_text = sys.argv[3] if len(sys.argv) > 3 else "{}"
        args = json.loads(args_text)
        result = await call_mcp_tool(tool_name=tool_name, arguments=args)
        print("MCP call: OK" if result["ok"] else "MCP call: ERROR")
        if result["text"]:
            print(result["text"])
        if result["structured"]:
            print(json.dumps(result["structured"], ensure_ascii=False, indent=2))
        return
    tools = await list_mcp_tools()
    print("MCP connection: OK")
    print("Available tools:")
    for name, description in tools:
        print(f"- {name}: {description}")


if __name__ == "__main__":
    asyncio.run(main())
