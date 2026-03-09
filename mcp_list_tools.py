import asyncio
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


async def main() -> None:
    tools = await list_mcp_tools()
    print("MCP connection: OK")
    print("Available tools:")
    for name, description in tools:
        print(f"- {name}: {description}")


if __name__ == "__main__":
    asyncio.run(main())
