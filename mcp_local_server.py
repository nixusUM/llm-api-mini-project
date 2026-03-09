from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-demo-server")


@mcp.tool()
def ping() -> str:
    return "pong"


@mcp.tool()
def sum_two_numbers(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
