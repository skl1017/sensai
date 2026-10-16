"""MCP server: expose `ITool`s over the Model Context Protocol (wiki §8.7).

Each `ITool` is published as an MCP tool (`name`, `description`,
`inputSchema = parameters`) and `tools/call` delegates to `ITool.run`. An
exception raised by a tool becomes an `isError` result carrying the message,
never a protocol failure, mirroring the "errors are observations" rule.

Run standalone: `python -m mcp_server.server` (stdio, serves the calculator).
"""

from __future__ import annotations

import anyio
from mcp import types
from mcp.server import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from core.tool import ITool
from tools.calculator import CalculatorTool

SERVER_NAME = "sensai"


def build_server(tools: list[ITool]) -> Server:
    """Build a low-level MCP `Server` serving `tools`."""
    by_name = {t.name: t for t in tools}
    if len(by_name) != len(tools):
        raise ValueError("duplicate tool names")

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, inputSchema=t.parameters)
                for t in tools
            ]
        )

    async def call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        tool = by_name.get(params.name)
        if tool is None:
            return _result(f"Unknown tool: {params.name}", is_error=True)
        try:
            text = await tool.run(**(params.arguments or {}))
        except Exception as exc:
            return _result(str(exc) or type(exc).__name__, is_error=True)
        return _result(text, is_error=False)

    return Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)


def _result(text: str, *, is_error: bool) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=text)], isError=is_error)


async def serve(tools: list[ITool]) -> None:
    """Serve `tools` over stdio until the client disconnects."""
    server = build_server(tools)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    anyio.run(serve, [CalculatorTool()])


if __name__ == "__main__":
    main()
