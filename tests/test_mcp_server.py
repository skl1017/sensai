"""Tests for the MCP server (in-memory transport: no subprocess)."""

from __future__ import annotations

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from mcp_server.server import build_server
from tools.calculator import CalculatorTool


class BoomTool(CalculatorTool):
    name = "boom"

    async def run(self, expression: str) -> str:
        raise ValueError("kaboom")


async def _run(tools, scenario):
    import anyio

    server = build_server(tools)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                server.run,
                server_streams[0],
                server_streams[1],
                server.create_initialization_options(),
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                result = await scenario(session)
            tg.cancel_scope.cancel()
    return result


async def test_lists_tools_with_schema():
    calc = CalculatorTool()

    async def scenario(session):
        return await session.list_tools()

    listing = await _run([calc], scenario)
    (tool,) = listing.tools
    assert tool.name == "calculator"
    assert tool.description == calc.description
    assert tool.input_schema == calc.parameters


async def test_call_returns_run_output():
    async def scenario(session):
        return await session.call_tool("calculator", {"expression": "2+3"})

    result = await _run([CalculatorTool()], scenario)
    assert not result.is_error
    assert "5" in result.content[0].text


@pytest.mark.parametrize("name,args", [("boom", {"expression": "1"}), ("nope", {})])
async def test_errors_become_is_error_results(name, args):
    async def scenario(session):
        return await session.call_tool(name, args)

    result = await _run([BoomTool(), CalculatorTool()], scenario)
    assert result.is_error
    assert result.content[0].text


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        build_server([CalculatorTool(), CalculatorTool()])
