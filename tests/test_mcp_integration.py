"""Integration: MCPToolAdapter against real MCP servers over stdio (marked slow)."""

from __future__ import annotations

import shutil
import sys
from contextlib import AsyncExitStack

import pytest

from config_loader import McpServerConfig
from tools.mcp_adapter import connect_servers

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("npx") is None, reason="npx not installed")
async def test_reference_filesystem_server(tmp_path):
    (tmp_path / "hello.txt").write_text("hello mcp")
    config = McpServerConfig(
        name="fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
    )
    async with AsyncExitStack() as stack:
        tools = {t.name: t for t in await connect_servers([config], stack)}
        assert "fs__read_text_file" in tools or "fs__read_file" in tools
        read = tools.get("fs__read_text_file") or tools["fs__read_file"]
        out = await read.run(path=str(tmp_path / "hello.txt"))
        assert "hello mcp" in out


async def test_home_server_calculator():
    config = McpServerConfig(
        name="sensai", command=sys.executable, args=["-m", "mcp_server.server"]
    )
    async with AsyncExitStack() as stack:
        tools = {t.name: t for t in await connect_servers([config], stack)}
        calc = tools["sensai__calculator"]
        assert "5" in await calc.run(expression="2+3")
        with pytest.raises(RuntimeError):
            await calc.run(expression="__import__('os')")
