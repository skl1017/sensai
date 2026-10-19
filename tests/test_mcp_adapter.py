"""Tests for the MCP client adapter (fake session: no network, no subprocess)."""

from __future__ import annotations

from contextlib import AsyncExitStack
from types import SimpleNamespace

import anyio
import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from config_loader import McpServerConfig
from core.tool import ITool
from tools.mcp_adapter import (
    MCPServerUnavailable,
    MCPToolAdapter,
    _Connection,
    connect_servers,
    flatten_schema,
)


def _text(*parts: str, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(text=p) for p in parts], is_error=is_error)


class FakeSession:
    """Scripted `ClientSession`: each call pops a result or raises an exception."""

    def __init__(self, script=None, tools=None):
        self.script = list(script or [])
        self.tools = tools or []
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _remote(name="echo", schema=None, description="Echo."):
    return SimpleNamespace(name=name, description=description, input_schema=schema)


def _opener(*sessions):
    """Opener handing out the given sessions in order; records how many opens happened."""
    queue = list(sessions)
    opened = []

    async def opener(config, stack):
        if not queue:
            raise ConnectionError("cannot spawn")
        session = queue.pop(0)
        opened.append(session)
        return session

    opener.opened = opened
    return opener


CFG = McpServerConfig(name="fs", command="npx")


class TestFlattenSchema:
    def test_empty_schema(self):
        assert flatten_schema(None) == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def test_forces_additional_properties_false(self):
        flat = flatten_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": True,
            }
        )
        assert flat["additionalProperties"] is False
        assert flat["required"] == ["a"]

    def test_resolves_ref_and_drops_defs(self):
        flat = flatten_schema(
            {
                "type": "object",
                "properties": {"mode": {"$ref": "#/$defs/Mode", "description": "How."}},
                "$defs": {"Mode": {"type": "string", "enum": ["r", "w"]}},
            }
        )
        assert "$defs" not in flat
        assert flat["properties"]["mode"] == {
            "type": "string",
            "enum": ["r", "w"],
            "description": "How.",
        }

    def test_nested_ref_object_gets_additional_properties_false(self):
        flat = flatten_schema(
            {
                "type": "object",
                "properties": {"opts": {"$ref": "#/$defs/Opts"}},
                "$defs": {"Opts": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            }
        )
        assert flat["properties"]["opts"]["additionalProperties"] is False

    def test_recursive_ref_terminates(self):
        flat = flatten_schema(
            {
                "type": "object",
                "properties": {"tree": {"$ref": "#/$defs/Node"}},
                "$defs": {
                    "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}
                },
            }
        )
        assert flat["properties"]["tree"]["type"] == "object"

    def test_any_of_collapses_to_first_non_null(self):
        flat = flatten_schema(
            {
                "type": "object",
                "properties": {
                    "n": {"anyOf": [{"type": "null"}, {"type": "integer"}], "description": "N."}
                },
            }
        )
        assert flat["properties"]["n"] == {"type": "integer", "description": "N."}

    def test_type_list_with_null(self):
        flat = flatten_schema({"type": "object", "properties": {"s": {"type": ["null", "string"]}}})
        assert flat["properties"]["s"]["type"] == "string"

    def test_does_not_mutate_input(self):
        original = {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}}
        flatten_schema(original)
        assert original == {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}}


class TestAdapter:
    async def test_naming_and_attributes(self):
        session = FakeSession()
        conn = _Connection(CFG, _opener(session))
        await conn.open()
        tool = MCPToolAdapter("fs", "read_file", "Read.", {"type": "object"}, conn)
        assert isinstance(tool, ITool)
        assert tool.name == "fs__read_file"
        assert tool.server == "fs"
        assert tool.remote_name == "read_file"
        assert tool.parameters["additionalProperties"] is False

    async def test_run_concatenates_text_blocks_and_ignores_others(self):
        result = CallToolResult(
            content=[
                TextContent(text="a"),
                ImageContent(data="AAAA", mime_type="image/png"),
                TextContent(text="b"),
            ]
        )
        session = FakeSession([result])
        conn = _Connection(CFG, _opener(session))
        await conn.open()
        tool = MCPToolAdapter("fs", "echo", "", None, conn)
        assert await tool.run(msg="x") == "a\nb"
        assert session.calls == [("echo", {"msg": "x"})]

    async def test_is_error_raises_with_server_text(self):
        session = FakeSession([_text("no such file", is_error=True)])
        conn = _Connection(CFG, _opener(session))
        await conn.open()
        tool = MCPToolAdapter("fs", "read_file", "", None, conn)
        with pytest.raises(RuntimeError, match="no such file"):
            await tool.run(path="/x")


class TestReconnection:
    async def test_reconnects_once_after_transport_error(self):
        dead = FakeSession([anyio.ClosedResourceError()])
        fresh = FakeSession([_text("ok")])
        opener = _opener(dead, fresh)
        conn = _Connection(CFG, opener)
        await conn.open()
        tool = MCPToolAdapter("fs", "echo", "", None, conn)
        assert await tool.run() == "ok"
        assert opener.opened == [dead, fresh]

    async def test_unavailable_when_reconnect_fails(self):
        dead = FakeSession([BrokenPipeError("pipe")])
        conn = _Connection(CFG, _opener(dead))  # no second session: reopening fails
        await conn.open()
        tool = MCPToolAdapter("fs", "echo", "", None, conn)
        with pytest.raises(MCPServerUnavailable, match="MCP server 'fs' unavailable"):
            await tool.run()

    async def test_unavailable_when_retry_fails_again(self):
        conn = _Connection(
            CFG,
            _opener(FakeSession([anyio.EndOfStream()]), FakeSession([anyio.EndOfStream()])),
        )
        await conn.open()
        tool = MCPToolAdapter("fs", "echo", "", None, conn)
        with pytest.raises(MCPServerUnavailable, match="'fs' unavailable"):
            await tool.run()

    async def test_non_transport_error_is_not_retried(self):
        session = FakeSession([ValueError("bad args")])
        opener = _opener(session, FakeSession())
        conn = _Connection(CFG, opener)
        await conn.open()
        tool = MCPToolAdapter("fs", "echo", "", None, conn)
        with pytest.raises(ValueError, match="bad args"):
            await tool.run()
        assert opener.opened == [session]


class TestConnectServers:
    async def test_lists_tools_from_every_server(self):
        s1 = FakeSession(tools=[_remote("a"), _remote("b")])
        s2 = FakeSession(tools=[_remote("a")])
        sessions = {"one": s1, "two": s2}

        async def opener(config, stack):
            return sessions[config.name]

        async with AsyncExitStack() as stack:
            tools = await connect_servers(
                [McpServerConfig("one", "x"), McpServerConfig("two", "y")], stack, opener=opener
            )
        assert [t.name for t in tools] == ["one__a", "one__b", "two__a"]
        assert [t.server for t in tools] == ["one", "one", "two"]

    async def test_startup_failure_is_unavailable(self):
        async with AsyncExitStack() as stack:
            with pytest.raises(MCPServerUnavailable, match="MCP server 'fs' unavailable"):
                await connect_servers([CFG], stack, opener=_opener())

    async def test_stack_close_closes_connection(self):
        closed = []

        async def opener(config, stack):
            stack.callback(closed.append, config.name)
            return FakeSession(tools=[])

        async with AsyncExitStack() as stack:
            await connect_servers([CFG], stack, opener=opener)
            assert closed == []
        assert closed == ["fs"]
