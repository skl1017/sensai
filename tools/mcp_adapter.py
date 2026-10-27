"""MCP client adapter: expose the tools of an MCP server as `ITool`s (wiki §8.7).

Each remote tool becomes an `MCPToolAdapter`, so the `ReActLoopNode` calls it
like any other tool. Nothing in the pipeline knows MCP exists; the only
visible traces are the tool name (`<server>__<tool>`, which keeps names unique
across servers) and the `server` / `remote_name` attributes, which let the
approval layer match a tool against the server's `read_only` list.

Connections
-----------
`connect_servers` spawns each configured server over stdio (`stdio_client` +
`ClientSession`) and registers its cleanup on the caller's `AsyncExitStack`:
the app owns the lifetime, closing the stack closes every subprocess. A
`_Connection` wraps one session and survives a dead server: on a transport
error (closed pipe, process gone) it reopens the session once and retries the
call; if that fails too, it raises `MCPServerUnavailable` with a clear message
which the ReAct loop turns into an observation. Tool timeouts are *not*
handled here: the ReAct loop already applies `tool_timeout`.

Schemas
-------
MCP servers publish arbitrary JSON Schemas (`$ref`/`$defs`, `anyOf` for
optional fields) which small local models handle badly. `flatten_schema`
inlines references, collapses `anyOf` to its first non-null branch and forces
`additionalProperties: false`, the shape every other `ITool` uses.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, CallToolResult, TextContent

from config_loader import McpServerConfig
from core.tool import ITool

MAX_REF_DEPTH = 10  # guards against self-referencing `$ref` (recursive schemas)

# Failures that mean "the pipe to the server is gone", as opposed to a tool
# error reported by a healthy server.
_TRANSPORT_ERRORS = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    OSError,  # includes ConnectionError / BrokenPipeError and spawn failures
    EOFError,
)


class MCPServerUnavailable(RuntimeError):
    """The MCP server cannot be reached, even after one reconnection attempt."""


def flatten_schema(schema: dict | None) -> dict:
    """Turn an MCP `inputSchema` into a flat JSON Schema usable as `ITool.parameters`.

    `$ref` to `$defs`/`definitions` are inlined, `anyOf`/`oneOf` collapse to
    the first non-null branch, `["string", "null"]` type lists to the first
    non-null type, and every object gets `additionalProperties: false`. The
    input is never mutated.
    """
    schema = copy.deepcopy(schema or {})
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    flat = _flatten(schema, defs, 0)
    return {
        **flat,
        "type": "object",
        "properties": flat.get("properties", {}),
        "additionalProperties": False,
    }


def _flatten(node: Any, defs: dict, depth: int) -> Any:
    if isinstance(node, list):
        return [_flatten(item, defs, depth) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {k: v for k, v in node.items() if k != "$ref"}
        target = defs.get(ref.rsplit("/", 1)[-1])
        if target is None or depth >= MAX_REF_DEPTH:
            # Unresolvable or recursive: degrade to a free-form value.
            return rest
        return _flatten({**target, **rest}, defs, depth + 1)

    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches:
            rest = {k: v for k, v in node.items() if k != key}
            chosen = next(
                (b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")),
                branches[0],
            )
            # The outer keys (description, default) win over the branch's own.
            return _flatten({**(chosen if isinstance(chosen, dict) else {}), **rest}, defs, depth)

    out = {
        k: _flatten(v, defs, depth) for k, v in node.items() if k not in ("$defs", "definitions")
    }
    type_ = out.get("type")
    if isinstance(type_, list):
        out["type"] = next((t for t in type_ if t != "null"), "string")
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out


def _result_text(result: CallToolResult) -> str:
    """Concatenate the `TextContent` blocks of a tool result (other blocks are ignored)."""
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


async def _open_session(config: McpServerConfig, stack: AsyncExitStack) -> ClientSession:
    """Spawn the server over stdio and return an initialized session owned by `stack`."""
    params = StdioServerParameters(
        command=config.command, args=list(config.args), env=config.env or None
    )
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


Opener = Callable[[McpServerConfig, AsyncExitStack], Awaitable[ClientSession]]


class _Connection:
    """One live session to one MCP server, with a single reconnection attempt."""

    def __init__(self, config: McpServerConfig, opener: Opener = _open_session) -> None:
        self.config = config
        self._opener = opener
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def open(self) -> ClientSession:
        """(Re)open the session, closing the previous one first. Returns the session."""
        await self.aclose()
        stack = AsyncExitStack()
        try:
            self._session = await self._opener(self.config, stack)
        except BaseException:
            await _quiet_close(stack)
            raise
        self._stack = stack
        return self._session

    async def aclose(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            await _quiet_close(stack)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        """Call a remote tool; on a transport error reconnect once, then give up clearly."""
        try:
            return await self._session.call_tool(name, arguments)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
        try:
            return await (await self.open()).call_tool(name, arguments)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            raise MCPServerUnavailable(
                f"MCP server {self.config.name!r} unavailable: {exc or type(exc).__name__}"
            ) from exc


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, MCPError):
        return exc.code == CONNECTION_CLOSED
    return isinstance(exc, _TRANSPORT_ERRORS)


async def _quiet_close(stack: AsyncExitStack) -> None:
    # Closing a dead transport may itself fail; that must not mask the real error.
    with suppress(Exception):
        await stack.aclose()


class MCPToolAdapter(ITool):
    """One remote MCP tool seen as an `ITool`. Named `<server>__<tool>`."""

    def __init__(
        self,
        server: str,
        remote_name: str,
        description: str,
        input_schema: dict | None,
        connection: _Connection,
    ) -> None:
        self.server = server
        self.remote_name = remote_name
        self.name = f"{server}__{remote_name}"
        self.description = description or f"Tool {remote_name!r} of MCP server {server!r}."
        self.parameters = flatten_schema(input_schema)
        self._connection = connection

    async def run(self, **kwargs) -> str:
        # `call_tool` is already async (stdio pipes under anyio): nothing blocks the loop.
        result = await self._connection.call_tool(self.remote_name, kwargs)
        text = _result_text(result)
        if result.is_error:
            raise RuntimeError(text or f"MCP tool {self.name!r} failed")
        return text


async def connect_servers(
    configs: list[McpServerConfig],
    stack: AsyncExitStack,
    *,
    opener: Opener = _open_session,
) -> list[ITool]:
    """Connect every configured server and return all their tools as `ITool`s.

    Connections live as long as `stack`: closing it shuts the subprocesses
    down. A server that cannot start raises `MCPServerUnavailable`. `opener`
    is a seam for tests (a fake session instead of a real subprocess).
    """
    tools: list[ITool] = []
    for config in configs:
        connection = _Connection(config, opener)
        try:
            session = await connection.open()
            listing = await session.list_tools()
        except Exception as exc:
            await connection.aclose()
            raise MCPServerUnavailable(
                f"MCP server {config.name!r} unavailable: {exc or type(exc).__name__}"
            ) from exc
        stack.push_async_callback(connection.aclose)
        tools.extend(
            MCPToolAdapter(config.name, t.name, t.description or "", t.input_schema, connection)
            for t in listing.tools
        )
    return tools
