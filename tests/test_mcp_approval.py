"""MCP approval rule and Agent.aclose wiring (no network: fake adapters)."""

from __future__ import annotations

from contextlib import AsyncExitStack

import app
from app import build_agent, needs_approval
from config_loader import AgentConfig, LLMConfig, McpServerConfig
from core.agent import Agent
from tests.fakes import FakeLLM, FakeTool
from tools.approved_tool import ApprovedTool
from tools.mcp_adapter import MCPToolAdapter


class _NoopUI:
    def on_token(self, text: str) -> None:
        pass

    def on_event(self, kind: str, data: dict) -> None:
        pass

    async def ask(self, question: str, kind: str = "confirm") -> str:
        return "no"


def _adapter(server: str, remote: str) -> MCPToolAdapter:
    return MCPToolAdapter(server, remote, "d", None, connection=None)  # type: ignore[arg-type]


def _cfg(**kw) -> AgentConfig:
    servers = [McpServerConfig(name="fs", command="x", read_only=["read"])]
    return AgentConfig(llm=LLMConfig(provider="ollama", model="m"), mcp_servers=servers, **kw)


def test_mcp_read_only_needs_no_approval():
    assert not needs_approval(_adapter("fs", "read"), _cfg())


def test_mcp_not_read_only_needs_approval():
    assert needs_approval(_adapter("fs", "write"), _cfg())


def test_mcp_unknown_server_needs_approval():
    assert needs_approval(_adapter("other", "read"), _cfg())


def test_non_mcp_tool_keeps_fnmatch_rule():
    cfg = _cfg(approval=["write_*"])
    assert needs_approval(FakeTool("write_file"), cfg)
    assert not needs_approval(FakeTool("calculator"), cfg)


def _write_config(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "llm:\\n  provider: ollama\\n  model: m\\n".replace("\\n", "\n")
        + "pipeline:\n  - react\ntools: []\n"
        + "mcp_servers:\n  - name: fs\n    command: x\n    read_only: [read]\n"
    )
    return str(path)


async def test_build_agent_wraps_non_read_only_and_aclose_closes_stack(monkeypatch, tmp_path):
    closed: list[str] = []

    async def fake_connect(configs, stack):
        stack.push_async_callback(_record(closed))
        return [_adapter("fs", "read"), _adapter("fs", "write")]

    monkeypatch.setattr(app, "connect_servers", fake_connect)
    agent = await build_agent(_write_config(tmp_path), _NoopUI(), llm=FakeLLM([]))

    assert isinstance(agent.tools["fs__write"], ApprovedTool)
    assert isinstance(agent.tools["fs__read"], MCPToolAdapter)
    assert closed == []
    await agent.aclose()
    assert closed == ["closed"]
    await agent.aclose()  # idempotent
    assert closed == ["closed"]


async def test_build_agent_closes_stack_on_failure(monkeypatch, tmp_path):
    closed: list[str] = []

    async def fake_connect(configs, stack):
        stack.push_async_callback(_record(closed))
        return [_adapter("fs", "read")]

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "connect_servers", fake_connect)
    monkeypatch.setattr(app, "Agent", boom)
    try:
        await build_agent(_write_config(tmp_path), _NoopUI(), llm=FakeLLM([]))
    except RuntimeError:
        pass
    assert closed == ["closed"]


async def test_agent_aclose_without_stack_is_noop():
    await Agent(FakeLLM([]), [], []).aclose()


async def test_agent_aclose_closes_given_stack():
    closed: list[str] = []
    stack = AsyncExitStack()
    stack.push_async_callback(_record(closed))
    await Agent(FakeLLM([]), [], [], stack=stack).aclose()
    assert closed == ["closed"]


def _record(closed: list[str]):
    async def cb() -> None:
        closed.append("closed")

    return cb
