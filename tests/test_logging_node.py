"""Tests for `LoggingNode` and its measuring proxies."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from core import runtime
from core.agent import Agent
from core.node import PipelineNode
from core.types import Context, ToolCall
from nodes.logging_node import LoggingNode, MeasuredLLM, MeasuredTool
from nodes.react_loop import ReActLoopNode
from storage.event_log import EventLog, read_events
from tests.fakes import FakeLLM, FakeTool


def make_agent(tmp_path, script, tools=(), extra_nodes=(), **kwargs):
    path = tmp_path / "events.jsonl"
    node = LoggingNode(EventLog.open(path), **kwargs)
    agent = Agent(FakeLLM(script), list(tools), [node, *extra_nodes, ReActLoopNode(tool_timeout=5)])
    return agent, path


async def test_success_event(tmp_path):
    agent, path = make_agent(tmp_path, ["hello there world"])
    state = {
        "session_id": "s1",
        "persona": "default",
        "prompt": {"persona": "default", "template": "base", "version": "v2"},
        "parent_trace_id": "p1",
    }
    ctx = await agent.run("hi", [], state)
    (event,) = read_events(path)
    assert event["trace_id"] == ctx.state["trace_id"]
    assert event["status"] == "ok"
    assert event["error"] is None
    assert event["input"] == "hi"
    assert event["response"] == "hello there world"
    assert event["session_id"] == "s1"
    assert event["persona"] == "default"
    assert event["prompt_version"] == "base@v2"
    assert event["parent_trace_id"] == "p1"
    assert event["cache_hit"] is None and event["blocked"] is None
    assert event["react_truncated"] is False
    assert event["eval_scores"] is None and event["pii_found"] is None
    assert event["ts"].endswith("+00:00")
    assert len(event["llm_calls"]) == 1
    call = event["llm_calls"][0]
    assert call["structured"] is False and call["error"] is None
    assert call["ttft_ms"] is not None and call["latency_ms"] >= call["ttft_ms"]


async def test_metrics_and_tokens_from_done_chunk(tmp_path):
    tool = FakeTool("t", result="r")
    agent, path = make_agent(tmp_path, [[ToolCall("1", "t", {})], "final answer"], tools=[tool])
    ctx = await agent.run("go", [])
    llm = agent.llm
    expected_prompt = sum(llm.count_tokens(c["messages"]) for c in llm.calls)
    (event,) = read_events(path)
    assert len(event["llm_calls"]) == 2
    assert event["prompt_tokens"] == sum(c["prompt_tokens"] for c in event["llm_calls"])
    assert event["prompt_tokens"] > 0 and event["completion_tokens"] > 0
    assert event["tool_calls"] == [
        {"name": "t", "latency_ms": event["tool_calls"][0]["latency_ms"], "error": None}
    ]
    m = ctx.state["metrics"]
    assert m["llm_calls"] == 2 and m["tool_calls"] == 1
    assert m["prompt_tokens"] == event["prompt_tokens"]
    assert m["completion_tokens"] == event["completion_tokens"]
    assert m["latency_ms"] == event["latency_ms"]
    assert m["ttft_ms"] == event["llm_calls"][0]["ttft_ms"]
    assert expected_prompt >= 0


async def test_tool_error_recorded_but_request_ok(tmp_path):
    tool = FakeTool("t", result=ValueError("boom"))
    agent, path = make_agent(tmp_path, [[ToolCall("1", "t", {})], "done"], tools=[tool])
    ctx = await agent.run("go", [])
    (event,) = read_events(path)
    assert event["status"] == "ok"
    assert event["tool_calls"][0]["error"] == "ValueError"
    assert ctx.response == "done"


async def test_exception_logged_and_reraised(tmp_path):
    agent, path = make_agent(tmp_path, [])  # FakeLLM raises AssertionError (script exhausted)
    with pytest.raises(AssertionError):
        await agent.run("hi", [])
    (event,) = read_events(path)
    assert event["status"] == "error"
    assert event["error"]["type"] == "AssertionError"
    assert "exhausted" in event["error"]["message"]
    assert event["llm_calls"][0]["error"] == "AssertionError"
    assert runtime.trace_id.get() is None


async def test_cancellation(tmp_path):
    started = asyncio.Event()

    async def hang(**_):
        started.set()
        await asyncio.sleep(60)
        return "never"

    tool = FakeTool("slow", result=hang, enforces_own_timeout=True)
    agent, path = make_agent(tmp_path, [[ToolCall("1", "slow", {})], "x"], tools=[tool])
    task = asyncio.create_task(agent.run("go", []))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (event,) = read_events(path)
    assert event["status"] == "cancelled"
    assert event["error"] is None
    assert event["tool_calls"][0]["name"] == "slow"


class _CacheHit(PipelineNode):
    async def handle(self, ctx, next):
        ctx.response = "cached"
        ctx.state["cache_hit"] = True
        return ctx


async def test_short_circuit_has_no_llm_calls(tmp_path):
    path = tmp_path / "e.jsonl"
    llm = FakeLLM([])
    agent = Agent(llm, [], [LoggingNode(EventLog.open(path)), _CacheHit()])
    await agent.run("hi", [])
    (event,) = read_events(path)
    assert event["llm_calls"] == [] and event["tool_calls"] == []
    assert event["cache_hit"] is True
    assert event["response"] == "cached"
    assert event["prompt_tokens"] == 0
    assert llm.calls == []


async def test_redact_inputs(tmp_path):
    agent, path = make_agent(tmp_path, ["ok"], redact_inputs=True)
    await agent.run("my secret", [])
    (event,) = read_events(path)
    assert event["input"] == "sha256:" + hashlib.sha256(b"my secret").hexdigest()
    assert "secret" not in event["input"]


async def test_restores_llm_and_tools_and_does_not_mutate_agent_tools(tmp_path):
    tool = FakeTool("t")
    agent, _ = make_agent(tmp_path, [[ToolCall("1", "t", {})], "done"], tools=[tool])
    original = dict(agent.tools)
    seen: dict = {}

    class Spy(PipelineNode):
        async def handle(self, ctx, next):
            seen["llm"] = ctx.llm
            seen["tools"] = ctx.tools
            return await next(ctx)

    agent2 = Agent(agent.llm, [tool], [agent.pipeline[0], Spy(), ReActLoopNode()])
    ctx = await agent2.run("go", [])
    assert isinstance(seen["llm"], MeasuredLLM)
    assert isinstance(seen["tools"]["t"], MeasuredTool)
    assert seen["tools"] is not agent2.tools
    assert ctx.llm is agent2.llm
    assert ctx.tools is agent2.tools
    assert agent2.tools["t"] is tool
    assert agent.tools == original


async def test_restores_on_error(tmp_path):
    agent, _ = make_agent(tmp_path, [], tools=[FakeTool("t")])
    ctx = Context("hi", [], agent.llm, agent.tools)
    with pytest.raises(RuntimeError):
        await agent.pipeline[0].handle(ctx, _boom)
    assert ctx.llm is agent.llm and ctx.tools is agent.tools


async def _boom(ctx):
    raise RuntimeError("x")


async def test_tool_sees_trace_id(tmp_path):
    seen = []

    def grab(**_):
        seen.append(runtime.trace_id.get())
        return "ok"

    tool = FakeTool("t", result=grab)
    agent, path = make_agent(tmp_path, [[ToolCall("1", "t", {})], "done"], tools=[tool])
    ctx = await agent.run("go", [])
    assert seen == [ctx.state["trace_id"]]
    assert read_events(path)[0]["trace_id"] == seen[0]
    assert runtime.trace_id.get() is None


async def test_proxies_are_transparent(tmp_path):
    inner_llm = FakeLLM(["x"], model="m1", max_context_tokens=123)
    calls: list[dict] = []
    llm = MeasuredLLM(inner_llm, calls)
    assert llm.model == "m1" and llm.max_context_tokens == 123
    llm.update_options(temperature=0.5)
    assert llm.options == {"temperature": 0.5} == inner_llm.options
    assert llm.remaining == 1  # __getattr__ fallback
    assert llm.count_tokens([]) == 0
    chunks = [c async for c in llm.chat([], schema={"type": "object"})]
    assert chunks[-1].done
    assert calls[0]["structured"] is True

    inner_tool = FakeTool("t", description="d", enforces_own_timeout=True)
    inner_tool.persist_exchange = True
    tool = MeasuredTool(inner_tool, [])
    assert (tool.name, tool.description) == ("t", "d")
    assert tool.parameters is inner_tool.parameters
    assert tool.enforces_own_timeout is True and tool.persist_exchange is True
    assert tool.result == "ok"  # __getattr__ fallback
    assert await tool.run(a=1) == "ok" and inner_tool.calls == [{"a": 1}]


async def test_log_failure_never_masks_result_or_error(tmp_path, caplog):
    class Broken:
        async def write(self, event):
            raise OSError("disk full")

    agent = Agent(FakeLLM(["ok"]), [], [LoggingNode(Broken()), ReActLoopNode()])
    ctx = await agent.run("hi", [])
    assert ctx.response == "ok"

    agent = Agent(FakeLLM([]), [], [LoggingNode(Broken()), ReActLoopNode()])
    with pytest.raises(AssertionError):  # original error, not OSError
        await agent.run("hi", [])
    assert "failed to write event" in caplog.text


async def test_sqlite_sink(tmp_path):
    path = tmp_path / "e.db"
    agent = Agent(FakeLLM(["ok"]), [], [LoggingNode(EventLog.open(path)), ReActLoopNode()])
    ctx = await agent.run("hi", [])
    assert read_events(path)[0]["trace_id"] == ctx.state["trace_id"]


def test_from_config(tmp_path):
    sink = tmp_path / "x" / "e.jsonl"
    node = LoggingNode.from_config({"sink": str(sink), "redact_inputs": True}, SimpleNamespace())
    assert node.redact_inputs is True and sink.parent.exists()
    shared = object()
    node = LoggingNode.from_config({}, SimpleNamespace(event_log=shared))
    assert node.event_log is shared


def test_from_config_unknown_param():
    with pytest.raises(ValueError, match="bogus"):
        LoggingNode.from_config({"bogus": 1}, SimpleNamespace())
