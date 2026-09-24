"""Unit tests for `Agent`: chain construction, execution order, history and
state isolation, and ContextVar publication (`core/runtime.py`).

Stubs are defined locally (not in `tests/fakes.py`) to keep this file
self-contained.
"""

from __future__ import annotations

import asyncio

import pytest

from core import runtime
from core.agent import Agent
from core.llm import ILLM
from core.node import Next, PipelineNode
from core.tool import ITool
from core.types import Context, Message


class StubLLM(ILLM):
    model = "stub"
    max_context_tokens = 1024

    async def chat(self, messages, tools=None, schema=None):
        if False:
            yield  # pragma: no cover - makes this an async generator
        return

    def count_tokens(self, messages):
        return 0


class StubTool(ITool):
    def __init__(self, name: str = "stub_tool"):
        self.name = name
        self.description = "a stub tool"
        self.parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, **kwargs) -> str:
        return "ok"


class TransparentNode(PipelineNode):
    """Just calls next and returns the result unchanged."""

    async def handle(self, ctx: Context, next: Next) -> Context:
        return await next(ctx)


class WrapperNode(PipelineNode):
    """Does something before and after next."""

    def __init__(self, log: list[str]):
        self.log = log

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.log.append("before")
        ctx = await next(ctx)
        self.log.append("after")
        ctx.response = "wrapped"
        return ctx


class ShortCircuitNode(PipelineNode):
    """Sets ctx.response and never calls next."""

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.response = "short-circuited"
        return ctx


class RecordingNode(PipelineNode):
    def __init__(self, label: str, log: list[str]):
        self.label = label
        self.log = log

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.log.append(f"{self.label}>")
        ctx = await next(ctx)
        self.log.append(f"<{self.label}")
        return ctx


class MessageAppendingNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.messages = [*ctx.messages, Message(role="assistant", content="added")]
        return await next(ctx)


class NeverCalledNode(PipelineNode):
    """Fails the test if handle is ever invoked (used after a short-circuit)."""

    def __init__(self, log: list[str]):
        self.log = log

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.log.append("never")
        return await next(ctx)


class RuntimeReadingNode(PipelineNode):
    def __init__(self, captured: dict):
        self.captured = captured

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.captured["session_id"] = runtime.session_id.get()
        self.captured["depth"] = runtime.depth.get()
        return await next(ctx)


class CancellingNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        raise asyncio.CancelledError()


def make_agent(pipeline: list[PipelineNode], tools: list[ITool] | None = None) -> Agent:
    return Agent(llm=StubLLM(), tools=tools or [], pipeline=pipeline)


async def test_transparent_node_returns_ctx_from_next():
    agent = make_agent([TransparentNode()])
    ctx = await agent.run("hi", [])
    assert ctx.response is None
    assert ctx.user_input == "hi"


async def test_wrapper_node_acts_before_and_after_next():
    log: list[str] = []
    agent = make_agent([WrapperNode(log)])
    ctx = await agent.run("hi", [])
    assert log == ["before", "after"]
    assert ctx.response == "wrapped"


async def test_short_circuit_node_sets_response_and_skips_rest():
    log: list[str] = []
    agent = make_agent([ShortCircuitNode(), NeverCalledNode(log)])
    ctx = await agent.run("hi", [])
    assert ctx.response == "short-circuited"
    assert log == []


async def test_execution_order():
    log: list[str] = []
    agent = make_agent([RecordingNode("A", log), RecordingNode("B", log), RecordingNode("C", log)])
    await agent.run("hi", [])
    assert log == ["A>", "B>", "C>", "<C", "<B", "<A"]


async def test_empty_pipeline_returns_response_none_and_user_input():
    agent = make_agent([])
    ctx = await agent.run("hello there", [])
    assert ctx.response is None
    assert ctx.user_input == "hello there"


async def test_caller_history_not_mutated():
    history = [Message(role="user", content="first")]
    agent = make_agent([MessageAppendingNode()])
    ctx = await agent.run("hi", history)
    assert history == [Message(role="user", content="first")]
    assert ctx.messages is not history
    assert ctx.messages == [
        Message(role="user", content="first"),
        Message(role="assistant", content="added"),
    ]


async def test_state_none_gives_fresh_dict_each_run():
    agent = make_agent([TransparentNode()])
    ctx1 = await agent.run("hi", [])
    ctx2 = await agent.run("hi", [])
    assert ctx1.state == {}
    assert ctx2.state == {}
    assert ctx1.state is not ctx2.state


async def test_provided_state_is_passed_into_ctx():
    agent = make_agent([TransparentNode()])
    state = {"session_id": "s1", "foo": "bar"}
    ctx = await agent.run("hi", [], state=state)
    assert ctx.state == {"session_id": "s1", "foo": "bar"}


async def test_tools_indexed_by_name_and_llm_injected():
    llm = StubLLM()
    tool = StubTool(name="calculator")
    agent = Agent(llm=llm, tools=[tool], pipeline=[TransparentNode()])
    ctx = await agent.run("hi", [])
    assert ctx.tools == {"calculator": tool}
    assert ctx.llm is llm


async def test_chain_built_once_pipeline_mutation_after_init_has_no_effect():
    log: list[str] = []
    node_a = RecordingNode("A", log)
    agent = make_agent([node_a])
    agent.pipeline.append(RecordingNode("B", log))
    await agent.run("hi", [])
    assert log == ["A>", "<A"]


async def test_runtime_contextvars_see_state_values_during_run():
    captured: dict = {}
    agent = make_agent([RuntimeReadingNode(captured)])
    await agent.run("hi", [], state={"session_id": "sess-1", "depth": 3})
    assert captured == {"session_id": "sess-1", "depth": 3}
    assert runtime.session_id.get() is None
    assert runtime.depth.get() == 0


async def test_runtime_contextvars_default_when_state_lacks_keys():
    captured: dict = {}
    agent = make_agent([RuntimeReadingNode(captured)])
    await agent.run("hi", [], state={})
    assert captured == {"session_id": None, "depth": 0}


async def test_cancelled_error_propagates_and_resets_contextvars():
    agent = make_agent([CancellingNode()])
    with pytest.raises(asyncio.CancelledError):
        await agent.run("hi", [], state={"session_id": "sess-x", "depth": 2})
    assert runtime.session_id.get() is None
    assert runtime.depth.get() == 0
