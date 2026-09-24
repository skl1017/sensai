"""Tests for the request-scoped ContextVars (core/runtime.py) and how
`Agent.run` publishes them for tools and nodes that don't receive the
`Context` directly.
"""

from __future__ import annotations

from core import runtime
from core.agent import Agent
from core.node import Next, PipelineNode
from core.types import Context
from tests.fakes import FakeLLM, FakeTool


class WhoAmINode(PipelineNode):
    """Short-circuits the chain: calls the `whoami` tool and sets the
    response to its result, without ever touching `ctx` for session info.
    """

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.response = await ctx.tools["whoami"].run()
        return ctx


class EchoLLMNode(PipelineNode):
    """Calls the LLM directly and concatenates streamed text into the
    response — shows a node can be exercised with FakeLLM, no Ollama.
    """

    async def handle(self, ctx: Context, next: Next) -> Context:
        text = ""
        async for chunk in ctx.llm.chat(ctx.messages):
            text += chunk.text
        ctx.response = text
        return ctx


def test_runtime_defaults_outside_any_run():
    assert runtime.session_id.get() is None
    assert runtime.depth.get() == 0
    assert runtime.trace_id.get() is None


async def test_tool_reads_session_id_and_depth_via_contextvars():
    whoami = FakeTool(
        name="whoami",
        result=lambda: f"{runtime.session_id.get()}:{runtime.depth.get()}",
    )
    agent = Agent(FakeLLM([]), [whoami], [WhoAmINode()])

    ctx = await agent.run("hi", [], state={"session_id": "s1", "depth": 2})

    assert ctx.response == "s1:2"


async def test_contextvars_reset_after_run():
    whoami = FakeTool(name="whoami", result="ok")
    agent = Agent(FakeLLM([]), [whoami], [WhoAmINode()])

    await agent.run("hi", [], state={"session_id": "s1", "depth": 2})

    assert runtime.session_id.get() is None
    assert runtime.depth.get() == 0


async def test_defaults_when_state_omits_session_and_depth():
    whoami = FakeTool(
        name="whoami",
        result=lambda: f"{runtime.session_id.get()}:{runtime.depth.get()}",
    )
    agent = Agent(FakeLLM([]), [whoami], [WhoAmINode()])

    ctx = await agent.run("hi", [])

    assert ctx.response == "None:0"


async def test_node_uses_fake_llm_without_ollama():
    llm = FakeLLM(["hello there"])
    agent = Agent(llm, [], [EchoLLMNode()])

    ctx = await agent.run("hi", [])

    assert ctx.response == "hello there"
    assert len(llm.calls) == 1
