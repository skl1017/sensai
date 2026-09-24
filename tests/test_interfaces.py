"""Unit tests for the core interfaces: ILLM, IEmbedder, ITool, PipelineNode."""

from collections.abc import AsyncIterator

import pytest

from core.llm import ILLM, IEmbedder
from core.node import Next, PipelineNode
from core.tool import ITool
from core.types import Context, LLMChunk, Message


def test_illm_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ILLM()


def test_iembedder_cannot_be_instantiated():
    with pytest.raises(TypeError):
        IEmbedder()


def test_itool_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ITool()


def test_pipeline_node_cannot_be_instantiated():
    with pytest.raises(TypeError):
        PipelineNode()


def test_illm_subclass_missing_count_tokens_raises():
    class IncompleteLLM(ILLM):
        model = "fake"
        max_context_tokens = 1024

        async def chat(self, messages, tools=None, schema=None):
            yield LLMChunk(text="hi", done=True)

    with pytest.raises(TypeError):
        IncompleteLLM()


def test_itool_subclass_missing_run_raises():
    class IncompleteTool(ITool):
        name = "incomplete"
        description = "missing run"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    with pytest.raises(TypeError):
        IncompleteTool()


def test_pipeline_node_subclass_missing_handle_raises():
    class IncompleteNode(PipelineNode):
        pass

    with pytest.raises(TypeError):
        IncompleteNode()


class FakeLLM(ILLM):
    """Minimal concrete ILLM: chat is an async def generator."""

    model = "fake"
    max_context_tokens = 4096

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        schema: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text="hel")
        yield LLMChunk(text="lo", done=True, prompt_tokens=3, completion_tokens=2)

    def count_tokens(self, messages: list[Message]) -> int:
        return len(messages)


async def test_fake_llm_chat_can_be_iterated_and_last_chunk_is_done():
    llm = FakeLLM()
    chunks = [chunk async for chunk in llm.chat([Message(role="user", content="hi")])]
    assert len(chunks) == 2
    assert chunks[0].done is False
    assert chunks[-1].done is True
    assert chunks[-1].prompt_tokens == 3
    assert chunks[-1].completion_tokens == 2


class FakeTool(ITool):
    name = "fake_tool"
    description = "A fake tool used only for tests."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, **kwargs) -> str:
        return "ok"


async def test_fake_tool_run_returns_str_and_defaults_enforces_own_timeout():
    tool = FakeTool()
    result = await tool.run()
    assert isinstance(result, str)
    assert result == "ok"
    assert tool.enforces_own_timeout is False


class FakeNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.state["visited"] = True
        return await next(ctx)


async def test_fake_node_calls_next():
    async def terminal(ctx: Context) -> Context:
        ctx.response = "final"
        return ctx

    ctx = Context(user_input="hi", messages=[], llm=object(), tools={})
    result = await FakeNode().handle(ctx, terminal)
    assert result is ctx
    assert result.response == "final"
    assert result.state["visited"] is True
