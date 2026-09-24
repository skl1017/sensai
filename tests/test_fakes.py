"""Unit tests for the test doubles themselves (tests.fakes)."""

from __future__ import annotations

import pytest

from core.llm import ILLM
from core.tool import ITool
from core.types import Message, ToolCall
from tests.fakes import FakeLLM, FakeTool


class TestFakeLLM:
    def test_is_an_illm(self):
        assert isinstance(FakeLLM([]), ILLM)

    async def test_text_step_streams_and_joins_to_original(self):
        llm = FakeLLM(["hello there world"])
        chunks = [c async for c in llm.chat([Message(role="user", content="hi")])]

        text_chunks = [c for c in chunks if not c.done]
        assert len(text_chunks) >= 2  # streaming is exercised
        assert "".join(c.text for c in text_chunks) == "hello there world"

    async def test_last_chunk_done_with_int_token_counts(self):
        llm = FakeLLM(["hello there"])
        chunks = [c async for c in llm.chat([Message(role="user", content="hi")])]

        last = chunks[-1]
        assert last.done is True
        assert isinstance(last.prompt_tokens, int)
        assert isinstance(last.completion_tokens, int)

    async def test_tool_call_step_yields_tool_calls(self):
        calls = [ToolCall(id="1", name="calc", arguments={"a": 1})]
        llm = FakeLLM([calls])
        chunks = [c async for c in llm.chat([Message(role="user", content="hi")])]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_calls == calls
        assert chunks[-1].done is True

    async def test_successive_calls_consume_script_in_order(self):
        llm = FakeLLM(["first", "second"])

        first = [c async for c in llm.chat([Message(role="user", content="a")])]
        second = [c async for c in llm.chat([Message(role="user", content="b")])]

        assert "".join(c.text for c in first if not c.done) == "first"
        assert "".join(c.text for c in second if not c.done) == "second"

    async def test_records_calls_with_messages_tools_schema(self):
        llm = FakeLLM(["hi"])
        messages = [Message(role="user", content="question")]
        tools = [{"name": "calc"}]
        schema = {"type": "object"}

        async for _ in llm.chat(messages, tools=tools, schema=schema):
            pass

        assert len(llm.calls) == 1
        recorded = llm.calls[0]
        assert recorded["messages"] == messages
        assert recorded["tools"] == tools
        assert recorded["schema"] == schema

    async def test_exhausted_script_raises_assertion_error(self):
        llm = FakeLLM([])
        with pytest.raises(AssertionError):
            async for _ in llm.chat([Message(role="user", content="hi")]):
                pass

    async def test_remaining_property(self):
        llm = FakeLLM(["one", "two"])
        assert llm.remaining == 2
        async for _ in llm.chat([Message(role="user", content="hi")]):
            pass
        assert llm.remaining == 1

    def test_count_tokens_heuristic(self):
        llm = FakeLLM([])
        messages = [Message(role="user", content="abcd"), Message(role="user", content="")]
        assert llm.count_tokens(messages) == (4 // 4 + 1) + (0 // 4 + 1)


class TestFakeTool:
    def test_is_an_itool(self):
        assert isinstance(FakeTool(), ITool)

    def test_default_parameters_schema(self):
        tool = FakeTool()
        assert tool.parameters == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def test_returns_default_result(self):
        tool = FakeTool(result="42")
        assert await tool.run() == "42"

    async def test_records_kwargs(self):
        tool = FakeTool()
        await tool.run(a=1, b="two")
        assert tool.calls == [{"a": 1, "b": "two"}]

    async def test_raises_scripted_exception(self):
        tool = FakeTool(result=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await tool.run()

    async def test_sync_callable_result(self):
        tool = FakeTool(result=lambda **kw: f"got:{kw.get('x')}")
        assert await tool.run(x="y") == "got:y"

    async def test_async_callable_result(self):
        async def handler(**kw):
            return f"async:{kw.get('x')}"

        tool = FakeTool(result=handler)
        assert await tool.run(x="z") == "async:z"

    def test_custom_name_and_description(self):
        tool = FakeTool(name="custom", description="does custom things")
        assert tool.name == "custom"
        assert tool.description == "does custom things"
