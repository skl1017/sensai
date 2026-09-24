"""Unit tests for core/types.py dataclasses."""

from core.types import Context, LLMChunk, Message, ToolCall


def test_tool_call_has_no_defaults():
    tc = ToolCall(id="1", name="calc", arguments={"a": 1})
    assert tc.id == "1"
    assert tc.name == "calc"
    assert tc.arguments == {"a": 1}


def test_message_defaults():
    msg = Message(role="user")
    assert msg.role == "user"
    assert msg.content == ""
    assert msg.tool_calls == []
    assert msg.tool_call_id is None
    assert msg.name is None


def test_llm_chunk_defaults():
    chunk = LLMChunk()
    assert chunk.text == ""
    assert chunk.tool_calls == []
    assert chunk.done is False
    assert chunk.prompt_tokens is None
    assert chunk.completion_tokens is None


def test_context_defaults():
    ctx = Context(user_input="hi", messages=[], llm=object(), tools={})
    assert ctx.response is None
    assert ctx.state == {}


def test_message_instances_do_not_share_tool_calls_list():
    a = Message(role="assistant")
    b = Message(role="assistant")
    a.tool_calls.append(ToolCall(id="1", name="t", arguments={}))
    assert a.tool_calls != b.tool_calls
    assert b.tool_calls == []


def test_llm_chunk_instances_do_not_share_tool_calls_list():
    a = LLMChunk()
    b = LLMChunk()
    a.tool_calls.append(ToolCall(id="1", name="t", arguments={}))
    assert a.tool_calls != b.tool_calls
    assert b.tool_calls == []


def test_successive_contexts_do_not_share_state():
    ctx_a = Context(user_input="a", messages=[], llm=object(), tools={})
    ctx_b = Context(user_input="b", messages=[], llm=object(), tools={})
    ctx_a.state["session_id"] = "abc"
    assert "session_id" not in ctx_b.state
    assert ctx_a.state != ctx_b.state


def test_successive_contexts_response_is_none_by_default():
    ctx_a = Context(user_input="a", messages=[], llm=object(), tools={})
    ctx_a.response = "done"
    ctx_b = Context(user_input="b", messages=[], llm=object(), tools={})
    assert ctx_b.response is None


def test_context_keeps_messages_list_by_reference():
    messages = [Message(role="user", content="hi")]
    ctx = Context(user_input="hi", messages=messages, llm=object(), tools={})
    ctx.messages.append(Message(role="assistant", content="hello"))
    assert len(messages) == 2
    assert messages[-1].content == "hello"
