"""Unit tests for the `llm/ollama.py` adapter.

Everything runs against `httpx.MockTransport`: no real network call is ever
made. Streaming responses are built as NDJSON (one JSON object per line, as
`/api/chat` and `/api/embed` actually reply) via the `ndjson` helper below.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.llm import LLMError, LLMModelNotFound, LLMToolsUnsupported, LLMUnavailable
from core.types import Message, ToolCall
from llm.ollama import Ollama, OllamaEmbedder

HOST = "http://ollama.test"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def ndjson(*lines: str | dict) -> bytes:
    """Build a NDJSON body. A `str` line is inserted verbatim (e.g. "" for a
    blank line to skip), a `dict` line is JSON-encoded."""
    parts = [line if isinstance(line, str) else json.dumps(line) for line in lines]
    return ("\n".join(parts) + "\n").encode()


def done_line(prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    return {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def make_llm(handler, **kwargs) -> Ollama:
    client = httpx.AsyncClient(base_url=HOST, transport=httpx.MockTransport(handler))
    return Ollama(client=client, host=HOST, **kwargs)


def make_embedder(handler, **kwargs) -> OllamaEmbedder:
    client = httpx.AsyncClient(base_url=HOST, transport=httpx.MockTransport(handler))
    return OllamaEmbedder(client=client, host=HOST, **kwargs)


async def collect(chat_iter):
    return [chunk async for chunk in chat_iter]


def never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no network call was expected, got {request.method} {request.url}")


# --------------------------------------------------------------------------
# Construction / attributes
# --------------------------------------------------------------------------


async def test_default_attributes():
    llm = make_llm(lambda r: httpx.Response(200, content=ndjson(done_line())))
    assert llm.model == "llama3.1"
    assert llm.embedder is None
    assert llm.host == HOST
    assert llm.max_context_tokens == 8192
    assert llm._options == {"num_ctx": 8192}


async def test_options_include_num_ctx_and_user_options():
    llm = make_llm(never_called, max_context_tokens=4096, options={"temperature": 0.3})
    assert llm._options == {"num_ctx": 4096, "temperature": 0.3}


async def test_user_num_ctx_option_overrides_max_context_tokens():
    llm = make_llm(never_called, max_context_tokens=4096, options={"num_ctx": 100})
    assert llm._options == {"num_ctx": 100}


async def test_async_context_manager_closes_client():
    client = httpx.AsyncClient(base_url=HOST, transport=httpx.MockTransport(never_called))
    async with Ollama(client=client, host=HOST) as llm:
        assert llm.model == "llama3.1"
    assert client.is_closed


async def test_aclose_closes_client():
    client = httpx.AsyncClient(base_url=HOST, transport=httpx.MockTransport(never_called))
    llm = Ollama(client=client, host=HOST)
    await llm.aclose()
    assert client.is_closed


# --------------------------------------------------------------------------
# `chat` payload shape
# --------------------------------------------------------------------------


async def test_chat_payload_has_model_stream_num_ctx_and_no_tools_or_format():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(200, content=ndjson(done_line()))

    llm = make_llm(handler)
    await collect(llm.chat([Message(role="user", content="hi")]))

    assert captured["path"] == "/api/chat"
    payload = captured["payload"]
    assert payload["model"] == "llama3.1"
    assert payload["stream"] is True
    assert payload["options"] == {"num_ctx": 8192}
    assert "tools" not in payload
    assert "format" not in payload


async def test_chat_payload_options_reflect_merged_and_overridden_options():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(done_line()))

    llm = make_llm(handler, max_context_tokens=4096, options={"num_ctx": 100, "temperature": 0.2})
    await collect(llm.chat([Message(role="user", content="hi")]))

    assert captured["payload"]["options"] == {"num_ctx": 100, "temperature": 0.2}


async def test_chat_payload_includes_tools_when_given():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(done_line()))

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    llm = make_llm(handler)
    await collect(llm.chat([Message(role="user", content="hi")], tools=tools))

    assert captured["payload"]["tools"] == tools


async def test_chat_payload_includes_format_when_schema_given():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(done_line()))

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    llm = make_llm(handler)
    await collect(llm.chat([Message(role="user", content="hi")], schema=schema))

    assert captured["payload"]["format"] == schema


async def test_chat_payload_messages_use_to_wire():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(done_line()))

    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]
    llm = make_llm(handler)
    await collect(llm.chat(messages))

    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


# --------------------------------------------------------------------------
# `_to_wire`
# --------------------------------------------------------------------------


def test_to_wire_system_message():
    wire = Ollama._to_wire(Message(role="system", content="be nice"))
    assert wire == {"role": "system", "content": "be nice"}


def test_to_wire_user_message():
    wire = Ollama._to_wire(Message(role="user", content="hi"))
    assert wire == {"role": "user", "content": "hi"}


def test_to_wire_assistant_with_tool_calls():
    m = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="x1", name="get_weather", arguments={"city": "Paris"})],
    )
    wire = Ollama._to_wire(m)
    assert wire == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    }
    assert "tool_call_id" not in wire
    assert "id" not in wire["tool_calls"][0]


def test_to_wire_tool_message():
    m = Message(role="tool", content="42", name="get_weather", tool_call_id="x1")
    wire = Ollama._to_wire(m)
    assert wire == {"role": "tool", "content": "42", "tool_name": "get_weather"}
    assert "tool_call_id" not in wire


# --------------------------------------------------------------------------
# `chat` streaming: text, done, token counts, empty lines
# --------------------------------------------------------------------------


async def test_chat_streams_text_chunks_and_final_chunk_has_token_counts():
    body = ndjson(
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        done_line(prompt_tokens=12, completion_tokens=5),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.done for c in chunks] == [False, False, True]
    assert chunks[-1].prompt_tokens == 12
    assert chunks[-1].completion_tokens == 5


async def test_chat_skips_empty_lines_in_stream():
    body = ndjson(
        "",
        {"message": {"role": "assistant", "content": "Hi"}, "done": False},
        "",
        "",
        done_line(prompt_tokens=1, completion_tokens=1),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    assert [c.text for c in chunks] == ["Hi", ""]
    assert chunks[-1].done is True


# --------------------------------------------------------------------------
# `chat` tool call parsing
# --------------------------------------------------------------------------


async def test_chat_parses_tool_call_with_dict_arguments():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                ],
            },
            "done": False,
        },
        done_line(),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    tool_calls = chunks[0].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"city": "Paris"}
    assert isinstance(tool_calls[0].id, str) and tool_calls[0].id


async def test_chat_parses_tool_call_with_string_arguments():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            "done": False,
        },
        done_line(),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    assert chunks[0].tool_calls[0].arguments == {"city": "Paris"}


async def test_chat_tool_call_missing_arguments_defaults_to_empty_dict():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "ping"}}],
            },
            "done": False,
        },
        done_line(),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    assert chunks[0].tool_calls[0].arguments == {}


async def test_chat_tool_call_ids_unique_within_same_chunk():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": {}}},
                    {"function": {"name": "b", "arguments": {}}},
                ],
            },
            "done": False,
        },
        done_line(),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    ids = [tc.id for tc in chunks[0].tool_calls]
    assert len(ids) == 2
    assert ids[0] != ids[1]


async def test_chat_tool_call_ids_unique_across_chunks():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "a", "arguments": {}}}],
            },
            "done": False,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "b", "arguments": {}}}],
            },
            "done": False,
        },
        done_line(),
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    chunks = await collect(llm.chat([Message(role="user", content="hi")]))

    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 2
    assert tool_chunks[0].tool_calls[0].id != tool_chunks[1].tool_calls[0].id


# --------------------------------------------------------------------------
# `count_tokens`
# --------------------------------------------------------------------------


def test_count_tokens_exact_formula():
    llm = make_llm(never_called)
    messages = [
        Message(role="user", content="hello world"),
        Message(role="assistant", content="hi"),
    ]
    expected = sum(len(m.content) for m in messages) // 4 + 4 * len(messages)
    assert llm.count_tokens(messages) == expected


def test_count_tokens_deterministic_and_makes_no_network_call():
    llm = make_llm(never_called)
    messages = [Message(role="user", content="hello world")]
    assert llm.count_tokens(messages) == llm.count_tokens(messages)


# --------------------------------------------------------------------------
# `embed`
# --------------------------------------------------------------------------


async def test_embed_empty_input_returns_empty_list_without_network():
    embedder = make_embedder(never_called)
    assert await embedder.embed([]) == []


async def test_embed_posts_expected_body_and_returns_vectors():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    embedder = make_embedder(handler)
    result = await embedder.embed(["a", "b"])

    assert captured["path"] == "/api/embed"
    assert captured["payload"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert result == [[0.1, 0.2], [0.3, 0.4]]


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


async def test_chat_connect_error_raises_llm_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    llm = make_llm(handler)
    with pytest.raises(LLMUnavailable) as exc_info:
        await collect(llm.chat([Message(role="user", content="hi")]))

    msg = str(exc_info.value).lower()
    assert "ollama serve" in msg or HOST.lower() in msg


async def test_embed_connect_error_raises_llm_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    embedder = make_embedder(handler)
    with pytest.raises(LLMUnavailable) as exc_info:
        await embedder.embed(["x"])

    msg = str(exc_info.value).lower()
    assert "ollama serve" in msg or HOST.lower() in msg


async def test_chat_404_raises_llm_model_not_found_with_pull_hint():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'llama3.1' not found"})

    llm = make_llm(handler)
    with pytest.raises(LLMModelNotFound) as exc_info:
        await collect(llm.chat([Message(role="user", content="hi")]))

    assert "ollama pull llama3.1" in str(exc_info.value)


async def test_embed_404_raises_llm_model_not_found_with_pull_hint():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'nomic-embed-text' not found"})

    embedder = make_embedder(handler)
    with pytest.raises(LLMModelNotFound) as exc_info:
        await embedder.embed(["x"])

    assert "ollama pull nomic-embed-text" in str(exc_info.value)


async def test_chat_400_with_tools_raises_llm_tools_unsupported():
    def handler(request):
        return httpx.Response(
            400, json={"error": "registry.ollama.ai model does not support tools"}
        )

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    llm = make_llm(handler)
    with pytest.raises(LLMToolsUnsupported):
        await collect(llm.chat([Message(role="user", content="hi")], tools=tools))


async def test_chat_400_without_tools_raises_plain_llm_error_not_tools_unsupported():
    def handler(request):
        return httpx.Response(400, json={"error": "bad request"})

    llm = make_llm(handler)
    with pytest.raises(LLMError) as exc_info:
        await collect(llm.chat([Message(role="user", content="hi")]))

    assert not isinstance(exc_info.value, LLMToolsUnsupported)


async def test_chat_500_raises_llm_error_with_status_code():
    def handler(request):
        return httpx.Response(500, json={"error": "internal server error"})

    llm = make_llm(handler)
    with pytest.raises(LLMError) as exc_info:
        await collect(llm.chat([Message(role="user", content="hi")]))

    assert "500" in str(exc_info.value)


async def test_chat_mid_stream_error_line_raises_llm_error():
    body = ndjson(
        {"message": {"role": "assistant", "content": "Hi"}, "done": False},
        {"error": "model runner crashed"},
    )
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    with pytest.raises(LLMError):
        await collect(llm.chat([Message(role="user", content="hi")]))


async def test_chat_malformed_line_raises_llm_error():
    body = ndjson({"message": {"role": "assistant", "content": "Hi"}, "done": False}, "{oops")
    llm = make_llm(lambda r: httpx.Response(200, content=body))
    with pytest.raises(LLMError, match="malformed"):
        await collect(llm.chat([Message(role="user", content="hi")]))


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


class HangingStream(httpx.AsyncByteStream):
    """Yields one line, then hangs forever until cancelled; records aclose()."""

    def __init__(self, first_line: bytes) -> None:
        self.first_line = first_line
        self.closed = False
        self.event = asyncio.Event()

    async def __aiter__(self):
        yield self.first_line
        await self.event.wait()

    async def aclose(self) -> None:
        self.closed = True


async def test_cancelling_chat_consumption_closes_the_stream():
    line = json.dumps({"message": {"role": "assistant", "content": "Hi"}, "done": False}).encode()
    stream = HangingStream(line + b"\n")

    def handler(request):
        return httpx.Response(200, stream=stream)

    llm = make_llm(handler)
    got_first_chunk = asyncio.Event()

    async def consume():
        async for _ in llm.chat([Message(role="user", content="hi")]):
            got_first_chunk.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got_first_chunk.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
