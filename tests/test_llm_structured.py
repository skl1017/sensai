"""Unit tests for `chat_json` (core/llm.py), using `FakeLLM` — no network."""

from __future__ import annotations

import pytest

from core.llm import (
    LLMError,
    LLMInvalidOutput,
    LLMModelNotFound,
    LLMToolsUnsupported,
    LLMUnavailable,
    chat_json,
)
from core.types import Message
from tests.fakes import FakeLLM

MESSAGES = [Message(role="user", content="give me json")]
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


async def test_valid_json_first_try():
    llm = FakeLLM(script=['{"answer": "ok"}'])

    result = await chat_json(llm, MESSAGES, SCHEMA)

    assert result == {"answer": "ok"}
    assert len(llm.calls) == 1


async def test_invalid_then_valid():
    llm = FakeLLM(script=["not json", '{"answer": "ok"}'])

    result = await chat_json(llm, MESSAGES, SCHEMA)

    assert result == {"answer": "ok"}
    assert len(llm.calls) == 2


async def test_invalid_twice_raises():
    llm = FakeLLM(script=["not json", "still not json"])

    with pytest.raises(LLMInvalidOutput) as exc_info:
        await chat_json(llm, MESSAGES, SCHEMA)

    assert isinstance(exc_info.value, LLMError)
    assert len(llm.calls) == 2
    assert "still not json" in str(exc_info.value)


async def test_retries_zero_gives_up_after_one_call():
    llm = FakeLLM(script=["not json"])

    with pytest.raises(LLMInvalidOutput):
        await chat_json(llm, MESSAGES, SCHEMA, retries=0)

    assert len(llm.calls) == 1


async def test_schema_forwarded_tools_none():
    llm = FakeLLM(script=['{"answer": "ok"}'])

    await chat_json(llm, MESSAGES, SCHEMA)

    assert llm.calls[0]["schema"] == SCHEMA
    assert llm.calls[0]["tools"] is None


async def test_long_invalid_output_excerpt_truncated():
    long_text = "x" * 500
    llm = FakeLLM(script=[long_text, long_text])

    with pytest.raises(LLMInvalidOutput) as exc_info:
        await chat_json(llm, MESSAGES, SCHEMA)

    message = str(exc_info.value)
    assert "x" * 200 in message
    assert "x" * 500 not in message
    assert "…" in message


def test_error_hierarchy():
    assert issubclass(LLMUnavailable, LLMError)
    assert issubclass(LLMModelNotFound, LLMError)
    assert issubclass(LLMToolsUnsupported, LLMError)
    assert issubclass(LLMInvalidOutput, LLMError)
