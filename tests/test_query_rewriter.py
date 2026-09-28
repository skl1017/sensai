"""Unit tests for `rewrite_query` (A7, issue #32 part 2)."""

from __future__ import annotations

import asyncio

import pytest

from core.types import Message
from nodes.query_rewriter import rewrite_query
from tests.fakes import FakeLLM

_HISTORY = [
    Message("user", "What are the three largest cities in France?"),
    Message(
        "assistant",
        "The three largest cities in France are Paris, Marseille and Lyon.",
    ),
]


async def test_rewrites_using_history() -> None:
    llm = FakeLLM(["What is the second largest city in France?"])

    result = await rewrite_query(llm, _HISTORY, "and the second one?")

    assert result == "What is the second largest city in France?"
    assert len(llm.calls) == 1
    sent = llm.calls[0]["messages"]
    assert sent[0].role == "system"
    user_prompt = sent[1].content
    assert "largest cities in France" in user_prompt
    assert "Marseille and Lyon" in user_prompt
    assert "and the second one?" in user_prompt


async def test_empty_history_skips_llm_call() -> None:
    llm = FakeLLM(["should not be used"])

    result = await rewrite_query(llm, [], "and the second one?")

    assert result == "and the second one?"
    assert llm.calls == []


async def test_history_without_user_or_assistant_skips_llm_call() -> None:
    llm = FakeLLM(["should not be used"])
    history = [Message("system", "you are a helpful assistant"), Message("tool", "42")]

    result = await rewrite_query(llm, history, "and the second one?")

    assert result == "and the second one?"
    assert llm.calls == []


async def test_llm_error_falls_back_to_original_input(caplog: pytest.LogCaptureFixture) -> None:
    class BoomLLM(FakeLLM):
        async def chat(self, messages, tools=None, schema=None):
            raise RuntimeError("backend unavailable")
            yield  # pragma: no cover - never reached, keeps this an async generator

    llm = BoomLLM([])

    with caplog.at_level("WARNING"):
        result = await rewrite_query(llm, _HISTORY, "and the second one?")

    assert result == "and the second one?"
    assert "backend unavailable" in caplog.text


async def test_empty_output_falls_back_to_original_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(["   "])

    with caplog.at_level("WARNING"):
        result = await rewrite_query(llm, _HISTORY, "and the second one?")

    assert result == "and the second one?"
    assert "empty" in caplog.text.lower()


async def test_quotes_and_label_are_stripped() -> None:
    llm = FakeLLM(['Query: "What is the second largest city in France?"'])

    result = await rewrite_query(llm, _HISTORY, "and the second one?")

    assert result == "What is the second largest city in France?"


async def test_only_last_max_turns_are_included() -> None:
    llm = FakeLLM(["rewritten"])
    history = [
        Message("user", "first user message"),
        Message("assistant", "first assistant reply"),
        Message("user", "second user message"),
        Message("assistant", "second assistant reply"),
    ]

    await rewrite_query(llm, history, "and the second one?", max_turns=1)

    user_prompt = llm.calls[0]["messages"][1].content
    assert "second user message" in user_prompt
    assert "second assistant reply" in user_prompt
    assert "first user message" not in user_prompt
    assert "first assistant reply" not in user_prompt


async def test_cancelled_error_propagates() -> None:
    class CancellingLLM(FakeLLM):
        async def chat(self, messages, tools=None, schema=None):
            raise asyncio.CancelledError()
            yield  # pragma: no cover - never reached, keeps this an async generator

    llm = CancellingLLM([])

    with pytest.raises(asyncio.CancelledError):
        await rewrite_query(llm, _HISTORY, "and the second one?")
