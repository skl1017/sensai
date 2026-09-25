"""Unit tests for `rewrite_query` (A7, issue #32 part 2)."""

from __future__ import annotations

import asyncio

import pytest

from core.types import Message
from nodes.query_rewriter import rewrite_query
from tests.fakes import FakeLLM

_HISTORY = [
    Message("user", "Quelles sont les trois plus grandes villes de France ?"),
    Message(
        "assistant",
        "Les trois plus grandes villes de France sont Paris, Marseille et Lyon.",
    ),
]


async def test_rewrites_using_history() -> None:
    llm = FakeLLM(["Quelle est la deuxième plus grande ville de France ?"])

    result = await rewrite_query(llm, _HISTORY, "et le deuxième ?")

    assert result == "Quelle est la deuxième plus grande ville de France ?"
    assert len(llm.calls) == 1
    sent = llm.calls[0]["messages"]
    assert sent[0].role == "system"
    user_prompt = sent[1].content
    assert "villes de France" in user_prompt
    assert "Marseille et Lyon" in user_prompt
    assert "et le deuxième ?" in user_prompt


async def test_empty_history_skips_llm_call() -> None:
    llm = FakeLLM(["should not be used"])

    result = await rewrite_query(llm, [], "et le deuxième ?")

    assert result == "et le deuxième ?"
    assert llm.calls == []


async def test_history_without_user_or_assistant_skips_llm_call() -> None:
    llm = FakeLLM(["should not be used"])
    history = [Message("system", "you are a helpful assistant"), Message("tool", "42")]

    result = await rewrite_query(llm, history, "et le deuxième ?")

    assert result == "et le deuxième ?"
    assert llm.calls == []


async def test_llm_error_falls_back_to_original_input(caplog: pytest.LogCaptureFixture) -> None:
    class BoomLLM(FakeLLM):
        async def chat(self, messages, tools=None, schema=None):
            raise RuntimeError("backend unavailable")
            yield  # pragma: no cover - never reached, keeps this an async generator

    llm = BoomLLM([])

    with caplog.at_level("WARNING"):
        result = await rewrite_query(llm, _HISTORY, "et le deuxième ?")

    assert result == "et le deuxième ?"
    assert "backend unavailable" in caplog.text


async def test_empty_output_falls_back_to_original_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(["   "])

    with caplog.at_level("WARNING"):
        result = await rewrite_query(llm, _HISTORY, "et le deuxième ?")

    assert result == "et le deuxième ?"
    assert "empty" in caplog.text.lower()


async def test_quotes_and_label_are_stripped() -> None:
    llm = FakeLLM(['Query: "Quelle est la deuxième plus grande ville de France ?"'])

    result = await rewrite_query(llm, _HISTORY, "et le deuxième ?")

    assert result == "Quelle est la deuxième plus grande ville de France ?"


async def test_only_last_max_turns_are_included() -> None:
    llm = FakeLLM(["rewritten"])
    history = [
        Message("user", "first user message"),
        Message("assistant", "first assistant reply"),
        Message("user", "second user message"),
        Message("assistant", "second assistant reply"),
    ]

    await rewrite_query(llm, history, "et le deuxième ?", max_turns=1)

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
        await rewrite_query(llm, _HISTORY, "et le deuxième ?")
