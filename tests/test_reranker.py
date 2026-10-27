"""Unit tests for `LLMReranker` (nodes/reranker.py), using `FakeLLM` — no network."""

from __future__ import annotations

import pytest

from core.llm import LLMInvalidOutput
from nodes.reranker import LLMReranker
from tests.fakes import FakeLLM


@pytest.mark.asyncio
async def test_empty_passages_skip_llm_call():
    llm = FakeLLM([])

    assert await LLMReranker(llm).rerank("q", []) == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_one_schema_call_per_passage_with_query_and_passage():
    llm = FakeLLM(['{"score": 1}', '{"score": 2}'])

    await LLMReranker(llm).rerank("qui est Loan ?", ["texte a", "texte b"])

    assert len(llm.calls) == 2
    assert all(call["schema"] is not None for call in llm.calls)
    prompts = [call["messages"][-1].content for call in llm.calls]
    assert all("qui est Loan ?" in p for p in prompts)
    assert "texte a" in prompts[0] and "texte b" in prompts[1]


@pytest.mark.asyncio
async def test_scores_are_normalized_and_in_input_order():
    llm = FakeLLM(['{"score": 3}', '{"score": 10}'])

    assert await LLMReranker(llm).rerank("q", ["a", "b"]) == pytest.approx([0.3, 1.0])


@pytest.mark.asyncio
async def test_out_of_range_grades_are_clamped():
    llm = FakeLLM(['{"score": 42}', '{"score": -3}'])

    assert await LLMReranker(llm).rerank("q", ["a", "b"]) == pytest.approx([1.0, 0.0])


@pytest.mark.asyncio
async def test_long_passages_are_truncated_in_prompt():
    llm = FakeLLM(['{"score": 5}'])

    await LLMReranker(llm).rerank("q", ["x" * 5000])

    assert "x" * 1001 not in llm.calls[0]["messages"][-1].content


@pytest.mark.asyncio
async def test_invalid_json_raises():
    llm = FakeLLM(["nope", "still nope"])

    with pytest.raises(LLMInvalidOutput):
        await LLMReranker(llm).rerank("q", ["a"])


@pytest.mark.asyncio
async def test_wrong_payload_shape_raises():
    llm = FakeLLM(['{"grade": 4}'])

    with pytest.raises(ValueError, match="unexpected rerank payload"):
        await LLMReranker(llm).rerank("q", ["a"])


@pytest.mark.asyncio
async def test_one_failed_passage_raises_after_all_gradings_ran():
    llm = FakeLLM(['{"grade": 4}', '{"score": 8}'])

    with pytest.raises(ValueError):
        await LLMReranker(llm).rerank("q", ["a", "b"])
    assert llm.remaining == 0
