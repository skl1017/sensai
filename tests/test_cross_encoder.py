"""Unit tests for `FastEmbedReranker` (llm/cross_encoder.py), with a fake encoder — no download."""

from __future__ import annotations

import builtins

import pytest

from llm.cross_encoder import FastEmbedReranker


class FakeEncoder:
    """Stands in for `fastembed.TextCrossEncoder`: returns preset logits, records calls."""

    def __init__(self, logits):
        self.logits = logits
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter(self.logits)


@pytest.mark.asyncio
async def test_logits_go_through_sigmoid_in_input_order():
    reranker = FastEmbedReranker(encoder=FakeEncoder([0.0, 4.0, -4.0]))

    scores = await reranker.rerank("q", ["a", "b", "c"])

    assert scores == pytest.approx([0.5, 0.982, 0.018], abs=1e-3)


@pytest.mark.asyncio
async def test_query_and_passages_are_forwarded():
    encoder = FakeEncoder([1.0, 2.0])

    await FastEmbedReranker(encoder=encoder).rerank("qui est Loan ?", ["a", "b"])

    assert encoder.calls == [("qui est Loan ?", ["a", "b"])]


@pytest.mark.asyncio
async def test_empty_passages_skip_encoder():
    encoder = FakeEncoder([])

    assert await FastEmbedReranker(encoder=encoder).rerank("q", []) == []
    assert encoder.calls == []


@pytest.mark.asyncio
async def test_score_count_mismatch_raises():
    reranker = FastEmbedReranker(encoder=FakeEncoder([1.0]))

    with pytest.raises(RuntimeError, match="1 scores for 2 passages"):
        await reranker.rerank("q", ["a", "b"])


def test_construction_does_not_load_model():
    reranker = FastEmbedReranker(model="some/model", cache_dir="x")

    assert reranker.model == "some/model"
    assert reranker._encoder is None


@pytest.mark.asyncio
async def test_missing_fastembed_raises_explicit_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("fastembed"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[rerank\]'"):
        await FastEmbedReranker().rerank("q", ["a"])
