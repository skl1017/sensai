"""Unit tests for `nodes/citations.py` and their wiring in `ContextBuilderNode` (#124)."""

from __future__ import annotations

from types import SimpleNamespace

from core.agent import Agent
from core.types import Context
from nodes.citations import cited_numbers, resolve_citations
from nodes.context_builder import ContextBuilderNode
from nodes.rag_section import RAGSection
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeEmbedder, FakeLLM


def chunk(index: int, source_name: str | None = "rh.pdf") -> dict:
    return {
        "index": index,
        "chunk_id": f"c{index}",
        "source_id": index,
        "source_name": source_name,
        "chunk_index": 0,
        "score": 0.9,
    }


# --- cited_numbers ---------------------------------------------------------------


def test_cited_numbers_reads_single_and_grouped_markers():
    assert cited_numbers("Cinq semaines [1]. RTT [2, 3] et [3][4].") == {1, 2, 3, 4}


def test_cited_numbers_ignores_text_without_markers():
    assert cited_numbers("Pas de source ici, ni [a] ni [].") == set()


# --- resolve_citations -----------------------------------------------------------


def test_valid_citations_resolve_to_their_chunks_once_in_order():
    retrieved = [chunk(1), chunk(2), chunk(3)]

    result = resolve_citations("A [3]. B [1]. C [3].", retrieved)

    assert result == {"cited": [chunk(1), chunk(3)], "invalid": []}


def test_number_without_passage_is_invalid_never_cited():
    result = resolve_citations("Selon [1] et [7].", [chunk(1), chunk(2)])

    assert [c["index"] for c in result["cited"]] == [1]
    assert result["invalid"] == [7]


def test_answer_without_citation_cites_nothing():
    assert resolve_citations("Je ne sais pas.", [chunk(1)]) == {"cited": [], "invalid": []}


# --- ContextBuilderNode wiring ---------------------------------------------------


class FakeChunkStore:
    def __init__(self, hits):
        self.hits = hits

    async def search(self, embedding, k):
        return self.hits


def rag_agent(answer: str, hits) -> tuple[Agent, FakeLLM]:
    llm = FakeLLM([answer])
    section = RAGSection(FakeEmbedder(), FakeChunkStore(hits))
    return Agent(llm, [], [ContextBuilderNode([section]), ReActLoopNode()]), llm


def hit(chunk_id: str, text: str, source_name: str):
    return (
        SimpleNamespace(
            id=chunk_id, text=text, source_id=1, source_name=source_name, chunk_index=2
        ),
        0.8,
    )


async def test_cited_source_recorded_for_answer_on_ingested_document():
    agent, llm = rag_agent(
        "Vous avez cinq semaines de congés [1].",
        [hit("c1", "Cinq semaines de congés payés.", "rh.pdf")],
    )

    ctx = await agent.run("Combien de congés ?", [])

    system = llm.calls[0]["messages"][0].content
    assert '<document id="1" source="rh.pdf">' in system
    assert [c["source_name"] for c in ctx.state["citations"]["cited"]] == ["rh.pdf"]
    assert ctx.state["citations"]["invalid"] == []


async def test_hallucinated_citation_reported_as_invalid():
    agent, _ = rag_agent("D'après [2].", [hit("c1", "Texte.", "rh.pdf")])

    ctx = await agent.run("Question ?", [])

    assert ctx.state["citations"] == {"cited": [], "invalid": [2]}


async def test_no_citations_key_without_retrieved_chunks():
    agent, _ = rag_agent("Réponse [1].", [])

    ctx = await agent.run("Question ?", [])

    assert "citations" not in ctx.state


async def test_citations_resolved_on_final_response_after_later_nodes():
    """Later nodes (e.g. the output guardrail) may rewrite the response first."""

    class RewriteNode:
        async def handle(self, ctx: Context, next):
            ctx = await next(ctx)
            ctx.response = "Réécrit [1]."
            return ctx

    llm = FakeLLM(["Brut sans citation."])
    section = RAGSection(FakeEmbedder(), FakeChunkStore([hit("c1", "Texte.", "rh.pdf")]))
    agent = Agent(llm, [], [ContextBuilderNode([section]), RewriteNode(), ReActLoopNode()])

    ctx = await agent.run("Question ?", [])

    assert [c["index"] for c in ctx.state["citations"]["cited"]] == [1]
