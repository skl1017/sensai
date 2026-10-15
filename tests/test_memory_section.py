"""`MemorySection`: passive recall of long-term facts (#51)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import build_agent, chat_turn
from core.types import Context, ToolCall
from nodes.context_builder import ContextBuilderNode
from nodes.context_section import ContextSection
from nodes.memory_section import MAX_VALUE_CHARS, MemorySection
from storage.memory_db import MemoryDB
from storage.stores import InMemoryStores
from tests.fakes import FakeEmbedder, FakeLLM, FakeUI


def _ctx(text: str = "colour", **state) -> Context:
    return Context(user_input=text, messages=[], llm=FakeLLM([]), tools={}, state=state)


def _db(n: int = 1) -> MemoryDB:
    db = MemoryDB()
    db.save("Alice", {f"colour{i}": f"colour shade {i}" for i in range(n)})
    return db


def _deps(db=None, embedder=None):
    return SimpleNamespace(stores=SimpleNamespace(memory_db=db), embedder=embedder)


async def test_recalls_facts_and_caps_at_k():
    section = MemorySection(_db(8), k=3)

    out = await section.build(_ctx("colour"))

    assert out is not None
    assert out.startswith("Known facts (from memory):\n- Alice / colour")
    assert len(out.splitlines()) == 1 + 3


async def test_default_k_is_five():
    out = await MemorySection(_db(8)).build(_ctx("colour"))

    assert len(out.splitlines()) == 1 + 5


async def test_none_when_nothing_matches():
    assert await MemorySection(_db()).build(_ctx("zebra")) is None
    assert await MemorySection(MemoryDB()).build(_ctx("colour")) is None


async def test_truncates_long_values():
    db = MemoryDB()
    db.save("Alice", {"bio": "colour " + "x" * 1000})

    out = await MemorySection(db).build(_ctx("bio"))

    assert out.endswith("...")
    assert len(out.splitlines()[1]) < MAX_VALUE_CHARS + 50


async def test_uses_rewritten_query_from_state():
    out = await MemorySection(_db()).build(_ctx("zebra", query="colour"))

    assert out is not None


async def test_keyword_only_without_embedder():
    db = MemoryDB()
    db.save("Alice", {"colour": "green"})
    # Different vocabulary: only a vector could match it.
    assert await MemorySection(db).build(_ctx("verdant")) is None


async def test_embedder_is_used_for_the_query():
    db = MemoryDB()
    emb = FakeEmbedder()
    vec = (await emb.embed(["verdant"]))[0]
    db.save("Alice", {"colour": "green"}, embeddings={"colour": vec})
    emb.calls.clear()

    # No keyword overlap: only the vector can match.
    out = await MemorySection(db, emb).build(_ctx("verdant"))

    assert emb.calls == [["verdant"]]
    assert out is not None and "Alice / colour: green" in out


@pytest.mark.parametrize("k", [0, 11, True, "3"])
def test_invalid_k_rejected(k):
    with pytest.raises(ValueError, match="k must be"):
        MemorySection(MemoryDB(), k=k)


def test_from_config_params():
    section = MemorySection.from_config({"k": 7}, _deps(MemoryDB(), FakeEmbedder()))

    assert section.k == 7
    assert section.priority == 30
    assert section.embedder is not None


def test_from_config_unknown_param():
    with pytest.raises(ValueError, match="Unknown MemorySection param"):
        MemorySection.from_config({"bogus": 1}, _deps(MemoryDB()))


@pytest.mark.parametrize("deps", [_deps(None), SimpleNamespace(stores=None, embedder=None)])
def test_from_config_requires_memory_db(deps):
    with pytest.raises(ValueError, match="memory section requires"):
        MemorySection.from_config({}, deps)


def test_context_builder_wires_memory_params():
    node = ContextBuilderNode.from_config(
        {"sections": ["memory"], "memory": {"k": 2}}, _deps(MemoryDB())
    )

    assert isinstance(node.sections[0], MemorySection)
    assert node.sections[0].k == 2


async def test_priority_orders_memory_between_other_sections():
    class Fixed(ContextSection):
        def __init__(self, priority: int, text: str) -> None:
            self.priority = priority
            self.text = text

        async def build(self, ctx):
            return self.text

    node = ContextBuilderNode(
        [Fixed(40, "FEWSHOT"), MemorySection(_db()), Fixed(10, "PERSONA")],
    )

    async def last(ctx):
        return ctx

    ctx = await node.handle(_ctx("colour"), last)

    system = ctx.messages[0].content
    assert system.index("PERSONA") < system.index("Known facts") < system.index("FEWSHOT")


async def test_failing_search_is_skipped_by_context_builder():
    class Broken(MemoryDB):
        async def asearch(self, *a, **kw):
            raise RuntimeError("db down")

    node = ContextBuilderNode([MemorySection(Broken())])

    async def last(ctx):
        return ctx

    ctx = await node.handle(_ctx("colour"), last)

    assert all(m.role != "system" for m in ctx.messages)


async def test_fact_saved_in_one_session_is_recalled_in_the_next():
    cfg = "config/agent.yaml"
    db = MemoryDB()
    stores = InMemoryStores(memory_db=db)
    embedder = FakeEmbedder()

    save = ToolCall(
        id="1",
        name="memory_save",
        arguments={"entity": "user", "facts": {"favorite_language": "OCaml"}},
    )
    llm1 = FakeLLM([[save], "Noted."])
    llm1.embedder = embedder
    agent1 = await build_agent(cfg, FakeUI(), llm=llm1, stores=stores)
    await chat_turn(agent1, stores, "session-1", "My favorite language is OCaml")

    assert len(db.list_facts("user")) == 1

    llm2 = FakeLLM(["OCaml."])
    llm2.embedder = embedder
    agent2 = await build_agent(cfg, FakeUI(), llm=llm2, stores=stores)
    await chat_turn(agent2, stores, "session-2", "What is my favorite language?")

    system = llm2.calls[0]["messages"][0]
    assert system.role == "system"
    assert "user / favorite_language: OCaml" in system.content
