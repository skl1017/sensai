"""Unit tests for the memory tools (`tools/memory_tool.py`)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from app import TOOLS, build_agent
from core import runtime
from storage.memory_db import MemoryDB
from storage.stores import InMemoryStores
from tests.fakes import FakeEmbedder, FakeLLM
from tools.approved_tool import ApprovedTool
from tools.memory_tool import (
    MAX_FACTS_PER_SAVE,
    MAX_VALUE_CHARS,
    NO_MATCH,
    MemoryForgetTool,
    MemorySaveTool,
    MemorySearchTool,
    MemoryUpdateTool,
    from_config,
)


def _tools(embedder=None):
    db = MemoryDB()
    deps = SimpleNamespace(stores=InMemoryStores(memory_db=db), embedder=embedder)
    search, save, update, forget = from_config({}, deps)
    return db, search, save, update, forget


def _first_id(text: str) -> int:
    return int(re.search(r"\[(\d+)\]", text).group(1))


@pytest.mark.parametrize("embedder", [None, FakeEmbedder()], ids=["fts", "embedder"])
async def test_full_cycle_save_search_update_forget(embedder):
    db, search, save, update, forget = _tools(embedder)
    assert await search.run(query="green") == NO_MATCH

    out = await save.run(
        entity="Alice",
        entity_type="person",
        facts={"favorite_color": "green", "city": "Paris"},
        relations=[{"rel": "works_at", "target": "Acme"}],
    )
    assert "Saved 2 fact(s)" in out
    assert await db.arelations_of("Alice") == [("works_at", "Acme")]

    found = await search.run(query="green")
    assert "Alice" in found and "favorite_color" in found and "green" in found
    fact_id = _first_id(found)

    await update.run(id=fact_id, value="blue")
    assert (await db.aget_fact(fact_id)).value == "blue"
    assert "blue" in await search.run(query="blue")

    assert "Forgot" in await forget.run(id=fact_id)
    assert await search.run(query="favorite_color") == NO_MATCH


async def test_embedder_is_used_for_query_and_facts():
    emb = FakeEmbedder()
    _, search, save, update, _ = _tools(emb)
    await save.run(entity="Bob", facts={"pet": "a small dog"})
    assert emb.calls == [["pet: a small dog"]]
    # no shared keyword with the stored fact: only the vector can match
    assert "Bob" in await search.run(query="dog")
    await update.run(id=_first_id(await search.run(query="dog")), value="a cat")
    assert emb.calls[-1] == ["pet: a cat"]


async def test_source_session_comes_from_context_var():
    db, _, save, _, _ = _tools()
    token = runtime.session_id.set("sess-42")
    try:
        await save.run(entity="Alice", facts={"k": "v"})
    finally:
        runtime.session_id.reset(token)
    assert (await db.alist_facts())[0].source_session == "sess-42"


async def test_save_twice_updates_same_fact():
    db, _, save, _, _ = _tools()
    await save.run(entity="Alice", facts={"city": "Paris"})
    await save.run(entity="Alice", facts={"city": "Lyon"})
    facts = await db.alist_facts("Alice")
    assert [(f.key, f.value) for f in facts] == [("city", "Lyon")]


async def test_long_values_are_truncated():
    db, _, save, _, _ = _tools()
    await save.run(entity="Alice", facts={"bio": "x" * 5000})
    assert len((await db.alist_facts())[0].value) <= MAX_VALUE_CHARS + 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entity": "", "facts": {"k": "v"}},
        {"entity": "A", "facts": {}},
        {"entity": "A", "facts": {"": "v"}},
        {"entity": "A", "facts": {"k": ""}},
        {"entity": "A", "facts": {str(i): "v" for i in range(MAX_FACTS_PER_SAVE + 1)}},
        {"entity": "A", "facts": {"k": "v"}, "relations": ["oops"]},
        {"entity": "A", "facts": {"k": "v"}, "relations": [{"rel": "r"}]},
    ],
)
async def test_save_rejects_invalid_input(kwargs):
    _, _, save, _, _ = _tools()
    with pytest.raises(ValueError):
        await save.run(**kwargs)


async def test_errors_raise_value_error():
    _, search, _, update, forget = _tools()
    with pytest.raises(ValueError, match="query"):
        await search.run(query="  ")
    with pytest.raises(ValueError, match="Unknown fact id 99"):
        await update.run(id=99, value="x")
    with pytest.raises(ValueError, match="Unknown fact id 99"):
        await forget.run(id=99)
    with pytest.raises(ValueError, match="integer"):
        await forget.run(id="1")
    with pytest.raises(ValueError, match="value"):
        await update.run(id=1, value="")


def test_schemas_and_names():
    tools = _tools()[1:]
    names = [t.name for t in tools]
    assert names == ["memory_search", "memory_save", "memory_update", "memory_forget"]
    assert len(set(names)) == 4
    for t in tools:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", t.name)
        assert t.parameters["type"] == "object"
        assert t.parameters["additionalProperties"] is False
        assert t.parameters["required"]
        assert set(t.parameters["required"]) <= set(t.parameters["properties"])
        assert t.description
    assert isinstance(tools[0], MemorySearchTool)
    assert isinstance(tools[1], MemorySaveTool)
    assert isinstance(tools[2], MemoryUpdateTool)
    assert isinstance(tools[3], MemoryForgetTool)


def test_from_config_requires_memory_db():
    for stores in (None, InMemoryStores()):
        deps = SimpleNamespace(stores=stores, embedder=None)
        with pytest.raises(ValueError, match="memory_db"):
            from_config({}, deps)


def test_from_config_rejects_unknown_params():
    deps = SimpleNamespace(stores=InMemoryStores(memory_db=MemoryDB()), embedder=None)
    with pytest.raises(ValueError, match="bogus"):
        from_config({"bogus": 1}, deps)


def test_registered_in_tools():
    assert TOOLS["memory"] is from_config


class _NoopUI:
    def on_token(self, text: str) -> None:
        pass

    def on_event(self, kind: str, data: dict) -> None:
        pass

    async def ask(self, question: str, kind: str = "confirm") -> str:
        return "no"


async def test_memory_forget_wrapped_by_approval_config(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "llm:\n  provider: ollama\n  model: fake\npipeline:\n  - react\n"
        "tools:\n  - memory\napproval:\n  - memory_forget\n",
        encoding="utf-8",
    )
    agent = await build_agent(str(path), _NoopUI(), llm=FakeLLM([]))
    assert isinstance(agent.tools["memory_forget"], ApprovedTool)
    for name in ("memory_search", "memory_save", "memory_update"):
        assert not isinstance(agent.tools[name], ApprovedTool)
