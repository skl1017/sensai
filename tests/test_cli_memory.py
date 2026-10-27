"""`/memory` CLI command: audit, correct and purge long-term facts (issue #52)."""

from __future__ import annotations

import io

import pytest

from app import build_agent
from core.types import ToolCall
from storage.memory_db import MemoryDB
from storage.stores import InMemoryStores
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_cli import scripted_read_line
from ui.cli import CliUI, run_cli

pytestmark = pytest.mark.usefixtures("no_mcp")

SAVE = ToolCall(
    id="1",
    name="memory_save",
    arguments={
        "entity": "Alice",
        "facts": {"color": "green"},
        "relations": [{"rel": "works_at", "target": "Acme"}],
    },
)


async def _session(lines: list[str], db: MemoryDB | None, script=None) -> str:
    out = io.StringIO()
    ui = CliUI(out)
    llm = FakeLLM(script if script is not None else [[SAVE], "saved"])
    llm.embedder = FakeEmbedder()
    stores = InMemoryStores(memory_db=db)
    agent = await build_agent(
        "config/agent.yaml", ui, llm=llm, stores=InMemoryStores(memory_db=db or MemoryDB())
    )
    await run_cli(agent, ui, scripted_read_line([*lines, "/quit"]), stores, "sess")
    return out.getvalue()


async def test_memory_list_edit_delete_end_to_end():
    db = MemoryDB()
    text = await _session(
        [
            "my name is Alice",
            "/memory list",
            "/memory edit 1 bright blue",
            "/memory list Alice",
            "/memory delete 1",
            "/memory list",
        ],
        db,
    )
    assert "[1] Alice / color: green" in text
    assert "Alice --works_at--> Acme" in text
    assert "Updated fact 1" in text
    assert "[1] Alice / color: bright blue" in text
    assert "Deleted fact 1" in text
    assert text.rstrip().endswith("Memory is empty.")
    assert db.list_facts() == []


async def test_memory_edit_changes_db():
    db = MemoryDB()
    await _session(["hi", "/memory edit 1 teal"], db)
    assert db.get_fact(1).value == "teal"


async def test_memory_unknown_id():
    text = await _session(["/memory edit 42 x", "/memory delete 42"], MemoryDB(), script=[])
    assert text.count("No fact with id 42") == 2


async def test_memory_bad_args_print_usage():
    cmds = [
        "/memory",
        "/memory edit abc x",
        "/memory edit 1",
        "/memory delete abc",
        "/memory delete",
        "/memory delete 1 2",
        "/memory nope",
        "/memory list a b",
    ]
    text = await _session(cmds, MemoryDB(), script=[])
    assert text.count("Usage: /memory") == len(cmds)
    assert "Traceback" not in text


async def test_memory_empty_list():
    assert "Memory is empty." in await _session(["/memory list"], MemoryDB(), script=[])


async def test_memory_disabled():
    out = io.StringIO()
    ui = CliUI(out)
    # The agent needs a db to build from the real config; the CLI's stores have none.
    agent_stores = InMemoryStores(memory_db=MemoryDB())
    llm = FakeLLM([])
    llm.embedder = FakeEmbedder()
    agent = await build_agent("config/agent.yaml", ui, llm=llm, stores=agent_stores)
    cmds = ["/memory list", "/memory edit 1 x", "/memory delete 1", "/quit"]
    await run_cli(agent, ui, scripted_read_line(cmds), InMemoryStores(), "sess")
    assert out.getvalue().count("Memory is not enabled.") == 3
