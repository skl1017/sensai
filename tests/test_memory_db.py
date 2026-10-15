"""Unit tests for `MemoryDB` (issue #49)."""

from __future__ import annotations

import pytest

from app import _stores_from_config
from config_loader import AgentConfig
from storage.memory_db import MemoryDB


@pytest.fixture
def db():
    d = MemoryDB(":memory:")
    yield d
    d.close()


def _save(db, entity, facts, **kw):
    """Save `facts` with a distinct unit-ish vector per key (3-D, index by position)."""
    vecs = {k: [1.0, float(n), 0.0] for n, k in enumerate(facts)}
    return db.save(entity, facts, embeddings=vecs, **kw)


def test_save_twice_updates_single_row(db):
    [first] = _save(db, "alice", {"city": "Paris"})
    [second] = _save(db, "alice", {"city": "Lyon"}, session="s2")
    assert first == second
    facts = db.list_facts("alice")
    assert len(facts) == 1
    assert facts[0].value == "Lyon"
    assert facts[0].source_session == "s2"


def test_search_ranks_by_cosine(db):
    db.save(
        "e",
        {"a": "apple pie", "b": "unrelated text", "c": "other"},
        embeddings={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.9, 0.1]},
    )
    assert [f.key for f in db.search([1.0, 0.0])] == ["a", "c"]  # b scores 0: dropped
    assert [f.key for f in db.search([0.0, 1.0])] == ["b", "c"]
    assert len(db.search([1.0, 1.0], k=1)) == 1
    assert db.search([1.0, 0.0], k=0) == []
    assert db.search([0.0, 0.0]) == []


def test_update_fact_reembeds(db):
    [fid] = db.save("alice", {"city": "Paris"}, embeddings={"city": [1.0, 0.0]})
    db.update_fact(fid, "Nice", [0.0, 1.0])
    assert db.get_fact(fid).value == "Nice"
    assert db.search([1.0, 0.0]) == []
    assert [f.id for f in db.search([0.0, 1.0])] == [fid]
    with pytest.raises(ValueError, match="Unknown fact id 999"):
        db.update_fact(999, "x", [1.0, 0.0])


def test_forget(db):
    [fid] = _save(db, "alice", {"city": "Paris"})
    assert db.forget(fid) is True
    assert db.forget(fid) is False
    assert db.search([1.0, 0.0, 0.0]) == []
    with pytest.raises(ValueError, match="Unknown fact id"):
        db.get_fact(fid)


def test_empty_names_rejected(db):
    with pytest.raises(ValueError, match="entity name"):
        db.upsert_entity("  ")
    with pytest.raises(ValueError, match="entity name"):
        _save(db, "", {"k": "v"})
    with pytest.raises(ValueError, match="fact key"):
        _save(db, "alice", {"": "v"})


def test_upsert_entity_is_idempotent_and_keeps_type(db):
    eid = db.upsert_entity("alice", "person")
    assert db.upsert_entity("alice") == eid


def test_relations(db):
    db.save("alice", {}, relations=[("works_at", "acme"), ("knows", "bob")])
    db.save("alice", {}, relations=[("works_at", "acme")])  # no duplicate
    assert db.relations_of("alice") == [("works_at", "acme"), ("knows", "bob")]
    assert db.relations_of("acme") == []


def test_cascade_delete_entity_removes_facts_and_relations(db):
    _save(db, "alice", {"k": "v"}, relations=[("knows", "bob")])
    eid = db.upsert_entity("alice")
    with db._lock, db._conn:
        db._conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
    assert db.list_facts() == []
    assert db.search([1.0, 0.0, 0.0]) == []
    assert db.relations_of("alice") == []


def test_file_db_persists_and_uses_wal(tmp_path):
    path = tmp_path / "sub" / "mem.db"
    path.parent.mkdir()
    first = MemoryDB(path)
    _save(first, "alice", {"city": "Paris"})
    assert first._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    first.close()
    second = MemoryDB(path)
    assert second.search([1.0, 0.0, 0.0])[0].entity == "alice"
    second.close()


async def test_async_wrappers(db):
    [fid] = await db.asave(
        "alice", {"city": "Paris"}, relations=[("knows", "bob")], embeddings={"city": [1.0, 0.0]}
    )
    assert (await db.asearch([1.0, 0.0]))[0].id == fid
    assert (await db.aget_fact(fid)).key == "city"
    await db.aupdate_fact(fid, "Rome", [1.0, 0.0])
    assert [f.value for f in await db.alist_facts("alice")] == ["Rome"]
    assert await db.arelations_of("alice") == [("knows", "bob")]
    assert await db.aupsert_entity("alice") > 0
    assert await db.aforget(fid) is True


def test_stores_from_config_builds_memory_db(tmp_path):
    path = tmp_path / "data" / "memory.db"
    cfg = AgentConfig(
        llm={}, persona="p", pipeline=[], tools=[("memory", {"db": str(path)})], approval=[]
    )
    stores = _stores_from_config(cfg)
    assert path.exists()
    stores.memory_db.close()
    assert _stores_from_config(AgentConfig(llm={}, persona="p", pipeline=[])).memory_db is None
