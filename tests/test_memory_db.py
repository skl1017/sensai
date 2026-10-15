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


def test_save_twice_updates_single_row(db):
    [first] = db.save("alice", {"city": "Paris"})
    [second] = db.save("alice", {"city": "Lyon"}, session="s2")
    assert first == second
    facts = db.list_facts("alice")
    assert len(facts) == 1
    assert facts[0].value == "Lyon"
    assert facts[0].source_session == "s2"
    assert db.search("Paris") == []  # FTS index followed the update
    assert [f.id for f in db.search("Lyon")] == [first]


def test_search_by_key_and_value(db):
    db.save("alice", {"favorite_color": "blue", "pet": "a cat named Tom"})
    assert db.search("favorite_color")[0].value == "blue"
    hit = db.search("cat")[0]
    assert (hit.entity, hit.key, hit.value) == ("alice", "pet", "a cat named Tom")


@pytest.mark.parametrize(
    "query", ['"', "a AND", "foo*", "(", "NEAR(", "x:y", "'; DROP TABLE facts;--", ""]
)
def test_special_chars_in_query_do_not_raise(db, query):
    db.save("alice", {"k": "v"})
    assert isinstance(db.search(query), list)


def test_search_respects_k(db):
    db.save("e", {f"k{i}": "common word" for i in range(5)})
    assert len(db.search("common", k=3)) == 3
    assert db.search("common", k=0) == []


def test_update_fact(db):
    [fid] = db.save("alice", {"city": "Paris"})
    db.update_fact(fid, "Nice")
    assert db.get_fact(fid).value == "Nice"
    assert [f.id for f in db.search("Nice")] == [fid]
    with pytest.raises(ValueError, match="Unknown fact id 999"):
        db.update_fact(999, "x")


def test_forget(db):
    [fid] = db.save("alice", {"city": "Paris"})
    assert db.forget(fid) is True
    assert db.forget(fid) is False
    assert db.search("Paris") == []
    with pytest.raises(ValueError, match="Unknown fact id"):
        db.get_fact(fid)


def test_empty_names_rejected(db):
    with pytest.raises(ValueError, match="entity name"):
        db.upsert_entity("  ")
    with pytest.raises(ValueError, match="entity name"):
        db.save("", {"k": "v"})
    with pytest.raises(ValueError, match="fact key"):
        db.save("alice", {"": "v"})


def test_upsert_entity_is_idempotent_and_keeps_type(db):
    eid = db.upsert_entity("alice", "person")
    assert db.upsert_entity("alice") == eid


def test_relations(db):
    db.save("alice", {}, relations=[("works_at", "acme"), ("knows", "bob")])
    db.save("alice", {}, relations=[("works_at", "acme")])  # no duplicate
    assert db.relations_of("alice") == [("works_at", "acme"), ("knows", "bob")]
    assert db.relations_of("acme") == []


def test_cascade_delete_entity_removes_facts_and_relations(db):
    db.save("alice", {"k": "v"}, relations=[("knows", "bob")])
    eid = db.upsert_entity("alice")
    with db._lock, db._conn:
        db._conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
    assert db.list_facts() == []
    assert db.search("v") == []
    assert db.relations_of("alice") == []


def test_vector_search_merges_with_fts(db):
    db.save(
        "e",
        {"a": "apple pie", "b": "unrelated text", "c": "other"},
        embeddings={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.9, 0.1]},
    )
    assert db.search("zzz") == []
    by_vec = db.search("zzz", vector=[1.0, 0.0])
    assert [f.key for f in by_vec] == ["a", "c"]
    merged = db.search("apple", vector=[0.0, 1.0])
    assert {f.key for f in merged} == {"a", "b", "c"}
    assert len(db.search("apple", k=1, vector=[0.0, 1.0])) == 1


def test_file_db_persists_and_uses_wal(tmp_path):
    path = tmp_path / "sub" / "mem.db"
    path.parent.mkdir()
    first = MemoryDB(path)
    first.save("alice", {"city": "Paris"})
    assert first._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    first.close()
    second = MemoryDB(path)
    assert second.search("Paris")[0].entity == "alice"
    second.close()


async def test_async_wrappers(db):
    [fid] = await db.asave("alice", {"city": "Paris"}, relations=[("knows", "bob")])
    assert (await db.asearch("Paris"))[0].id == fid
    assert (await db.aget_fact(fid)).key == "city"
    await db.aupdate_fact(fid, "Rome")
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
