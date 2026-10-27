"""Long-term memory in SQLite: entities, facts, relations (wiki §9.3).

`MemoryDB` is a sync class over a shared SQLAlchemy `Engine` (calls serialized by a lock).
The `a`-prefixed methods run the sync ones through `asyncio.to_thread`.
Every fact carries an embedding; facts are searched by cosine similarity.
"""

from __future__ import annotations

import asyncio
import math
import struct
import threading
from dataclasses import dataclass

from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.pool import StaticPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_session TEXT,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, key)
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    src_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel TEXT NOT NULL,
    dst_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    UNIQUE(src_id, rel, dst_id)
);
"""

_SELECT_FACT = (
    "SELECT f.id, e.name, f.key, f.value, f.source_session, f.updated_at "
    "FROM facts f JOIN entities e ON e.id = f.entity_id"
)


@dataclass(frozen=True)
class Fact:
    id: int
    entity: str
    key: str
    value: str
    source_session: str | None = None
    updated_at: str | None = None


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def _clean(name: str, what: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError(f"{what} must be a non-empty string")
    return name


def in_memory_engine() -> Engine:
    """A private in-memory SQLite engine (one shared connection) with foreign keys on."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


class MemoryDB:
    """Long-term memory over a shared SQLAlchemy `Engine` (`config/db/db.py`).

    The engine must enable `PRAGMA foreign_keys=ON` on connect (cascade deletes).
    Defaults to a private in-memory engine, for tests.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._lock = threading.Lock()
        self._owns_engine = engine is None
        self._engine = engine if engine is not None else in_memory_engine()
        with self._lock, self._engine.begin() as conn:
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    conn.exec_driver_sql(statement)

    def close(self) -> None:
        """Dispose the engine if `MemoryDB` created it; a shared engine is left alone."""
        if self._owns_engine:
            with self._lock:
                self._engine.dispose()

    # -- writes --------------------------------------------------------------

    @staticmethod
    def _upsert_entity(conn: Connection, name: str, type: str | None) -> int:
        row = conn.exec_driver_sql(
            "INSERT INTO entities(name, type) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET type = COALESCE(excluded.type, entities.type) "
            "RETURNING id",
            (name, type),
        ).fetchone()
        return row[0]

    def upsert_entity(self, name: str, type: str | None = None) -> int:
        name = _clean(name, "entity name")
        with self._lock, self._engine.begin() as conn:
            return self._upsert_entity(conn, name, type)

    def save(
        self,
        entity: str,
        facts: dict[str, str],
        relations: list[tuple[str, str]] | None = None,
        session: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[int]:
        """Upsert `entity`, its facts (same key = update) and relations; returns fact ids.

        `embeddings` maps every fact key to its vector.
        """
        entity = _clean(entity, "entity name")
        embeddings = embeddings or {}
        ids: list[int] = []
        with self._lock, self._engine.begin() as conn:
            eid = self._upsert_entity(conn, entity, None)
            for key, value in facts.items():
                key = _clean(key, "fact key")
                row = conn.exec_driver_sql(
                    "INSERT INTO facts(entity_id, key, value, source_session, embedding) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(entity_id, key) DO UPDATE SET value = excluded.value, "
                    "source_session = excluded.source_session, embedding = excluded.embedding, "
                    "updated_at = CURRENT_TIMESTAMP RETURNING id",
                    (eid, key, value, session, _pack(embeddings[key])),
                ).fetchone()
                ids.append(row[0])
            for rel, target in relations or []:
                rel = _clean(rel, "relation name")
                dst = self._upsert_entity(conn, _clean(target, "relation target"), None)
                conn.exec_driver_sql(
                    "INSERT OR IGNORE INTO relations(src_id, rel, dst_id) VALUES (?, ?, ?)",
                    (eid, rel, dst),
                )
        return ids

    def update_fact(self, fact_id: int, value: str, embedding: list[float]) -> None:
        with self._lock, self._engine.begin() as conn:
            cur = conn.exec_driver_sql(
                "UPDATE facts SET value = ?, embedding = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (value, _pack(embedding), fact_id),
            )
        if cur.rowcount == 0:
            raise ValueError(f"Unknown fact id {fact_id}")

    def forget(self, fact_id: int) -> bool:
        with self._lock, self._engine.begin() as conn:
            cur = conn.exec_driver_sql("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

    # -- reads ---------------------------------------------------------------

    def get_fact(self, fact_id: int) -> Fact:
        with self._lock, self._engine.connect() as conn:
            row = conn.exec_driver_sql(_SELECT_FACT + " WHERE f.id = ?", (fact_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown fact id {fact_id}")
        return Fact(*row)

    def list_facts(self, entity: str | None = None) -> list[Fact]:
        with self._lock, self._engine.connect() as conn:
            rows = conn.exec_driver_sql(
                _SELECT_FACT + " WHERE (? IS NULL OR e.name = ?) ORDER BY f.id", (entity, entity)
            ).fetchall()
        return [Fact(*row) for row in rows]

    def relations_of(self, entity: str) -> list[tuple[str, str]]:
        """Outgoing relations of `entity` as `(rel, target_name)`."""
        with self._lock, self._engine.connect() as conn:
            rows = conn.exec_driver_sql(
                "SELECT r.rel, d.name FROM relations r "
                "JOIN entities s ON s.id = r.src_id JOIN entities d ON d.id = r.dst_id "
                "WHERE s.name = ? ORDER BY r.id",
                (entity,),
            ).fetchall()
        return [tuple(row) for row in rows]

    def search(self, vector: list[float], k: int = 10) -> list[Fact]:
        """The `k` facts closest to `vector` by cosine similarity (score > 0)."""
        if k <= 0:
            return []
        with self._lock, self._engine.connect() as conn:
            scored = sorted(
                (
                    (_cosine(vector, _unpack(blob)), fid)
                    for fid, blob in conn.exec_driver_sql("SELECT id, embedding FROM facts")
                ),
                reverse=True,
            )
            top = [fid for score, fid in scored[:k] if score > 0]
            if not top:
                return []
            marks = ",".join("?" * len(top))
            rows = conn.exec_driver_sql(_SELECT_FACT + f" WHERE f.id IN ({marks})", tuple(top))
            by_id = {row[0]: Fact(*row) for row in rows}
        return [by_id[i] for i in top]

    # -- async wrappers ------------------------------------------------------

    async def aupsert_entity(self, name: str, type: str | None = None) -> int:
        return await asyncio.to_thread(self.upsert_entity, name, type)

    async def asave(
        self,
        entity: str,
        facts: dict[str, str],
        relations: list[tuple[str, str]] | None = None,
        session: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[int]:
        return await asyncio.to_thread(self.save, entity, facts, relations, session, embeddings)

    async def asearch(self, vector: list[float], k: int = 10) -> list[Fact]:
        return await asyncio.to_thread(self.search, vector, k)

    async def aupdate_fact(self, fact_id: int, value: str, embedding: list[float]) -> None:
        await asyncio.to_thread(self.update_fact, fact_id, value, embedding)

    async def aforget(self, fact_id: int) -> bool:
        return await asyncio.to_thread(self.forget, fact_id)

    async def aget_fact(self, fact_id: int) -> Fact:
        return await asyncio.to_thread(self.get_fact, fact_id)

    async def alist_facts(self, entity: str | None = None) -> list[Fact]:
        return await asyncio.to_thread(self.list_facts, entity)

    async def arelations_of(self, entity: str) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self.relations_of, entity)
