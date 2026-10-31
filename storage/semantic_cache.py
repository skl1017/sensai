"""Semantic cache: answers a question from a previous, similar one.

Each cached exchange is a `CacheEntry` row (question, response, scope, creation
time) whose question embedding lives in a vector table of its own, sharing the
row id. Lookups are restricted to a scope and, optionally, to a TTL.
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlmodel import Session, select

from config.db.models import CacheEntry
from core.storage import IVectorStorage


@dataclass
class CacheHit:
    """A cached exchange close enough to the query: the question is kept for the caller's guard."""

    question: str
    response: str
    similarity: float


class SemanticCache:
    def __init__(
        self,
        engine,
        vectors: IVectorStorage,
        collection: str = "cache_vectors",
        clock: Callable[[], float] = time.time,
    ):
        self._engine = engine
        self._vectors = vectors
        self._collection = collection
        self._clock = clock

    def _candidate_ids(self, scope: str, ttl_seconds: float | None) -> list[str]:
        statement = select(CacheEntry.id).where(CacheEntry.scope == scope)
        if ttl_seconds is not None:
            statement = statement.where(CacheEntry.created_at >= self._clock() - ttl_seconds)
        with Session(self._engine) as session:
            return list(session.exec(statement).all())

    def _load_entry(self, entry_id: str) -> tuple[str, str] | None:
        with Session(self._engine) as session:
            entry = session.get(CacheEntry, entry_id)
            return (entry.question, entry.response) if entry is not None else None

    def _insert(self, entry: CacheEntry) -> None:
        with Session(self._engine) as session:
            session.add(entry)
            session.commit()

    def _expired_ids(self, ttl_seconds: float) -> list[str]:
        cutoff = self._clock() - ttl_seconds
        with Session(self._engine) as session:
            return list(
                session.exec(select(CacheEntry.id).where(CacheEntry.created_at < cutoff)).all()
            )

    def _delete_rows(self, entry_ids: list[str]) -> None:
        with Session(self._engine) as session:
            for entry in session.exec(select(CacheEntry).where(CacheEntry.id.in_(entry_ids))).all():
                session.delete(entry)
            session.commit()

    async def lookup(
        self, scope: str, embedding: list[float], ttl_seconds: float | None
    ) -> CacheHit | None:
        """Return the closest cached exchange in `scope` as a `CacheHit`, or None."""
        ids = await asyncio.to_thread(self._candidate_ids, scope, ttl_seconds)
        if not ids:
            return None
        hits = await self._vectors.search(self._collection, embedding, 1, filters=ids)
        if not hits:
            return None
        hit = hits[0]
        entry = await asyncio.to_thread(self._load_entry, hit.chunk_id)
        if entry is None:
            return None
        return CacheHit(question=entry[0], response=entry[1], similarity=1.0 - hit.distance)

    async def store(self, scope: str, question: str, embedding: list[float], response: str) -> str:
        """Cache a question/response pair under `scope` and return the new entry id."""
        entry_id = uuid.uuid4().hex
        entry = CacheEntry(
            id=entry_id,
            scope=scope,
            question=question,
            response=response,
            created_at=self._clock(),
        )
        await asyncio.to_thread(self._insert, entry)
        await self._vectors.add(entry_id, embedding)
        return entry_id

    async def purge_expired(self, ttl_seconds: float) -> int:
        """Delete entries older than `ttl_seconds`, with their vectors. Return how many."""
        ids = await asyncio.to_thread(self._expired_ids, ttl_seconds)
        if not ids:
            return 0
        for entry_id in ids:
            await self._vectors.delete(entry_id)
        await asyncio.to_thread(self._delete_rows, ids)
        return len(ids)
