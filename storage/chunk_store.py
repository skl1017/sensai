from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from sqlmodel import Session, select

from config.db.models import Chunk, IngestedFile
from storage.vector_storage import IVectorStorage


@dataclass
class ChunkData:
    id: str
    text: str
    source_id: int
    chunk_index: int
    doc_category: str | None
    created_at: str
    source_name: str | None = None
    document_date: str | None = None  # ISO date of the source file's last modification


class ChunkStore:
    def __init__(self, engine, vector_storage: IVectorStorage):
        self._engine = engine
        self._vectors = vector_storage

    async def add_chunks(self, chunks: list[ChunkData], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks mais {len(embeddings)} embeddings")
        with Session(self._engine) as session:
            session.add_all(
                Chunk(
                    id=c.id,
                    text=c.text,
                    source_id=c.source_id,
                    chunk_index=c.chunk_index,
                    doc_category=c.doc_category,
                    created_at=c.created_at,
                )
                for c in chunks
            )
            session.commit()

        for c, emb in zip(chunks, embeddings, strict=True):
            await self._vectors.add(c.id, emb)

    async def delete_by_source(self, source_id: str) -> None:
        with Session(self._engine) as session:
            chunks = session.exec(select(Chunk).where(Chunk.source_id == source_id)).all()
            for chunk in chunks:
                await self._vectors.delete(chunk.id)
                session.delete(chunk)
            session.commit()

    @staticmethod
    def _similarity(distance: float) -> float:
        return 1.0 - distance**2 / 2

    def file_names(self) -> list[str]:
        """Names of the source files that have chunks, sorted."""
        query = (
            select(IngestedFile.file_name)
            .join(Chunk, Chunk.source_id == IngestedFile.id)
            .distinct()
            .order_by(IngestedFile.file_name)
        )
        with Session(self._engine) as session:
            return list(session.exec(query).all())

    def _allowed_ids(
        self, file_name: str | None, date_from: date | None, date_to: date | None
    ) -> list[str] | None:
        """Chunk ids matching the filters, `None` when no filter applies.

        Both dates are inclusive and compare against the source file's
        `last_modified`; a chunk without a source file never matches a filter.
        """
        if file_name is None and date_from is None and date_to is None:
            return None
        query = select(Chunk.id).join(IngestedFile, Chunk.source_id == IngestedFile.id)
        if file_name is not None:
            query = query.where(IngestedFile.file_name == file_name)
        if date_from is not None:
            query = query.where(IngestedFile.last_modified >= datetime.combine(date_from, time.min))
        if date_to is not None:
            next_day = date_to + timedelta(days=1)
            query = query.where(IngestedFile.last_modified < datetime.combine(next_day, time.min))
        with Session(self._engine) as session:
            return list(session.exec(query).all())

    async def search(
        self,
        embedding: list[float],
        k: int = 4,
        file_name: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[tuple[ChunkData, float]]:
        """The `k` chunks closest to `embedding`, among those matching the optional filters."""
        allowed = self._allowed_ids(file_name, date_from, date_to)
        hits = await self._vectors.search("chunk_vectors", embedding, k, filters=allowed)
        if not hits:
            return []
        with Session(self._engine) as session:
            rows = session.exec(
                select(Chunk, IngestedFile.file_name, IngestedFile.last_modified)
                .join(IngestedFile, Chunk.source_id == IngestedFile.id, isouter=True)
                .where(Chunk.id.in_([h.chunk_id for h in hits]))
            ).all()
        by_id = {chunk.id: (chunk, name, modified) for chunk, name, modified in rows}
        return [
            (
                ChunkData(
                    r.id,
                    r.text,
                    r.source_id,
                    r.chunk_index,
                    r.doc_category,
                    r.created_at,
                    name,
                    modified.date().isoformat() if modified else None,
                ),
                self._similarity(h.distance),
            )
            for h in hits
            if (entry := by_id.get(h.chunk_id)) is not None
            for r, name, modified in [entry]
        ]
