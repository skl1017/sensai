from dataclasses import dataclass

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
        return 1.0 - distance

    async def search(self, embedding: list[float], k: int = 4) -> list[tuple[ChunkData, float]]:
        hits = await self._vectors.search("chunk_vectors", embedding, k)
        if not hits:
            return []
        with Session(self._engine) as session:
            rows = session.exec(
                select(Chunk, IngestedFile.file_name)
                .join(IngestedFile, Chunk.source_id == IngestedFile.id, isouter=True)
                .where(Chunk.id.in_([h.chunk_id for h in hits]))
            ).all()
        by_id = {chunk.id: (chunk, name) for chunk, name in rows}
        return [
            (
                ChunkData(
                    r.id, r.text, r.source_id, r.chunk_index, r.doc_category, r.created_at, name
                ),
                self._similarity(h.distance),
            )
            for h in hits
            if (entry := by_id.get(h.chunk_id)) is not None
            for r, name in [entry]
        ]
