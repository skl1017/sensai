from dataclasses import dataclass

from sqlmodel import Session, select

from config.db.models import Chunk
from storage.vector_storage import IVectorStorage


@dataclass
class ChunkData:
    id: str
    text: str
    source_id: int
    chunk_index: int
    doc_category: str | None
    created_at: str


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
