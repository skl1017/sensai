import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, delete, select

from config.db.db import engine
from config.db.models import IngestedFile
from core.llm import IEmbedder
from llm.ollama import OllamaEmbedder
from storage.chunk_store import ChunkData, ChunkStore
from storage.vector_storage import SqliteVectorStorage


def to_ingested(p) -> IngestedFile:
    path = Path(p.path)
    return IngestedFile(
        file_path=p.path,
        content_hash=p.hash,
        file_name=path.name,
        last_modified=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
    )


def read_blocks(path: str, block_size: int = 500, overlap: int = 50):
    if overlap >= block_size:
        raise ValueError("overlap should be < block_size")
    step = block_size - overlap

    with open(path, encoding="utf-8") as f:
        buffer = f.read(block_size)
        i = 0
        while buffer:
            yield buffer, i
            i += 1

            new = f.read(step)
            if not new:
                break
            buffer = buffer[step:] + new


async def embed_file(
    source_id: int, path: str, embedder: IEmbedder
) -> tuple[list[ChunkData], list[list[float]]]:

    texts: list[str] = []
    chunk_data: list[ChunkData] = []

    for block, i in read_blocks(path):
        chunk_data.append(
            ChunkData(
                id=str(uuid.uuid4()),
                source_id=source_id,
                chunk_index=i,
                text=block,
                created_at=datetime.now(UTC).isoformat(),
                doc_category=None,
            )
        )
        texts.append(block)
    embedded = await embedder.embed(texts)
    return chunk_data, embedded


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class FileHash:
    path: str
    hash: str


def diff(path: str):
    local_files = {p.as_posix() for p in Path(path).rglob("*") if p.is_file()}

    with Session(engine) as session:
        results = session.exec(select(IngestedFile)).all()
    remote_files = {p.file_path for p in results}

    files_to_check = remote_files & local_files

    with Session(engine) as session:
        remote_to_check = {
            FileHash(p.file_path, p.content_hash)
            for p in session.exec(
                select(IngestedFile).where(IngestedFile.file_path.in_(files_to_check))
            ).all()
        }
    local_to_check = {FileHash(p, file_hash(Path(p))) for p in files_to_check}

    files_to_replace = local_to_check - remote_to_check

    added_files = local_files - remote_files
    removed_files = remote_files - local_files

    files_to_remove = list(removed_files | {p.path for p in files_to_replace})
    files_to_add = list({FileHash(p, file_hash(Path(p))) for p in added_files} | files_to_replace)
    return files_to_remove, files_to_add


async def ingest(
    path: str = ".docs/",
):
    files_to_remove, files_to_add = diff(path)
    vector_store = SqliteVectorStorage(engine)
    chunk_store = ChunkStore(engine, vector_store)
    with Session(engine) as session:
        to_remove: list[IngestedFile] = session.exec(
            select(IngestedFile).where(IngestedFile.file_path.in_(files_to_remove))
        ).all()
        for p in to_remove:
            await chunk_store.delete_by_source(p.id)
        session.exec(delete(IngestedFile).where(IngestedFile.id.in_([p.id for p in to_remove])))
        session.commit()

        session.add_all([to_ingested(p) for p in files_to_add])
        session.commit()

        to_add: list[IngestedFile] = session.exec(
            select(IngestedFile).where(IngestedFile.file_path.in_([p.path for p in files_to_add]))
        ).all()
        for p in to_add:
            chunks, embeddings = await embed_file(p.id, p.file_path, OllamaEmbedder())
            await chunk_store.add_chunks(chunks, embeddings)
