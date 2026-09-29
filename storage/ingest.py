from pathlib import Path
from dataclasses import dataclass, field
from config.db.db import engine
from config.db.models import IngestedFile
from sqlmodel import Session, select, delete
from datetime import datetime, timezone
import hashlib

def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()

@dataclass
class ChunkData:
    id: str
    text: str
    source_id: str
    chunk_index: int
    doc_category: str | None
    created_at: str

@dataclass(frozen=True)
class FileHash:
    path: str
    hash: str

def ingest(path: str):
    local_files = {p.as_posix() for p in Path(path).rglob("*") if p.is_file()}

    with Session(engine) as session:
        results = session.exec(select(IngestedFile)).all()
    remote_files = {p.file_path for p in results}

    files_to_check = remote_files & local_files

    with Session(engine) as session:
        remote_to_check = {FileHash(p.file_path, p.content_hash) for p in session.exec(
            select(IngestedFile).where(IngestedFile.file_path.in_(files_to_check))
        ).all()}
    local_to_check = {FileHash(p, file_hash(Path(p))) for p in files_to_check}

    files_to_replace = remote_to_check ^ local_to_check

    added_files = local_files - remote_files
    removed_files = remote_files - local_files

    files_to_remove = removed_files | {p.path for p in files_to_replace}
    files_to_add = [
                    IngestedFile(
                        file_name=p.name,
                        file_path=p.as_posix(),
                        last_modified=datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None),
                        content_hash=file_hash(p),
                    )
                for p in map(Path, added_files | {p.path for p in files_to_replace})]

    with Session(engine) as session:
        session.exec(
            delete(IngestedFile).where(IngestedFile.file_path.in_(files_to_remove)))
        session.commit()
        session.add_all(files_to_add)
        session.commit()
