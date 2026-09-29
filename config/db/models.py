from datetime import datetime
from pydantic import NaiveDatetime
from sqlmodel import Field, Relationship, SQLModel


class IngestedFile(SQLModel, table=True):
    __tablename__ = "ingested_files"

    id: int | None = Field(default=None, primary_key=True)
    file_name: str = Field(index=True)
    file_path: str = Field(unique=True, index=True)
    last_modified: NaiveDatetime
    content_hash: str
    ingested_at: NaiveDatetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="synced")

    chunks: list["Chunk"] = Relationship(
        back_populates="source_file",
        cascade_delete=True,
    )


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: str = Field(primary_key=True)
    text: str
    source_id: str
    doc_category: str | None = None
    created_at: str | None = None
    chunk_index: int | None = None

    ingested_file_id: int | None = Field(
        default=None, foreign_key="ingested_files.id", ondelete="CASCADE"
    )
    source_file: IngestedFile | None = Relationship(back_populates="chunks")
