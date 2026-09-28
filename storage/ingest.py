from dataclasses import dataclass


@dataclass
class ChunkData:
    id: str
    text: str
    source_id: str
    chunk_index: int
    doc_category: str | None
    created_at: str
