from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    chunk_id: str
    distance: float


class IVectorStorage(ABC):
    @abstractmethod
    async def add(self, chunk_id: str, embedding: list[float]) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        allowed_chunk_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]: ...

    @abstractmethod
    async def delete(self, chunk_id: str) -> None: ...
