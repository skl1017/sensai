from __future__ import annotations

from typing import Any

from core.llm import IEmbedder
from core.types import Context
from nodes.context_section import ContextSection
from storage.chunk_store import ChunkStore

_KNOWN_PARAMS = {"k", "min_score"}


class RAGSection(ContextSection):
    def __init__(
        self,
        embedder: IEmbedder,
        chunk_store: ChunkStore,
        k: int = 4,
        min_score: float = 0.0,
        priority: int = 60,
    ) -> None:
        self.embedder = embedder
        self.chunk_store = chunk_store
        self.k = k
        self.min_score = min_score
        self.priority = priority

    @staticmethod
    def _render_passage(index: int, chunk) -> str:
        passage = f"[{index}] {chunk.text}"
        if chunk.source_name:
            passage += f"\n(source: {chunk.source_name})"
        return passage

    async def build(self, ctx: Context) -> str | None:
        query = ctx.state.get("query") or ctx.user_input
        query_vector = (await self.embedder.embed([query]))[0]

        hits = await self.chunk_store.search(query_vector, self.k)
        for _, score in hits:
            print(score)
        hits = [(chunk, score) for chunk, score in hits if score >= self.min_score]
        if not hits:
            return None
        print(hits)

        ctx.state["retrieved_chunks"] = [
            {
                "chunk_id": chunk.id,
                "source_id": chunk.source_id,
                "source_name": chunk.source_name,
                "score": score,
            }
            for chunk, score in hits
        ]

        passages = [self._render_passage(i, chunk) for i, (chunk, _) in enumerate(hits, start=1)]
        return "Relevant context (use it if it helps answer):\n\n" + "\n\n".join(passages)

    @classmethod
    def from_config(cls, params: dict, deps) -> RAGSection:
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown RAGSection param(s): {', '.join(sorted(unknown))}")
        if deps.embedder is None:
            raise ValueError("rag section requires an embedder")

        kwargs: dict[str, Any] = {k: params[k] for k in _KNOWN_PARAMS if k in params}
        return cls(deps.embedder, deps.stores.chunk_store, **kwargs)
