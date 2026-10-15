"""`MemorySection`: passive recall of long-term facts into the system prompt (#51).

The query (`ctx.state["query"]` when set, else `ctx.user_input`) is searched in
the `MemoryDB`: hybrid (keyword + vector) when an embedder is available,
keyword-only otherwise. Writing is left to the memory tools.
"""

from __future__ import annotations

from core.llm import IEmbedder
from core.types import Context
from nodes.context_section import ContextSection
from storage.memory_db import MemoryDB

_KNOWN_PARAMS = {"k"}
DEFAULT_K = 5
MAX_K = 10
MAX_VALUE_CHARS = 300


def _cut(text: str) -> str:
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."


class MemorySection(ContextSection):
    """Renders up to `k` remembered facts relevant to the current query."""

    def __init__(
        self,
        db: MemoryDB,
        embedder: IEmbedder | None = None,
        k: int = DEFAULT_K,
        priority: int = 30,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"memory section k must be an integer in 1..{MAX_K}, got {k!r}")
        self.db = db
        self.embedder = embedder
        self.k = k
        self.priority = priority

    async def build(self, ctx: Context) -> str | None:
        query = ctx.state.get("query") or ctx.user_input
        vector = (await self.embedder.embed([query]))[0] if self.embedder is not None else None
        facts = (await self.db.asearch(query, self.k, vector))[: self.k]
        if not facts:
            return None
        lines = [f"- {f.entity} / {f.key}: {_cut(f.value)}" for f in facts]
        return "Known facts (from memory):\n" + "\n".join(lines)

    @classmethod
    def from_config(cls, params: dict, deps) -> MemorySection:
        """Build from `agent.yaml`'s `memory:` sub-dict of the `context` node.

        `deps` is untyped so this module never imports `app.py`.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown MemorySection param(s): {', '.join(sorted(unknown))}")

        stores = deps.stores
        db = getattr(stores, "memory_db", None) if stores is not None else None
        if db is None:
            raise ValueError("memory section requires deps.stores.memory_db (none configured)")

        kwargs = {"k": params["k"]} if "k" in params else {}
        return cls(db, deps.embedder, **kwargs)
