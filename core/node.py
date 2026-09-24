"""Pipeline node interface.

The pipeline is a chain of `PipelineNode`s composed by `Agent`. Each node
receives the shared `Context` and a `next` callable that runs the rest of
the chain, and can act before, after, or instead of it.

Contracts
---------
- `handle` calls `next(ctx)` at most once and returns the resulting
  `Context`: the pipeline is a chain, not a tree.
- Not calling `next` is a short-circuit: `ctx.response` must then be set
  (the cache, A6, and the guardrails, EV2, depend on this).
- `handle` never catches `asyncio.CancelledError`: interruption (X2) must
  propagate up to the caller.

A node can behave in three ways: transparent (just calls `next`), wrapper
(does work before and/or after calling `next`), or short-circuit (does not
call `next` and sets `ctx.response` itself).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from core.types import Context

Next = Callable[[Context], Awaitable[Context]]


class PipelineNode(ABC):
    @abstractmethod
    async def handle(self, ctx: Context, next: Next) -> Context: ...
