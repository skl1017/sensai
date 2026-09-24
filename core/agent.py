"""Agent: builds the pipeline chain once and runs it per request.

`Agent` contains no business logic: it assembles the `PipelineNode` chain in
`__init__` (so the order is frozen for the agent's lifetime and `run` stays
minimal), then feeds each request through it. The conversation history
belongs to the caller, not the `Agent`: `run` copies it into a fresh list
before handing it to the pipeline, so nodes appending to `ctx.messages` never
mutate the caller's list, and the caller can swap persona or storage without
losing anything.

`run` also publishes `session_id` and `depth` (from `state`) into the
`ContextVar`s declared in `core/runtime.py`, for the duration of the chain
call only: tools that need this request info read it from there since they
don't receive the `Context`. Both are reset in a `finally`, so interruption
(`asyncio.CancelledError`, which is never caught) still leaves the vars
clean.
"""

from __future__ import annotations

from core import runtime
from core.llm import ILLM
from core.node import Next, PipelineNode
from core.tool import ITool
from core.types import Context, Message


class Agent:
    def __init__(self, llm: ILLM, tools: list[ITool], pipeline: list[PipelineNode]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.pipeline = pipeline
        self._chain = self._build_chain()

    def _build_chain(self) -> Next:
        handler: Next = self._terminal
        for node in reversed(self.pipeline):  # the last node is the innermost
            handler = self._wrap(node, handler)
        return handler

    @staticmethod
    def _wrap(node: PipelineNode, nxt: Next) -> Next:
        async def call(ctx: Context) -> Context:
            return await node.handle(ctx, nxt)

        return call

    async def _terminal(self, ctx: Context) -> Context:
        return ctx

    async def run(
        self,
        user_input: str,
        history: list[Message],
        state: dict | None = None,
    ) -> Context:
        ctx = Context(user_input, list(history), self.llm, self.tools, state=state or {})
        session_token = runtime.session_id.set(ctx.state.get("session_id"))
        depth_token = runtime.depth.set(ctx.state.get("depth", 0))
        try:
            return await self._chain(ctx)
        finally:
            runtime.session_id.reset(session_token)
            runtime.depth.reset(depth_token)
