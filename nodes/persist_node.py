"""`PersistNode`: saves the exchange after the rest of the pipeline runs (#24).

Sits right after `LoggingNode` (pipeline position 2, wiki §9.2/§9.3): it
calls `next` *first* and only appends to the store once the whole downstream
chain has returned successfully. There is deliberately no `try`/`except`
around that call, so a tool error that reaches this node's caller as a raised
exception, or an interruption (`asyncio.CancelledError`, X2), saves nothing —
the store never ends up holding a half-finished exchange. A node that wants
its own request captured still has to call `next` normally and let it
return; a short-circuit (cache hit, guardrail block) is saved exactly like
any other turn, since `ctx.response` is set either way.

Only the current user/assistant pair is ever appended: whatever
`ReActLoopNode` added to `ctx.messages` along the way (tool calls,
observations, the budget-exhausted system instruction) is not part of the
persisted node — the store's tree is a conversation history, not a replay
log. One exception: a tool that sets `persist_exchange = True` (e.g.
`ask_user`, whose result is information the user gave) has its exchange
condensed to one `[Asked the user: ... -> ...]` line, prepended to the
persisted assistant message from `ctx.state["react_trace"]`, so the answer
is still in the next turn's history. The node only reads the flag on the
tool (through `ctx.tools`), it never knows which tool it is.

`ctx.state["blocked"]` (set by `InputGuardrailNode`) is skipped unless
`persist_blocked` is `True`: a blocked request has no real assistant answer
worth keeping in the tree by default. A missing `session_id` (e.g. a bare
`Agent.run` in a test, with no entry point seeding state) also skips the
write — there is nowhere to append to.
"""

from __future__ import annotations

import asyncio

from core.node import Next, PipelineNode
from core.types import Context, Message

_KNOWN_PARAMS = {"dir", "persist_blocked"}
_NOTE_MAX_CHARS = 200


def _exchange_notes(ctx: Context) -> list[str]:
    """One `[Asked the user: <question> -> <answer>]` line per `persist_exchange` tool call."""
    notes = []
    for entry in ctx.state.get("react_trace") or []:
        tool = ctx.tools.get(entry["action"])
        observation = entry["observation"]
        if not getattr(tool, "persist_exchange", False) or observation.startswith("Error:"):
            continue
        question = str(entry["args"].get("question", ""))[:_NOTE_MAX_CHARS]
        notes.append(f"[Asked the user: {question} -> {observation[:_NOTE_MAX_CHARS]}]")
    return notes


class PersistNode(PipelineNode):
    def __init__(self, store, persist_blocked: bool = False) -> None:
        self.store = store
        self.persist_blocked = persist_blocked

    @classmethod
    def from_config(cls, params: dict, deps) -> PersistNode:
        """Build from `agent.yaml` params; wires the store from `deps.stores.conversation`.

        `dir` is accepted but ignored here: it is consumed by `app.open_stores`,
        which decides whether `ConversationStore` is on-disk or in-memory.
        Any other key raises `ValueError`.

        `deps` is intentionally untyped here (not `Deps`) so this module never has to
        import `app.py`, which would break the `nodes/` -> `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown PersistNode param(s): {', '.join(sorted(unknown))}")
        return cls(deps.stores.conversation, persist_blocked=params.get("persist_blocked", False))

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx = await next(ctx)

        state = ctx.state
        if state.get("blocked") and not self.persist_blocked:
            return ctx

        session_id = state.get("session_id")
        if not session_id:
            return ctx

        answer = "\n".join(filter(None, [*_exchange_notes(ctx), ctx.response]))
        await asyncio.to_thread(
            self.store.append,
            session_id,
            state.get("parent_id"),
            [Message("user", ctx.user_input), Message("assistant", answer)],
            artifact=state.get("artifact"),
            summary=state.get("summary"),
        )
        return ctx
