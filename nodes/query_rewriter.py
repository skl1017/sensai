"""Standalone-query rewriting for retrieval, before few-shot/RAG (A7, issue #32 part 2).

`rewrite_query` turns the user's last message into a standalone question that
resolves references to earlier turns (« et le deuxième ? » -> « Quelle est la
deuxième plus grande ville de France ? »). The rewrite is only ever used for
retrieval: callers store it in `ctx.state["query"]` and leave the message sent
to the model (`ctx.user_input`) untouched.

This is a plain function, not a `PipelineNode`: it is meant to be called by
the few-shot section today and the RAG section later (Story 6), each deciding
for itself whether retrieval needs a rewritten query.
"""

from __future__ import annotations

import logging

from core.llm import ILLM
from core.types import Message

logger = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 500
_RELEVANT_ROLES = {"user", "assistant"}

_SYSTEM_PROMPT = (
    "Rewrite the last user message into a standalone query that can be "
    "understood without the conversation; resolve pronouns and references; "
    "keep the user's language; output only the rewritten query."
)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_MESSAGE_CHARS:
        return text
    return text[:_MAX_MESSAGE_CHARS]


def _clean(text: str) -> str:
    """Strip surrounding whitespace/quotes and a leading "Query:" label."""
    cleaned = text.strip()

    lowered = cleaned.lower()
    if lowered.startswith("query:"):
        cleaned = cleaned[len("query:") :].strip()

    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()

    return cleaned


async def rewrite_query(
    llm: ILLM,
    history: list[Message],
    user_input: str,
    max_turns: int = 3,
) -> str:
    """Rewrite `user_input` into a standalone query using the last `max_turns` turns.

    Returns `user_input` unchanged (no LLM call) when `history` is empty or
    holds no user/assistant message. Any failure (LLM error, empty output) is
    logged with `logger.warning` and falls back to `user_input` too, so
    retrieval always gets *some* query. `asyncio.CancelledError` is never
    caught (it isn't an `Exception`, so it propagates on its own).
    """
    relevant = [m for m in history if m.role in _RELEVANT_ROLES]
    user_indexes = [i for i, m in enumerate(relevant) if m.role == "user"]
    start = user_indexes[-max_turns] if len(user_indexes) >= max_turns else 0
    turns = relevant[start:]
    if not turns:
        return user_input

    transcript = "\n".join(f"{m.role}: {_truncate(m.content)}" for m in turns)
    prompt = f"Conversation:\n{transcript}\n\nLast user message: {_truncate(user_input)}"

    messages = [Message("system", _SYSTEM_PROMPT), Message("user", prompt)]

    try:
        text = ""
        async for chunk in llm.chat(messages):
            text += chunk.text
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back
        logger.warning("Query rewrite failed, falling back to raw input: %s", exc)
        return user_input

    rewritten = _clean(text)
    if not rewritten:
        logger.warning("Query rewrite returned an empty result, falling back to raw input")
        return user_input

    return rewritten
