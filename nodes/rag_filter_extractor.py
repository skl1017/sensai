"""Document filters deduced from the question, before retrieval (#135).

`extract_filters` asks the model whether the question restricts the search to a
file or to documents of a given date, and returns the matching
`ChunkStore.search` filters. Nothing else can be filtered on, and a question
that restricts nothing returns `{}`.

This is a plain function, not a `PipelineNode`, called by `RAGSection` like
`rewrite_query` is by `ContextBuilderNode`. The model's output is only a
suggestion: a file that isn't indexed or an unreadable date is dropped, so an
invented value can't hide every document.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from core.llm import ILLM, chat_json
from core.types import Message

logger = logging.getLogger(__name__)

_MAX_QUESTION_CHARS = 500
_MAX_FILES = 200

_NULLABLE_STRING = {"type": ["string", "null"]}
_SCHEMA = {
    "type": "object",
    "properties": {
        "file_name": _NULLABLE_STRING,
        "date_from": _NULLABLE_STRING,
        "date_to": _NULLABLE_STRING,
    },
    "required": ["file_name", "date_from", "date_to"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Decide whether the question restricts which documents to search. Answer in JSON.\n"
    "date_from, date_to (YYYY-MM-DD, both inclusive): only when the question selects "
    "documents by the date of the document itself, e.g. 'documents older than 2024' gives "
    "date_to 2023-12-31, 'files modified in October 2026' gives date_from 2026-10-01 and "
    "date_to 2026-10-31. A date about facts described inside the documents is not a "
    "restriction: 'where did I study in 2022?' gives null.\n"
    "file_name: only when the question designates one of the listed files; copy its name "
    "exactly.\n"
    "Anything else is null."
)


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _validated(raw: Any, file_names: list[str]) -> dict[str, Any]:
    """The usable filters of the model's output, dropping what can't be trusted."""
    if not isinstance(raw, dict):
        return {}
    filters: dict[str, Any] = {}
    if raw.get("file_name") in file_names:
        filters["file_name"] = raw["file_name"]
    date_from, date_to = _parse_date(raw.get("date_from")), _parse_date(raw.get("date_to"))
    if date_from is not None and date_to is not None and date_from > date_to:
        return filters
    if date_from is not None:
        filters["date_from"] = date_from
    if date_to is not None:
        filters["date_to"] = date_to
    return filters


async def extract_filters(
    llm: ILLM,
    question: str,
    file_names: list[str],
    today: date | None = None,
) -> dict[str, Any]:
    """`{file_name?, date_from?, date_to?}` deduced from `question`, `{}` when it restricts nothing.

    No LLM call without indexed files. Any failure is logged with
    `logger.warning` and returns `{}`, so retrieval still runs, over every
    document. `asyncio.CancelledError` is never caught (it isn't an `Exception`).
    """
    if not file_names:
        return {}

    listed = "\n".join(f"- {name}" for name in file_names[:_MAX_FILES])
    prompt = (
        f"Today: {(today or date.today()).isoformat()}\n"
        f"Files:\n{listed}\n"
        f"Question: {question[:_MAX_QUESTION_CHARS]}"
    )
    messages = [Message("system", _SYSTEM_PROMPT), Message("user", prompt)]

    try:
        raw = await chat_json(llm, messages, schema=_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back
        # Only the exception type is logged: messages may echo user text.
        logger.warning("RAG filter extraction failed, not filtering (%s)", type(exc).__name__)
        return {}

    return _validated(raw, file_names)
