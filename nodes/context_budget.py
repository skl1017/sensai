"""Token budget + rolling summary for `ContextBuilderNode` (#31, wiki M2).

`apply_budget` is a plain function, not a `PipelineNode`: it is meant to be
called by `ContextBuilderNode` once the system message and history have been
assembled, right before the current user message is handed to `ReActLoopNode`.
It never touches `ctx` directly (no `Context` import needed) so it stays
trivially testable with `FakeLLM`.

Algorithm (wiki §9.1 M2)
-------------------------
- `budget = llm.max_context_tokens - reserve_output`; compression is
  attempted once `llm.count_tokens(messages)` exceeds `trigger_ratio *
  budget`.
- A *turn* is a user message plus everything up to (excluding) the next user
  message (so an assistant `tool_calls` message and its `tool` results never
  get split apart).
- The leading system message, the last `keep_last_turns` turns and the
  current user message (always the last element of `messages`) are kept
  verbatim; older turns are folded into a rolling summary.
- The summary is persisted as `state["summary"] = {"text", "covered",
  "digest"}`: `covered` is how many leading history messages it accounts
  for, `digest` a sha256 over their `role`+`content` so a later turn can
  detect whether that prefix is still the same conversation (a fork/branch
  changes it) before trusting the cached text. When the digest still
  matches, the already-covered messages are replaced by the summary without
  calling the LLM again; only the turns that became "old" since then are
  folded in (merged with the previous summary text).
- If the LLM call fails or returns nothing, or the result is still over the
  hard budget (not just the trigger), the oldest turns are dropped instead
  (never the system message, never the current user message), and a
  `logger.warning` is emitted.
- `state["context_stats"] = {"tokens_before", "tokens_after", "compressed"}`
  is always written, `compressed` being `False` when the trigger was never
  reached.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from core.llm import ILLM
from core.types import Message

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "You are compressing an earlier part of a conversation between a user and an "
    "assistant into a concise rolling summary. You may be given a previous summary "
    "to merge with new turns: fold everything into a single, updated summary. "
    "Preserve names, numbers, decisions, constraints and open tasks exactly as "
    "stated; drop small talk and redundant detail. Reply with the summary text "
    "only, no preamble."
)

_SUMMARY_HEADER = "Summary of the earlier conversation:\n"


def _group_turns(messages: list[Message]) -> list[list[Message]]:
    """Split `messages` into turns: a user message plus everything until the next one."""
    turns: list[list[Message]] = []
    for message in messages:
        if message.role == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _digest(messages: list[Message]) -> str:
    """sha256 over `role`+`content` of `messages`, used to detect a fork/branch."""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message.role.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message.content.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _valid_summary(raw: object, history_len: int) -> dict | None:
    """Shape-check `state["summary"]`; tolerates `None` or a malformed dict."""
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    covered = raw.get("covered")
    digest = raw.get("digest")
    if not isinstance(text, str) or not isinstance(digest, str):
        return None
    if not isinstance(covered, int) or isinstance(covered, bool):
        return None
    if covered < 0 or covered > history_len:
        return None
    return {"text": text, "covered": covered, "digest": digest}


def _render_transcript(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        label = message.role if not message.name else f"{message.role}:{message.name}"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def _build_summary_input(previous_summary: str | None, messages: list[Message]) -> str:
    parts = []
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}")
    parts.append("Conversation to fold in:\n" + _render_transcript(messages))
    return "\n\n".join(parts)


async def _summarize(llm: ILLM, previous_summary: str | None, messages: list[Message]) -> str:
    """Call the LLM to fold `messages` into `previous_summary`; `""` on failure.

    Never catches `asyncio.CancelledError` (it must propagate); any other
    exception is logged by the caller and treated like an empty result.
    """
    prompt = _build_summary_input(previous_summary, messages)
    text = ""
    async for chunk in llm.chat([Message("system", SUMMARY_PROMPT), Message("user", prompt)]):
        text += chunk.text
    return text.strip()


def _flatten(turns: list[list[Message]]) -> list[Message]:
    return [message for turn in turns for message in turn]


async def apply_budget(
    messages: list[Message],
    llm: ILLM,
    state: dict,
    reserve_output: int = 1024,
    trigger_ratio: float = 0.8,
    keep_last_turns: int = 4,
) -> list[Message]:
    """Return a budget-fitted copy of `messages`; never mutates `messages` or its items.

    `messages` is `[system?] + history + [current_user_message]`, exactly as
    `ContextBuilderNode` assembles it. Reads/writes `state["summary"]` and
    always writes `state["context_stats"]`.
    """
    system = messages[0] if messages and messages[0].role == "system" else None
    rest = messages[1:] if system is not None else list(messages)

    tokens_before = llm.count_tokens(messages)
    budget = llm.max_context_tokens - reserve_output
    trigger = trigger_ratio * budget

    if not rest or tokens_before <= trigger:
        state["context_stats"] = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_before,
            "compressed": False,
        }
        return list(messages)

    current = rest[-1]
    history = rest[:-1]

    turns = _group_turns(history)
    keep_n = min(keep_last_turns, len(turns))
    old_turns = turns[: len(turns) - keep_n]
    kept_turns = turns[len(turns) - keep_n :]
    old_end = sum(len(turn) for turn in old_turns)

    raw_summary = state.get("summary")
    valid_existing = _valid_summary(raw_summary, len(history))
    if valid_existing is not None:
        covered_prefix_digest = _digest(history[: valid_existing["covered"]])
        if covered_prefix_digest != valid_existing["digest"]:
            valid_existing = None  # fork/other branch: ignore, but don't delete state["summary"]

    summary_text: str | None = valid_existing["text"] if valid_existing else None
    new_covered = valid_existing["covered"] if valid_existing else 0
    new_digest = valid_existing["digest"] if valid_existing else None

    remaining_old_turns = list(old_turns)
    if valid_existing is not None:
        acc = 0
        already_covered = 0
        for turn in old_turns:
            if acc + len(turn) <= valid_existing["covered"]:
                acc += len(turn)
                already_covered += 1
            else:
                break
        remaining_old_turns = old_turns[already_covered:]

    if remaining_old_turns:
        to_summarize = _flatten(remaining_old_turns)
        text = ""
        already_warned = False
        previous_text = valid_existing["text"] if valid_existing else None
        try:
            text = await _summarize(llm, previous_text, to_summarize)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - fallback path, never crash the turn
            logger.warning("Summarization failed, dropping oldest turns instead: %s", exc)
            already_warned = True

        if text:
            summary_text = text
            new_covered = old_end
            new_digest = _digest(history[:old_end])
        else:
            if not already_warned:
                logger.warning("Summarization returned no text, dropping oldest turns instead")
        remaining_old_turns = []  # summarized above, or dropped as the failure fallback

    new_system = system
    if summary_text:
        base = system.content if system is not None else ""
        suffix = f"{_SUMMARY_HEADER}{summary_text}"
        content = f"{base}\n\n{suffix}" if base else suffix
        new_system = Message("system", content)

    dropped_any = False
    candidate: list[Message] = []
    while True:
        body = _flatten(remaining_old_turns) + _flatten(kept_turns)
        candidate = ([new_system] if new_system is not None else []) + body + [current]
        tokens_now = llm.count_tokens(candidate)
        if tokens_now <= budget or (not remaining_old_turns and not kept_turns):
            break
        if remaining_old_turns:
            remaining_old_turns = remaining_old_turns[1:]
        else:
            kept_turns = kept_turns[1:]
        dropped_any = True

    if dropped_any:
        logger.warning("Still over budget after compression, dropped oldest turns")

    if summary_text is not None:
        state["summary"] = {"text": summary_text, "covered": new_covered, "digest": new_digest}

    tokens_after = llm.count_tokens(candidate)
    state["context_stats"] = {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "compressed": True,
    }
    return candidate
