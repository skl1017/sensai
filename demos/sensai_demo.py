"""The real `ui.cli`, plus a dim `[state]` block under each answer (demo only).

Some features leave no trace in the CLI output (semantic cache, token budget,
few-shot selection, RAG filters and scores). This wrapper prints the
`ctx.state` keys listed in `SENSAI_SHOW_STATE` (comma-separated) after each
turn, so a recording can show them. With the variable unset it is exactly the
CLI. Nothing in the application is changed: it only wraps `ui.cli.chat_turn`
and the citations formatter that `_run_turn` calls last.

Special keys: `fewshot` (examples injected in the system prompt) and
`retrieved_chunks` (source, date and score of each passage kept).
"""

from __future__ import annotations

import json
import os

import ui.cli as cli

KEYS = [k.strip() for k in os.environ.get("SENSAI_SHOW_STATE", "").split(",") if k.strip()]
DIM, RESET = "\033[2;36m", "\033[0m"
_FEWSHOT_HEADER = "Examples of good answers:"

_last_ctx = None
_orig_chat_turn = cli.chat_turn
_orig_format_citations = cli._format_citations


async def _chat_turn(*args, **kwargs):
    global _last_ctx
    _last_ctx = await _orig_chat_turn(*args, **kwargs)
    return _last_ctx


def _fewshot(ctx) -> str:
    for message in ctx.messages:
        if message.role == "system" and _FEWSHOT_HEADER in message.content:
            block = message.content.split(_FEWSHOT_HEADER, 1)[1]
            questions = [line[3:] for line in block.splitlines() if line.startswith("Q: ")]
            return "\n".join(f"    - {q}" for q in questions)
    return "    (none)"


def _chunks(chunks) -> str:
    if not chunks:
        return "    (none)"
    return "\n".join(
        f"    [{c['index']}] {c['source_name']} ({c['document_date']}) score={c['score']:.2f}"
        for c in chunks
    )


def _state_block() -> str:
    ctx, lines = _last_ctx, []
    if ctx is None:
        return ""
    for key in KEYS:
        if key == "fewshot":
            lines.append(f"  fewshot:\n{_fewshot(ctx)}")
        elif key == "retrieved_chunks":
            lines.append(f"  retrieved_chunks:\n{_chunks(ctx.state.get(key))}")
        else:
            value = json.dumps(ctx.state.get(key), ensure_ascii=False, default=str)
            lines.append(f"  {key}: {value}")
    return f"{DIM}[state]\n" + "\n".join(lines) + f"{RESET}\n"


def _format_citations(citations):
    global _last_ctx
    out = _orig_format_citations(citations)
    if KEYS:
        out += _state_block()
    _last_ctx = None
    return out


cli.chat_turn = _chat_turn
cli._format_citations = _format_citations

if __name__ == "__main__":
    cli.main()
