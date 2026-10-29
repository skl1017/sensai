"""Resolve `[n]` citations in a RAG answer against the injected passages (#124).

`RAGSection` numbers the passages it injects (`<document id="n" ...>`) and
records them in `ctx.state["retrieved_chunks"]` with their `index`. Once the
answer is final, `ContextBuilderNode` calls `resolve_citations` to link each
`[n]` (or `[n, m]`) marker of the response back to its passage:

- `cited`: the `retrieved_chunks` entries actually cited, in passage order,
  each listed once;
- `invalid`: cited numbers that match no injected passage (hallucinated
  references), sorted. They are never shown as sources: the answer has
  already been streamed, so they are reported rather than removed.

A response without any marker gives `cited == []`: the model answered
without relying on (or without crediting) the passages.
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> set[int]:
    """Every passage number cited in `text`, from `[n]` and `[n, m]` markers."""
    return {
        int(number) for match in _CITATION_RE.finditer(text) for number in match.group(1).split(",")
    }


def resolve_citations(response: str, retrieved: list[dict]) -> dict:
    """Split the numbers cited in `response` into known passages and invalid ones."""
    numbers = cited_numbers(response)
    by_index = {chunk["index"]: chunk for chunk in retrieved}
    return {
        "cited": [by_index[n] for n in sorted(numbers) if n in by_index],
        "invalid": sorted(n for n in numbers if n not in by_index),
    }
