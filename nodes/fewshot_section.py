"""`FewShotSection`: injects the k closest question/answer examples (#32).

The example bank currently lives in a YAML file (`config/fewshot.yaml`, a
list of `{question, answer}` pairs) and is embedded once, lazily, on first
`build()`. Loading is kept isolated in `_load_bank`/`_ensure_loaded` so it
is easy to swap for the vector store this is meant to migrate to (#97)
without touching `build()` itself.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import yaml

from core.llm import IEmbedder
from core.types import Context
from nodes.context_section import ContextSection

_KNOWN_PARAMS = {"path", "k", "min_score"}
_DEFAULT_FILENAME = "fewshot.yaml"


def _validate_bank(data: Any) -> list[dict[str, str]]:
    """Validate the raw YAML payload and return the list of examples.

    Expected shape: `{"examples": [{"question": str, "answer": str}, ...]}`,
    every question/answer a non-empty string. Anything else raises
    `ValueError`.
    """
    if not isinstance(data, dict) or "examples" not in data:
        raise ValueError("fewshot bank must be a mapping with an 'examples' key")

    examples = data["examples"]
    if not isinstance(examples, list):
        raise ValueError("fewshot bank 'examples' must be a list")

    bank: list[dict[str, str]] = []
    for item in examples:
        if not isinstance(item, dict):
            raise ValueError("each fewshot bank entry must be a mapping")
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("each fewshot bank entry needs a non-empty string 'question'")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("each fewshot bank entry needs a non-empty string 'answer'")
        bank.append({"question": question, "answer": answer})

    return bank


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FewShotSection(ContextSection):
    """Renders the `k` bank examples closest to the query, by cosine similarity.

    The bank is loaded (YAML, via `asyncio.to_thread`), validated, and
    embedded exactly once across the section's lifetime, on the first
    `build()` call; an `asyncio.Lock` guards that first load against
    concurrent builds (sections run concurrently via `asyncio.gather`).
    The query is `ctx.state["query"]` when set (written by the node when
    `rewrite_query` is on), otherwise `ctx.user_input`.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        path: str | Path,
        k: int = 3,
        priority: int = 40,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.path = Path(path)
        self.k = k
        self.priority = priority
        self.min_score = min_score
        self._bank: list[dict[str, str]] | None = None
        self._vectors: list[list[float]] | None = None
        self._lock = asyncio.Lock()

    def _load_bank(self) -> list[dict[str, str]]:
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _validate_bank(data)

    async def _ensure_loaded(self) -> tuple[list[dict[str, str]], list[list[float]]]:
        if self._bank is not None:
            return self._bank, self._vectors  # type: ignore[return-value]

        async with self._lock:
            if self._bank is None:
                bank = await asyncio.to_thread(self._load_bank)
                vectors = (
                    await self.embedder.embed([example["question"] for example in bank])
                    if bank
                    else []
                )
                self._bank = bank
                self._vectors = vectors

        return self._bank, self._vectors  # type: ignore[return-value]

    async def build(self, ctx: Context) -> str | None:
        bank, vectors = await self._ensure_loaded()
        if not bank:
            return None

        query = ctx.state.get("query") or ctx.user_input
        query_vector = (await self.embedder.embed([query]))[0]

        scored = [
            (_cosine(query_vector, vector), example)
            for vector, example in zip(vectors, bank, strict=True)
        ]
        scored = [pair for pair in scored if pair[0] >= self.min_score]
        if not scored:
            return None

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[: self.k]

        blocks = [f"Q: {example['question']}\nA: {example['answer']}" for _, example in top]
        return "Examples of good answers:\n\n" + "\n\n".join(blocks)

    @classmethod
    def from_config(cls, params: dict, deps) -> FewShotSection:
        """Build from `agent.yaml`'s `fewshot:` sub-dict of the `context` node.

        `path` defaults to `deps.config_dir / "fewshot.yaml"`. `deps` is
        intentionally untyped (not `Deps`) so this module never imports
        `app.py`, which would break the `nodes/` -> `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown FewShotSection param(s): {', '.join(sorted(unknown))}")

        if deps.embedder is None:
            raise ValueError("fewshot section requires an embedder")

        path = params.get("path", deps.config_dir / _DEFAULT_FILENAME)
        kwargs: dict[str, Any] = {}
        if "k" in params:
            kwargs["k"] = params["k"]
        if "min_score" in params:
            kwargs["min_score"] = params["min_score"]

        return cls(deps.embedder, path, **kwargs)
