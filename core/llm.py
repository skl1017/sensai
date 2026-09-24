"""LLM and embedder interfaces.

`ILLM` is the contract the rest of the pipeline expects from a language
model backend, `IEmbedder` the contract for a text-embedding backend.
Neither interface imports a concrete backend: adapters (e.g. `llm/ollama.py`)
implement these interfaces and are injected into the `Agent`.

Contracts
---------
- `ILLM.chat` is declared with a plain `def` in the interface (not
  `async def`) so its return type stays `AsyncIterator[LLMChunk]`;
  implementations are `async def` generators that `yield` chunks.
- The last chunk yielded by `chat` has `done=True` and carries the token
  counts (`prompt_tokens`, `completion_tokens`) so EV3 can measure usage
  without recounting.
- When `schema` is given, the full text streamed across all chunks forms
  valid JSON conforming to that JSON Schema.
- `tools`, when given, is a list of function-calling specs (one dict per
  tool, in the backend's function-calling format).

Errors
------
Backends translate their failures into the `LLMError` hierarchy below, so
nodes and UIs can catch them without importing a concrete adapter.
`chat_json` is the shared helper for schema-constrained calls (T5): it
retries once on invalid JSON, then raises `LLMInvalidOutput`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from core.types import LLMChunk, Message


class ILLM(ABC):
    model: str
    max_context_tokens: int  # ceiling used by M2

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,  # function-calling specs
        schema: dict | None = None,  # JSON Schema to enforce (T5)
    ) -> AsyncIterator[LLMChunk]: ...

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int: ...


class IEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMError(Exception):
    """Base class for every error raised by an LLM or embedder backend."""


class LLMUnavailable(LLMError):
    """The backend cannot be reached (e.g. the server is not running)."""


class LLMModelNotFound(LLMError):
    """The requested model is not installed on the backend."""


class LLMToolsUnsupported(LLMError):
    """Tools were sent to a model that does not support function calling."""


class LLMInvalidOutput(LLMError):
    """A schema-constrained call still returned invalid JSON after the retry."""


_EXCERPT_CHARS = 200


async def chat_json(
    llm: ILLM,
    messages: list[Message],
    schema: dict,
    retries: int = 1,
) -> Any:
    """Run a schema-constrained `chat` call and return the parsed JSON.

    The streamed text is concatenated then parsed; on `json.JSONDecodeError`
    the call is replayed up to `retries` times, then `LLMInvalidOutput` is
    raised with a truncated excerpt of the last invalid text.
    """
    err: json.JSONDecodeError | None = None
    text = ""
    for _ in range(retries + 1):
        text = ""
        async for chunk in llm.chat(messages, schema=schema):
            text += chunk.text
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            err = exc

    excerpt = text[:_EXCERPT_CHARS] + ("…" if len(text) > _EXCERPT_CHARS else "")
    raise LLMInvalidOutput(
        f"invalid JSON from {llm.model} after {retries + 1} attempts: {err.msg}; "
        f"output: {excerpt!r}"
    ) from err
