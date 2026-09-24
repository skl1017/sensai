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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

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
