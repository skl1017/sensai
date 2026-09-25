"""Test doubles used across the suite so everything is testable without Ollama.

`FakeLLM` scripts a sequence of responses (plain text or tool calls) and
streams them as an `ILLM` would, without ever touching the network.
`FakeTool` is a minimal `ITool` whose result (or raised exception) is
configured by the test.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from core.llm import ILLM, IEmbedder
from core.tool import ITool
from core.types import LLMChunk, Message, ToolCall


class FakeLLM(ILLM):
    """Scripted `ILLM` double.

    `script` is a list of steps consumed in order, one per `chat` call:
    a `str` is streamed as text (split into a couple of chunks so
    streaming is actually exercised), a `list[ToolCall]` is yielded as a
    single chunk carrying those tool calls. Every call ends with a final
    `LLMChunk(done=True, ...)` chunk carrying token counts.
    """

    def __init__(
        self,
        script: list[str | list[ToolCall]],
        model: str = "fake",
        max_context_tokens: int = 8192,
    ) -> None:
        self.script = list(script)
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.calls: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self.script)

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content) // 4 + 1 for m in messages)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        schema: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
                "schema": dict(schema) if schema is not None else None,
            }
        )

        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        step = self.script.pop(0)

        prompt_tokens = self.count_tokens(messages)
        completion_text = ""

        if isinstance(step, list):
            completion_text = ""
            yield LLMChunk(tool_calls=step)
        else:
            words = step.split(" ")
            mid = max(1, len(words) // 2)
            first = " ".join(words[:mid])
            second = " ".join(words[mid:])
            if mid < len(words):
                first += " "
            chunks = [c for c in (first, second) if c]
            if not chunks:
                chunks = [step]
            for chunk_text in chunks:
                completion_text += chunk_text
                yield LLMChunk(text=chunk_text)
            assert completion_text == step

        completion_tokens = len(completion_text) // 4 + 1
        yield LLMChunk(
            done=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class FakeTool(ITool):
    """Configurable `ITool` double.

    `result` is either the string to return, an exception instance to
    raise, or a callable (sync or async) invoked with the call's kwargs
    whose return value (or awaited return value) is used.
    """

    def __init__(
        self,
        name: str = "fake_tool",
        result: str | BaseException | Callable[..., str | Awaitable[str]] = "ok",
        description: str = "Fake tool for tests.",
        parameters: dict | None = None,
        enforces_own_timeout: bool = False,
    ) -> None:
        self.name = name
        self.result = result
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.enforces_own_timeout = enforces_own_timeout
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))

        if isinstance(self.result, BaseException):
            raise self.result

        if callable(self.result):
            value = self.result(**kwargs)
            if hasattr(value, "__await__"):
                value = await value
            return value

        return self.result


class FakeUI:
    """`UserInterface` double: records tokens/events, answers `ask` from `answers`.

    `answers` is consumed in order; once exhausted (or when empty) `default`
    is used. Questions asked are kept in `questions` as `(question, kind)`.
    """

    def __init__(self, answers: list[str] | None = None, default: str = "yes") -> None:
        self.answers = list(answers or [])
        self.default = default
        self.tokens: list[str] = []
        self.events: list[tuple[str, dict]] = []
        self.questions: list[tuple[str, str]] = []

    def on_token(self, text: str) -> None:
        self.tokens.append(text)

    def on_event(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    async def ask(self, question: str, kind: str = "confirm") -> str:
        self.questions.append((question, kind))
        return self.answers.pop(0) if self.answers else self.default


class FakeEmbedder(IEmbedder):
    """Deterministic bag-of-words `IEmbedder` double.

    Each text becomes a fixed-size vector of hashed lowercase word counts,
    so texts sharing words have a higher cosine similarity. `calls` records
    every batch embedded.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                vec[sum(map(ord, word)) % self.dim] += 1.0
            vectors.append(vec)
        return vectors
