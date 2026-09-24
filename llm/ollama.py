"""Ollama adapters: `Ollama` implements `ILLM`, `OllamaEmbedder` implements `IEmbedder`.

`Ollama` receives an `IEmbedder` (injected, not inherited) so embedding-based
features can be reached from the LLM without coupling it to one backend.

Design
------
- `chat` streams the response as it POSTs `/api/chat` with `stream: True`:
  each line of the response body is one JSON object, turned into one
  `LLMChunk`. The last chunk has `done=True` and carries the real
  `prompt_tokens`/`completion_tokens` (from `prompt_eval_count` /
  `eval_count`), which callers use for EV3 measurement and to recalibrate
  the pre-call estimate `count_tokens` gives (a cheap `len(content) // 4`
  heuristic, not the tokenizer's real count).
- `num_ctx` is always sent explicitly in `options`, seeded from
  `max_context_tokens`, because Ollama silently falls back to a small
  default context window otherwise.
- Tool calls from Ollama don't carry a call id, so the adapter mints one
  with `uuid.uuid4().hex` per call.
- Cancelling the task that runs `chat` propagates through the `async with
  self._client.stream(...)` block, which closes the HTTP connection and
  lets Ollama stop generating (X2). `asyncio.CancelledError` is never
  caught here — only `httpx.TransportError` is translated to
  `LLMUnavailable`.
- Failures are translated into the `LLMError` hierarchy from `core.llm`
  (`LLMUnavailable`, `LLMModelNotFound`, `LLMToolsUnsupported`, or the
  generic `LLMError`) so callers never need to know this is Ollama.

Wire format notes
------------------
- A `role="tool"` message is sent with a `tool_name` field (Ollama has no
  `tool_call_id` on its wire format, so the tool's name is what lets the
  model correlate the observation with its call).
- `tool_calls[].function.arguments` is normally an object on the wire, but
  some Ollama versions send it as a JSON-encoded string; the adapter
  normalizes both to a `dict` (missing/`None` becomes `{}`).
- Ollama never returns a call id: `ToolCall.id` is always a freshly minted
  `uuid.uuid4().hex`, not something the server assigned.
- Written against the Ollama 0.34.x `/api/chat` and `/api/embed` API.
  Re-verify against a real server with the slow integration suite
  (`tests/test_ollama_integration.py`) before relying on this against a
  different Ollama version.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx

from core.llm import (
    ILLM,
    IEmbedder,
    LLMError,
    LLMModelNotFound,
    LLMToolsUnsupported,
    LLMUnavailable,
)
from core.types import LLMChunk, Message, ToolCall

_EXCERPT_CHARS = 200


def _unreachable(host: str) -> LLMUnavailable:
    return LLMUnavailable(f"Ollama unreachable at {host}: is `ollama serve` running?")


async def _check(response: httpx.Response, *, model: str, has_tools: bool) -> None:
    body = await response.aread()
    try:
        data = json.loads(body)
        server_msg = str(data.get("error", data))
    except json.JSONDecodeError:
        server_msg = body.decode("utf-8", errors="replace")
    server_msg = server_msg[:_EXCERPT_CHARS]

    if response.status_code == 404:
        raise LLMModelNotFound(f"model '{model}' not found: run `ollama pull {model}`")
    if response.status_code == 400 and has_tools:
        raise LLMToolsUnsupported(
            f"model '{model}' rejected tools ({server_msg}): "
            "choose a model that supports tools (e.g. llama3.1, qwen2.5)"
        )
    raise LLMError(f"Ollama HTTP {response.status_code}: {server_msg}")


class OllamaEmbedder(IEmbedder):
    """`IEmbedder` adapter for a local Ollama server."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self._client = (
            client if client is not None else httpx.AsyncClient(base_url=host, timeout=None)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OllamaEmbedder:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            r = await self._client.post("/api/embed", json={"model": self.model, "input": texts})
            if r.status_code >= 400:
                await _check(r, model=self.model, has_tools=False)
        except httpx.TransportError as exc:
            raise _unreachable(self.host) from exc
        return r.json()["embeddings"]


class Ollama(ILLM):
    """`ILLM` adapter for a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1",
        embedder: IEmbedder | None = None,
        host: str = "http://localhost:11434",
        max_context_tokens: int = 8192,
        options: dict | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.embedder = embedder
        self.host = host
        self.max_context_tokens = max_context_tokens
        self._options = {"num_ctx": max_context_tokens, **(options or {})}
        self._client = (
            client if client is not None else httpx.AsyncClient(base_url=host, timeout=None)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Ollama:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        schema: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        payload: dict = {
            "model": self.model,
            "stream": True,
            "options": dict(self._options),
            "messages": [self._to_wire(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
        if schema is not None:
            payload["format"] = schema

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    await _check(r, model=self.model, has_tools=bool(tools))
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            f"Ollama sent a malformed line: {line[:_EXCERPT_CHARS]!r}"
                        ) from exc
                    if "error" in data:
                        raise LLMError(f"Ollama error: {data['error']}")
                    msg = data.get("message", {})
                    calls = [
                        ToolCall(
                            id=uuid.uuid4().hex,
                            name=c["function"]["name"],
                            arguments=self._parse_arguments(c["function"].get("arguments")),
                        )
                        for c in msg.get("tool_calls", [])
                    ]
                    yield LLMChunk(
                        text=msg.get("content", ""),
                        tool_calls=calls,
                        done=data.get("done", False),
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                    )
        except httpx.TransportError as exc:
            raise _unreachable(self.host) from exc

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content) for m in messages) // 4 + 4 * len(messages)

    @staticmethod
    def _parse_arguments(arguments: object) -> dict:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"Ollama tool call arguments are not valid JSON: {arguments!r}"
                ) from exc
        return arguments

    @staticmethod
    def _to_wire(m: Message) -> dict:
        d = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
            ]
        if m.role == "tool":
            d["tool_name"] = m.name
        return d
