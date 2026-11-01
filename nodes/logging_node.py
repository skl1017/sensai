"""`LoggingNode`: measures every request and writes one event (EV3, wiki §7.1).

First node of the pipeline: it wraps everything. Before `next` it creates the
`trace_id` (uuid4 hex, in `state["trace_id"]` and `runtime.trace_id`) and
swaps `ctx.llm` / `ctx.tools` for measuring proxies (`MeasuredLLM`,
`MeasuredTool`) that time each call, then restores the originals in a
`finally`. `ctx.tools` is replaced by a *new* dict: the agent's own dict is
shared across runs and is never mutated.

In the `finally` it writes `state["metrics"]` (`{latency_ms, ttft_ms,
llm_calls, tool_calls, prompt_tokens, completion_tokens}`) and one event
(see `storage/event_log.py`). Status is `ok`, `error` or `cancelled`; the
exception is always re-raised (`CancelledError` is never swallowed). A failure
to write the log is logged as a warning and never masks the original error.

State keys read (all optional): `session_id`, `persona`, `prompt`,
`parent_trace_id`, `cache_hit`, `blocked`, `react_truncated`, `eval_scores`,
`pii_found`, `context_stats`. Absent keys land as `null` in the event.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from core import runtime
from core.llm import ILLM
from core.node import Next, PipelineNode
from core.tool import ITool
from core.types import Context, LLMChunk, Message
from storage.event_log import EventLog

logger = logging.getLogger(__name__)

_KNOWN_PARAMS = {"sink", "redact_inputs"}
_DEFAULT_SINK = "logs/events.jsonl"
_ERROR_MESSAGE_MAX = 500


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _error(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:_ERROR_MESSAGE_MAX]}


class MeasuredLLM(ILLM):
    """Transparent `ILLM` proxy recording one entry per `chat` call in `calls`."""

    def __init__(self, inner: ILLM, calls: list[dict]) -> None:
        self._inner = inner
        self.calls = calls

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._inner.model

    @property
    def max_context_tokens(self) -> int:  # type: ignore[override]
        return self._inner.max_context_tokens

    @max_context_tokens.setter
    def max_context_tokens(self, value: int) -> None:
        self._inner.max_context_tokens = value

    @property
    def options(self) -> dict:
        return self._inner.options

    def update_options(self, **kwargs) -> None:
        self._inner.update_options(**kwargs)

    def count_tokens(self, messages: list[Message]) -> int:
        return self._inner.count_tokens(messages)

    def __getattr__(self, name: str):
        # Only reached for attributes not found on the proxy itself.
        if name in ("_inner", "calls"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        schema: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        record = {
            "latency_ms": None,
            "ttft_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "structured": schema is not None,
            "error": None,
        }
        self.calls.append(record)
        start = time.perf_counter()
        try:
            async for chunk in self._inner.chat(messages, tools=tools, schema=schema):
                if record["ttft_ms"] is None and (chunk.text or chunk.tool_calls):
                    record["ttft_ms"] = _ms(start)
                if chunk.done:
                    record["prompt_tokens"] = chunk.prompt_tokens
                    record["completion_tokens"] = chunk.completion_tokens
                yield chunk
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["latency_ms"] = _ms(start)


class MeasuredTool(ITool):
    """Transparent `ITool` proxy recording one entry per `run` call in `calls`."""

    def __init__(self, inner: ITool, calls: list[dict]) -> None:
        self._inner = inner
        self.calls = calls

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._inner.name

    @property
    def description(self) -> str:  # type: ignore[override]
        return self._inner.description

    @property
    def parameters(self) -> dict:  # type: ignore[override]
        return self._inner.parameters

    @property
    def enforces_own_timeout(self) -> bool:  # type: ignore[override]
        return self._inner.enforces_own_timeout

    @property
    def persist_exchange(self) -> bool:  # type: ignore[override]
        return self._inner.persist_exchange

    def __getattr__(self, name: str):
        if name in ("_inner", "calls"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def run(self, **kwargs) -> str:
        record = {"name": self._inner.name, "latency_ms": None, "error": None}
        self.calls.append(record)
        start = time.perf_counter()
        try:
            return await self._inner.run(**kwargs)
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["latency_ms"] = _ms(start)


def _prompt_version(prompt) -> str | None:
    if not isinstance(prompt, dict) or not prompt.get("template"):
        return None
    return f"{prompt['template']}@{prompt.get('version')}"


class LoggingNode(PipelineNode):
    def __init__(self, event_log: EventLog, redact_inputs: bool = False) -> None:
        self.event_log = event_log
        self.redact_inputs = redact_inputs

    @classmethod
    def from_config(cls, params: dict, deps) -> LoggingNode:
        """Build from `agent.yaml` params (`sink`, `redact_inputs`).

        Uses `deps.event_log` when provided (shared with the eval node), otherwise opens
        `sink` (default `logs/events.jsonl`). Unknown keys raise `ValueError`.

        `deps` is intentionally untyped here (not `Deps`) so this module never has to
        import `app.py`, which would break the `nodes/` -> `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown LoggingNode param(s): {', '.join(sorted(unknown))}")
        event_log = getattr(deps, "event_log", None)
        if event_log is None:
            event_log = EventLog.open(params.get("sink", _DEFAULT_SINK))
        return cls(event_log, redact_inputs=bool(params.get("redact_inputs", False)))

    async def handle(self, ctx: Context, next: Next) -> Context:
        state = ctx.state
        trace_id = uuid.uuid4().hex
        state["trace_id"] = trace_id
        token = runtime.trace_id.set(trace_id)

        llm_calls: list[dict] = []
        tool_calls: list[dict] = []
        orig_llm, orig_tools = ctx.llm, ctx.tools
        ctx.llm = MeasuredLLM(orig_llm, llm_calls)
        ctx.tools = {name: MeasuredTool(t, tool_calls) for name, t in orig_tools.items()}

        ts = datetime.now(UTC).isoformat()
        start = time.perf_counter()
        status = "ok"
        error: dict | None = None
        out = ctx
        try:
            out = await next(ctx)
            return out
        except BaseException as exc:
            if isinstance(exc, Exception):
                status, error = "error", _error(exc)
            else:  # CancelledError (and KeyboardInterrupt/SystemExit): not a failure
                status = "cancelled"
            raise
        finally:
            latency_ms = _ms(start)
            for c in {id(ctx): ctx, id(out): out}.values():
                c.llm, c.tools = orig_llm, orig_tools
            runtime.trace_id.reset(token)

            out_state = out.state
            if out_state is not state:
                state = out_state
            ttfts = [c["ttft_ms"] for c in llm_calls if c["ttft_ms"] is not None]
            prompt_tokens = sum(c["prompt_tokens"] or 0 for c in llm_calls)
            completion_tokens = sum(c["completion_tokens"] or 0 for c in llm_calls)
            state["metrics"] = {
                "latency_ms": latency_ms,
                "ttft_ms": ttfts[0] if ttfts else None,
                "llm_calls": len(llm_calls),
                "tool_calls": len(tool_calls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            try:
                event = {
                    "trace_id": trace_id,
                    "parent_trace_id": state.get("parent_trace_id"),
                    "ts": ts,
                    "session_id": state.get("session_id"),
                    "persona": state.get("persona"),
                    "prompt_version": _prompt_version(state.get("prompt")),
                    "input": self._input(ctx.user_input),
                    "response": out.response,
                    "status": status,
                    "latency_ms": latency_ms,
                    "llm_calls": llm_calls,
                    "tool_calls": tool_calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_hit": state.get("cache_hit"),
                    "blocked": state.get("blocked"),
                    "react_truncated": state.get("react_truncated"),
                    "eval_scores": state.get("eval_scores"),
                    "pii_found": state.get("pii_found"),
                    "context_stats": state.get("context_stats"),
                    "error": error,
                }
                await self.event_log.write(event)
            except Exception:
                logger.warning("LoggingNode: failed to write event %s", trace_id, exc_info=True)

    def _input(self, text: str) -> str:
        if not self.redact_inputs:
            return text
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
