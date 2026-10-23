"""Request-scoped context readable by tools.

`ITool.run` does not receive the `Context` (see `core/tool.py`): when a tool
needs request information, it reads these `ContextVar`s instead. `Agent.run`
publishes `session_id` and `depth` for the duration of a run; `LoggingNode`
publishes `trace_id`. Tools read them with `.get()`.

`subagent_calls` is the one mutable value: the root `Agent.run` (the one that
finds it unset) publishes a fresh list, and `AgentTool` appends its name to it
on every call. Sub-agents run inside the parent's tool call, so they see the
same list: it counts sub-agent calls across the whole request, nesting
included (A2 `max_calls`).
"""

from __future__ import annotations

from contextvars import ContextVar

session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
depth: ContextVar[int] = ContextVar("depth", default=0)
trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
subagent_calls: ContextVar[list[str] | None] = ContextVar("subagent_calls", default=None)
