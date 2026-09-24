"""Request-scoped context readable by tools.

`ITool.run` does not receive the `Context` (see `core/tool.py`): when a tool
needs request information, it reads these `ContextVar`s instead. `Agent.run`
publishes `session_id` and `depth` for the duration of a run; `LoggingNode`
publishes `trace_id`. Tools read them with `.get()`.
"""

from __future__ import annotations

from contextvars import ContextVar

session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
depth: ContextVar[int] = ContextVar("depth", default=0)
trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
