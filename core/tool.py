"""Tool interface.

`ITool` is the contract every tool implements and the `ReActLoopNode` calls
uniformly, regardless of the concrete tool.

Contracts
---------
- `run` always returns a `str`; the model only reads text back.
- Raising an exception from `run` is legitimate: the node turns it into an
  observation, it is not a crash.
- `run` never blocks the event loop: synchronous I/O goes through
  `asyncio.to_thread` so interruption (X2) and streaming stay smooth.
- `run` does not receive the `Context`: when a tool needs request
  information (e.g. the session id), it reads it from `ContextVar`s
  declared in `core/runtime.py`.
- Keep return values short (truncate at the source) and never leak secrets
  in returns or error messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ITool(ABC):
    name: str  # unique, snake_case
    description: str  # read by the model: say WHEN to use it
    parameters: dict  # JSON Schema, type "object", additionalProperties: false recommended
    enforces_own_timeout: bool = False  # True: the node does not apply its own timeout

    @abstractmethod
    async def run(self, **kwargs) -> str: ...
