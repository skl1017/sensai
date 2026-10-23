"""`AgentTool`: a whole `Agent` exposed as a tool (A2, wiki §8.8).

`ask_<name>(task)` runs the sub-agent on `task` and returns its final answer
as the observation: one agent calls another with no new concept in the core.
The sub-agent has its own LLM, persona, tools and (usually lighter) pipeline,
all from its own `agent.yaml`, assembled by `app.build_agent`.

- Context isolation: the sub-agent starts with an empty history, so `task`
  must carry everything it needs (the `description` tells the model so).
- State: the sub-agent run gets `{"depth": depth + 1, "parent_trace_id": ...}`,
  read from the `ContextVar`s of `core/runtime.py` since a tool never sees
  the `Context`.
- Guards, all raised as explicit errors (the ReAct node turns them into an
  observation): `max_depth` (nesting depth, 2 by default), `max_calls`
  (calls of this sub-agent per root request, nesting included, counted in
  `runtime.subagent_calls`), `timeout` (the sub-agent runs a whole ReAct
  loop, so it enforces its own, longer than the node's `tool_timeout`).
- UI: the sub-agent shares the parent's `UserInterface` through `SubAgentUI`,
  so its approvals reach the same user, prefixed with the sub-agent name; its
  events are forwarded tagged with `agent`, its tokens are muted (only its
  final answer matters, and it comes back as the observation).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from core import runtime
from core.agent import Agent
from core.tool import ITool
from core.ui import UserInterface

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_CALLS = 3
DEFAULT_TIMEOUT = 120.0
MAX_ANSWER_CHARS = 4000

TRUNCATED_NOTE = "\n[Note: the sub-agent hit its iteration limit; this answer may be incomplete.]"

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class SubAgentUI:
    """`UserInterface` handed to a sub-agent: the parent's UI, scoped to `name`.

    `ask` reaches the parent's user, prefixed with `[name]`; `on_event` is
    forwarded with `agent` added to the data (`"outer/inner"` when nested);
    `on_token` is muted.
    """

    def __init__(self, parent: UserInterface, name: str) -> None:
        self.parent = parent
        self.name = name

    def on_token(self, text: str) -> None:
        return None

    def on_event(self, kind: str, data: dict) -> Any:
        inner = data.get("agent")
        agent = f"{self.name}/{inner}" if inner else self.name
        return self.parent.on_event(kind, {**data, "agent": agent})

    async def ask(
        self, question: str, kind: str = "confirm", options: list[str] | None = None
    ) -> str:
        return await self.parent.ask(f"[{self.name}] {question}", kind=kind, options=options)


class AgentTool(ITool):
    enforces_own_timeout = True

    def __init__(
        self,
        name: str,
        description: str,
        agent: Agent,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_calls: int = DEFAULT_MAX_CALLS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not _NAME.match(name):
            raise ValueError(f"agent name must be snake_case, got {name!r}")
        self.agent_name = name
        self.name = f"ask_{name}"
        self.description = (
            f"{description.strip()} The sub-agent does not see this conversation: "
            "put everything it needs in `task`."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A self-contained task, with all the context it needs.",
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        }
        self.agent = agent
        self.max_depth = max_depth
        self.max_calls = max_calls
        self.timeout = timeout

    async def run(self, task: str) -> str:
        depth = runtime.depth.get()
        if depth >= self.max_depth:
            raise RuntimeError(
                f"maximum sub-agent depth ({self.max_depth}) reached: "
                f"{self.name} cannot be called from here, answer with what you have"
            )

        calls = runtime.subagent_calls.get()
        if calls is not None:
            if calls.count(self.agent_name) >= self.max_calls:
                raise RuntimeError(
                    f"{self.name} was already called {self.max_calls} times for this request, "
                    "answer with what you have"
                )
            calls.append(self.agent_name)

        state = {"depth": depth + 1, "parent_trace_id": runtime.trace_id.get()}
        deadline = asyncio.timeout(self.timeout)
        try:
            async with deadline:
                ctx = await self.agent.run(task, history=[], state=state)
        except TimeoutError:
            if not deadline.expired():
                raise
            raise RuntimeError(
                f"sub-agent {self.agent_name!r} did not answer within {self.timeout}s"
            ) from None

        if ctx.response is None:
            raise RuntimeError(f"sub-agent {self.agent_name!r} returned no answer")
        answer = ctx.response[:MAX_ANSWER_CHARS]
        if ctx.state.get("react_truncated"):
            answer += TRUNCATED_NOTE
        return answer
