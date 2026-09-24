"""`ApprovedTool`: human-in-the-loop wrapper around any `ITool` (A3, wiki §8.9).

The wrapper carries the inner tool's name, description and parameters, so the
model sees no difference. The user is asked *before* every run; anything but
`"yes"` (including no UI answer at all) is a refusal, returned as an
observation rather than raised, so it shows up in the ReAct trace like any
other tool result. `enforces_own_timeout` is `True` so a slow human does not
trip the node's `tool_timeout`: only the inner run, started after approval,
is bounded by `inner_timeout`.
"""

from __future__ import annotations

import asyncio
import json

from core.tool import ITool
from core.ui import UserInterface

REFUSAL = "Action refusée par l'utilisateur. Propose une alternative ou demande-lui."


class ApprovedTool(ITool):
    enforces_own_timeout = True

    def __init__(self, inner: ITool, ui: UserInterface, inner_timeout: float = 30.0) -> None:
        self.inner, self.ui, self.inner_timeout = inner, ui, inner_timeout
        self.name, self.description, self.parameters = (
            inner.name,
            inner.description,
            inner.parameters,
        )

    async def run(self, **kwargs) -> str:
        answer = await self.ui.ask(self._plan(kwargs), kind="confirm")
        if answer != "yes":
            return REFUSAL
        return await asyncio.wait_for(self.inner.run(**kwargs), self.inner_timeout)

    def _plan(self, kwargs: dict) -> str:
        args = json.dumps(kwargs, ensure_ascii=False)
        return f"L'agent veut exécuter {self.name} avec {args}. Autoriser ?"
