"""`AskUserTool`: lets the model ask the user for missing information (X4, wiki §8.6).

The model should never guess when a piece of information it needs is missing
from the conversation — it should call `ask_user` instead (see the tool's
`description`, read by the model). The tool is a thin wrapper around
`UserInterface.ask` (`core/ui.py`): the `question` and, for `kind="choice"`,
the `options` come from the model; the answer comes back as plain text,
`"yes"`/`"no"` for `kind="confirm"`.

Unlike `ApprovedTool`, this tool has no side effect and is never wrapped in
approval. It sets `enforces_own_timeout = True` because the optional
`timeout` is *this* tool's concern, not `ReActLoopNode.tool_timeout`'s: when
it elapses, `run` returns `NO_ANSWER` as a normal observation (not an
exception raised up), like `ApprovedTool.REFUSAL`, so the model can react to
it (ask differently, fall back, or proceed with a stated assumption) instead
of the turn failing.
"""

from __future__ import annotations

import asyncio

from core.tool import ITool
from core.ui import UserInterface

NO_ANSWER = "Aucune réponse de l'utilisateur."

_KINDS = ("text", "choice", "confirm")


class AskUserTool(ITool):
    name = "ask_user"
    description = (
        "Use this tool to ask the user a question whenever a piece of information you "
        "need (a date, a name, a choice, a confirmation, ...) is missing from the "
        "conversation and you cannot proceed without it. Do NOT ask the question in "
        "your own reply text and do NOT guess or invent the missing detail: call this "
        "tool instead, so the user's answer is captured. kind='confirm' for a yes/no "
        "question, kind='choice' with `options` for a multiple-choice question, "
        "kind='text' (default) for anything else."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask the user."},
            "kind": {
                "type": "string",
                "enum": list(_KINDS),
                "description": (
                    "'text' (default) for a free-form answer, 'choice' with `options` for "
                    "a multiple-choice answer, 'confirm' for a yes/no answer."
                ),
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The possible answers; required when kind='choice'.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    enforces_own_timeout = True

    def __init__(self, ui: UserInterface, timeout: float | None = None) -> None:
        self.ui = ui
        self.timeout = timeout

    async def run(self, question: str, kind: str = "text", options: list[str] | None = None) -> str:
        if kind not in _KINDS:
            raise ValueError(f"Unsupported kind {kind!r}; expected one of {_KINDS}")
        if kind == "choice" and not options:
            raise ValueError("kind='choice' requires a non-empty 'options' list")

        coro = self.ui.ask(question, kind=kind, options=options)
        if self.timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, self.timeout)
        except TimeoutError:
            return NO_ANSWER

    @classmethod
    def from_config(cls, params: dict, deps: object) -> AskUserTool:
        # `deps` is untyped here (not `Deps` from app.py): tools depend only
        # on core/, never on app.py, so the concrete type can't be imported.
        return cls(ui=deps.ui, timeout=params.get("timeout"))
