"""`HeadlessUI`: no-user `UserInterface` for unattended runs (A3, wiki §8.6/8.9).

Used wherever `build_agent` is handed no interactive UI — scheduled runs
(`config/schedules.yaml`), background jobs, anything with nobody at the
other end. `on_token`/`on_event` are no-ops (nobody is watching), and `ask`
always answers conservatively: deny by default, so *forgetting* to wire a
real `UserInterface` never turns into a tool silently running unattended, or
the agent hanging forever waiting for an answer that will never come.

Restriction: tools usable under `HeadlessUI`
----------------------------------------------
A pipeline run under `HeadlessUI` must only rely on tools that never need a
human answer to do their job:
- Any tool wrapped in `ApprovedTool` (see `agent.yaml`'s `approval` section)
  is always refused — `ask(kind="confirm")` returns `"no"`, i.e.
  `ApprovedTool.REFUSAL`, never runs the inner tool.
- Any `AskUserTool` call always gets `NO_USER` back instead of a real
  answer — the model sees it as a normal observation and must proceed
  without that information (or give up), it does not hang.

So a scheduled config should pick tools with no side effect requiring
approval and no dependency on `ask_user` for information the schedule can't
supply up front (via the prompt, profile, or artifact) — not tools that
assume someone is there to approve or answer.
"""

from __future__ import annotations

NO_USER = "Aucun utilisateur disponible."


class HeadlessUI:
    """`UserInterface` with nobody behind it (see module docstring)."""

    def on_token(self, text: str) -> None:
        pass

    def on_event(self, kind: str, data: dict) -> None:
        pass

    async def ask(
        self, question: str, kind: str = "confirm", options: list[str] | None = None
    ) -> str:
        return "no" if kind == "confirm" else NO_USER
