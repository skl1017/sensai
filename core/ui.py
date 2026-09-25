"""User interface contract shared by the pipeline and the tools it drives.

`UserInterface` is what a concrete UI (CLI, Web, headless later) provides to
the rest of the system: a way to stream tokens as they are produced, and a
way to be told about structured events happening inside a node (a thought, a
tool call, a final answer). It lives in `core/` — not `ui/` — because
tools like `ApprovedTool`/`AskUserTool` and `app.py` need the contract
without depending on any concrete UI implementation, and `core/` is the one
package every layer is allowed to depend on.

`ReActLoopNode.from_config` wires `on_token`/`on_event` from a `UserInterface`
found on `deps.ui` (see `nodes/react_loop.py`); a node itself only needs the
two plain callables, not the full `UserInterface`, so it stays decoupled from
concrete UIs.

`ask` is the human-in-the-loop entry point used by `ApprovedTool` (`kind="confirm"`,
answer `"yes"` or `"no"`) and, later, `AskUserTool` (other kinds). A UI with no
user behind it must answer `"no"`: deny by default.

Event kinds
-----------
Emitted via `on_event(kind, data)` by `ReActLoopNode`:
- `"thought"`: `{step, text}` — text produced before a tool call.
- `"tool_call"`: `{step, name, args}` — about to run a tool.
- `"observation"`: `{step, name, observation}` — a tool call's result.
- `"final"`: `{text, truncated}` — the final answer, `truncated` is `True`
  when the iteration budget was exhausted before it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

OnToken = Callable[[str], "None | Awaitable[None]"]
OnEvent = Callable[[str, dict], "None | Awaitable[None]"]


@runtime_checkable
class UserInterface(Protocol):
    def on_token(self, text: str) -> None | Awaitable[None]:
        """Called with each streamed text fragment as the model produces it."""
        ...

    def on_event(self, kind: str, data: dict) -> None | Awaitable[None]:
        """Called with a structured event (see module docstring for kinds)."""
        ...

    async def ask(self, question: str, kind: str = "confirm") -> str:
        """Ask the user; for `kind="confirm"` return `"yes"` or `"no"` (anything else = refusal)."""
        ...
