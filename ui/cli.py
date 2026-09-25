"""Terminal chatbot with resumable sessions and branches (B1, X2, wiki §9.2/§9.3).

`run_cli` is a plain input loop: read a line, feed it to the `Agent` via
`app.chat_turn`, stream the answer to stdout token by token via
`CliUI.on_token`. `ctx.response` is only printed directly when nothing was
streamed (a cache hit or a guardrail short-circuit) — this keeps the CLI
silent about the mechanism and just shows the text once.

Sessions are durable (`storage.conversation.ConversationStore`, wired by
`app.open_stores`): the current session id is held in this loop and can
change at runtime via `/resume`/`/new`. `/history` and `/tree` inspect the
branch tree, `/edit` forks an earlier user message into a sibling branch
(the original stays intact), and `/goto` moves the head for branch
navigation. `/profile` edits the static user profile reinjected into every
turn by `ContextBuilderNode`. `/persona <name>` switches the persona
(`config/personas/<name>.yaml`) for the following turns, history kept (A5).

Ctrl-C while a turn is in flight cancels *that turn* (X2), not the whole
process: `install_interrupt` wires a `SIGINT` handler to `task.cancel()` for
the turn's duration only, and the partial answer streamed so far is saved
with `interrupted=True` so `/history` shows it. `run_cli`'s own cancellation
(e.g. the process shutting down) is never swallowed — see `_run_turn`.

`/help` lists every command; any other `/x` is reported as unknown. EOF at
the prompt exits the loop cleanly.

`amain`/`main` are the process entry points: `python -m ui.cli` opens the
stores and builds the agent from `config/agent.yaml` (or `--config`), prints
the session id (`--session` to resume one), then runs the loop, closing the
LLM (and its embedder) on the way out.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TextIO

from app import ROOT, build_agent, chat_turn, open_stores
from core.agent import Agent
from core.llm import LLMError
from core.types import Message
from storage.stores import InMemoryStores, Stores

HELP_TEXT = """\
Available commands:
  /help                          Show this message.
  /quit                          Exit the chat (alias: /exit).
  /sessions                      List known sessions.
  /resume <id>                   Switch to an existing session.
  /new                           Start a new session.
  /history                       Show the current branch (root -> head).
  /tree                          Show the whole session tree (head marked *).
  /edit <node-id> <text>         Fork an earlier user message with new text.
  /goto <node-id>                Move the head to another node.
  /profile                       Show the current user profile.
  /profile set <key> <value>     Set a profile preference.
  /profile unset <key>           Remove a profile preference.
  /profile instructions <text>   Set the profile's custom instructions.
  /profile clear                 Clear the whole profile.
  /persona [<name>]              Show or switch the persona for next turns.
Node ids above accept any unique prefix of the full id.
"""

ReadLine = Callable[[str], Awaitable[str]]
InstallInterrupt = Callable[[asyncio.Task], Callable[[], None]]

_SHORT_ID_CHARS = 8
_TRUNCATE_CHARS = 60


class CliUI:
    """`UserInterface` implementation that streams tokens to a text stream."""

    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out
        self.streamed = False
        self.partial = ""

    def reset(self) -> None:
        """Call before each turn so `streamed`/`partial` reflect only that turn."""
        self.streamed = False
        self.partial = ""

    def on_token(self, text: str) -> None:
        self.streamed = True
        self.partial += text
        self.out.write(text)
        self.out.flush()

    async def ask(self, question: str, kind: str = "confirm") -> str:
        """Yes/no prompt on stdin; EOF or anything but yes/y is a refusal."""
        self.out.write(f"\n{question} [yes/no] ")
        self.out.flush()
        try:
            answer = await asyncio.to_thread(input)
        except EOFError:
            return "no"
        return "yes" if answer.strip().lower() in ("yes", "y") else "no"

    def on_event(self, kind: str, data: dict) -> None:
        if kind == "tool_call":
            raw = data["args"]
            if isinstance(raw, dict):
                args = ", ".join(f"{k}={v!r}" for k, v in raw.items())
            else:  # malformed arguments from the model: the node reports it as an observation
                args = repr(raw)
            self.out.write(f"\n[{data['name']}({args})]\n")
            self.out.flush()
        # "thought" is already streamed via on_token; "observation" and
        # "final" are left out on purpose to keep the CLI output clean.


async def _default_read_line(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _default_install_interrupt(task: asyncio.Task) -> Callable[[], None]:
    """Cancel `task` on Ctrl-C while installed; a no-op remover where unsupported.

    `loop.add_signal_handler` isn't implemented on every platform (notably
    Windows' default event loop): on `NotImplementedError` interruption is
    simply unavailable there, rather than crashing the CLI.
    """
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    except NotImplementedError:
        return lambda: None

    def remove() -> None:
        loop.remove_signal_handler(signal.SIGINT)

    return remove


def _short(node_id: str) -> str:
    return node_id[:_SHORT_ID_CHARS]


def _truncate(text: str, limit: int = _TRUNCATE_CHARS) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_node(node: dict) -> str:
    suffix = " [interrupted]" if node.get("interrupted") else ""
    return f"[{_short(node['id'])}] {node['role']}: {_truncate(node['content'])}{suffix}"


def _all_nodes(store, session_id: str) -> dict[str, dict]:
    """Every node of `session_id`, id -> node (BFS from the roots).

    Plain sync function, meant to be run through `asyncio.to_thread`:
    `ConversationStore` has no single "list all nodes" call, so this walks
    `children()` from the roots (`parent_id=None`) down.
    """
    nodes: dict[str, dict] = {}
    frontier: list[str | None] = [None]
    while frontier:
        parent = frontier.pop()
        for child in store.children(session_id, parent):
            nodes[child["id"]] = child
            frontier.append(child["id"])
    return nodes


def _resolve_prefix(nodes: dict[str, dict], prefix: str) -> str:
    """The one full node id starting with `prefix`; `KeyError` on 0 or >1 matches."""
    matches = [node_id for node_id in nodes if node_id.startswith(prefix)]
    if not matches:
        raise KeyError(f"No node id starts with {prefix!r}")
    if len(matches) > 1:
        raise KeyError(f"Ambiguous id {prefix!r}: matches {', '.join(sorted(matches))}")
    return matches[0]


def _render_tree(nodes: dict[str, dict], head_path: set[str]) -> list[str]:
    """One indented line per node, root(s) first; `head_path` members get a `*`."""
    children_of: dict[str | None, list[dict]] = {}
    for node in nodes.values():
        children_of.setdefault(node["parent_id"], []).append(node)
    for siblings in children_of.values():
        siblings.sort(key=lambda n: n["created_at"])

    lines: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for node in children_of.get(parent_id, []):
            marker = "*" if node["id"] in head_path else " "
            lines.append(f"{'  ' * depth}{marker} {_format_node(node)}")
            walk(node["id"], depth + 1)

    walk(None, 0)
    return lines


async def _run_turn(
    agent: Agent,
    ui: CliUI,
    stores: Stores,
    session_id: str,
    text: str,
    explicit_parent: str | None,
    install_interrupt: InstallInterrupt,
    persona: str | None = None,
) -> None:
    """Run one turn end to end, streaming to `ui`.

    `explicit_parent` overrides the session head (used by `/edit` to fork a
    sibling branch: `app.ROOT` or an older node id); `None` means "continue
    from the current head". The turn runs in its own `Task` so `Ctrl-C`
    (wired by `install_interrupt`) can cancel *it* without tearing down the
    input loop: on cancellation the partial answer streamed so far is saved
    as `interrupted=True` (`PersistNode` saved nothing, since the pipeline
    never returned normally). `run_cli`'s own cancellation is re-raised after
    cancelling the child task, never swallowed.
    """
    ui.reset()

    if explicit_parent is not None:
        parent_id = explicit_parent
    else:
        parent_id = await asyncio.to_thread(stores.conversation.head, session_id)

    task = asyncio.create_task(
        chat_turn(agent, stores, session_id, text, persona=persona, parent_id=parent_id)
    )
    remove_interrupt = install_interrupt(task)
    try:
        await asyncio.wait({task})
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        remove_interrupt()

    if task.cancelled():
        ui.out.write("\n[interrupted]\n")
        append_parent = None if parent_id == ROOT else parent_id
        await asyncio.to_thread(
            stores.conversation.append,
            session_id,
            append_parent,
            [Message("user", text), Message("assistant", ui.partial)],
            interrupted=True,
        )
        return

    try:
        ctx = task.result()
    except LLMError as exc:
        ui.out.write(f"Error: {exc}\n")
        return

    if not ui.streamed:
        ui.out.write(ctx.response or "")
    ui.out.write("\n")


async def _cmd_sessions(ui: CliUI, stores: Stores) -> None:
    sessions = await asyncio.to_thread(stores.conversation.list_sessions)
    if not sessions:
        ui.out.write("(no sessions yet)\n")
        return
    for session in sessions:
        title = session["title"] or "(empty)"
        ui.out.write(f"{session['id']}  {session['updated_at']}  {title}\n")


async def _cmd_resume(ui: CliUI, stores: Stores, current: str, arg: str) -> str:
    target = arg.strip()
    if not target:
        ui.out.write("Usage: /resume <id>\n")
        return current
    try:
        head = await asyncio.to_thread(stores.conversation.head, target)
    except ValueError as exc:
        ui.out.write(f"{exc}\n")
        return current

    ui.out.write(f"Resumed session {target}\n")
    if head is not None:
        nodes = await asyncio.to_thread(stores.conversation.path, target, head)
        for node in nodes[-6:]:
            ui.out.write(_format_node(node) + "\n")
    return target


async def _cmd_history(ui: CliUI, stores: Stores, session_id: str) -> None:
    head = await asyncio.to_thread(stores.conversation.head, session_id)
    if head is None:
        ui.out.write("(empty)\n")
        return
    nodes = await asyncio.to_thread(stores.conversation.path, session_id, head)
    for node in nodes:
        ui.out.write(_format_node(node) + "\n")


async def _cmd_tree(ui: CliUI, stores: Stores, session_id: str) -> None:
    nodes = await asyncio.to_thread(_all_nodes, stores.conversation, session_id)
    if not nodes:
        ui.out.write("(empty)\n")
        return

    head = await asyncio.to_thread(stores.conversation.head, session_id)
    head_path: set[str] = set()
    if head is not None:
        path_nodes = await asyncio.to_thread(stores.conversation.path, session_id, head)
        head_path = {node["id"] for node in path_nodes}

    for line in _render_tree(nodes, head_path):
        ui.out.write(line + "\n")


async def _cmd_goto(ui: CliUI, stores: Stores, session_id: str, arg: str) -> None:
    prefix = arg.strip()
    if not prefix:
        ui.out.write("Usage: /goto <node-id>\n")
        return

    nodes = await asyncio.to_thread(_all_nodes, stores.conversation, session_id)
    try:
        full_id = _resolve_prefix(nodes, prefix)
    except KeyError as exc:
        ui.out.write(f"{exc}\n")
        return

    await asyncio.to_thread(stores.conversation.set_head, session_id, full_id)
    ui.out.write(f"Head moved to [{_short(full_id)}]\n")


async def _cmd_edit(
    agent: Agent,
    ui: CliUI,
    stores: Stores,
    session_id: str,
    arg: str,
    install_interrupt: InstallInterrupt,
    persona: str | None = None,
) -> None:
    parts = arg.split(maxsplit=1)
    if len(parts) != 2:
        ui.out.write("Usage: /edit <node-id> <text>\n")
        return
    prefix, new_text = parts

    nodes = await asyncio.to_thread(_all_nodes, stores.conversation, session_id)
    try:
        full_id = _resolve_prefix(nodes, prefix)
    except KeyError as exc:
        ui.out.write(f"{exc}\n")
        return

    node = nodes[full_id]
    if node["role"] != "user":
        ui.out.write("Error: /edit target must be a user message\n")
        return

    fork_parent = ROOT if node["parent_id"] is None else node["parent_id"]
    await _run_turn(
        agent, ui, stores, session_id, new_text, fork_parent, install_interrupt, persona
    )


async def _cmd_profile(ui: CliUI, stores: Stores, arg: str) -> None:
    parts = arg.split(maxsplit=1)
    sub = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "":
        profile = await asyncio.to_thread(stores.profile.load)
        preferences = profile.get("preferences") or {}
        instructions = profile.get("instructions") or ""
        if not preferences and not instructions:
            ui.out.write("(empty profile)\n")
            return
        for key, value in preferences.items():
            ui.out.write(f"{key}: {value}\n")
        if instructions:
            ui.out.write(f"instructions: {instructions}\n")
    elif sub == "set":
        kv = rest.split(maxsplit=1)
        if len(kv) != 2:
            ui.out.write("Usage: /profile set <key> <value>\n")
            return
        key, value = kv
        await asyncio.to_thread(stores.profile.set_preference, key, value)
        ui.out.write(f"Set {key} = {value}\n")
    elif sub == "unset":
        key = rest.strip()
        if not key:
            ui.out.write("Usage: /profile unset <key>\n")
            return
        await asyncio.to_thread(stores.profile.unset_preference, key)
        ui.out.write(f"Unset {key}\n")
    elif sub == "instructions":
        if not rest.strip():
            ui.out.write("Usage: /profile instructions <text>\n")
            return
        await asyncio.to_thread(stores.profile.set_instructions, rest)
        ui.out.write("Instructions updated\n")
    elif sub == "clear":
        await asyncio.to_thread(stores.profile.clear)
        ui.out.write("Profile cleared\n")
    else:
        ui.out.write("Unknown /profile subcommand, type /help\n")


@dataclass
class _Shell:
    """What a slash-command handler may read or change."""

    agent: Agent
    ui: CliUI
    stores: Stores
    session_id: str
    install_interrupt: InstallInterrupt
    persona: str | None = None  # None: the config's root `persona`
    running: bool = True


CommandHandler = Callable[[_Shell, str], Awaitable[None]]


async def _do_quit(shell: _Shell, rest: str) -> None:
    shell.running = False


async def _do_help(shell: _Shell, rest: str) -> None:
    shell.ui.out.write(HELP_TEXT)


async def _do_sessions(shell: _Shell, rest: str) -> None:
    await _cmd_sessions(shell.ui, shell.stores)


async def _do_resume(shell: _Shell, rest: str) -> None:
    shell.session_id = await _cmd_resume(shell.ui, shell.stores, shell.session_id, rest)


async def _do_new(shell: _Shell, rest: str) -> None:
    shell.session_id = uuid.uuid4().hex
    shell.ui.out.write(f"New session: {shell.session_id}\n")


async def _do_history(shell: _Shell, rest: str) -> None:
    await _cmd_history(shell.ui, shell.stores, shell.session_id)


async def _do_tree(shell: _Shell, rest: str) -> None:
    await _cmd_tree(shell.ui, shell.stores, shell.session_id)


async def _do_goto(shell: _Shell, rest: str) -> None:
    await _cmd_goto(shell.ui, shell.stores, shell.session_id, rest)


async def _do_edit(shell: _Shell, rest: str) -> None:
    await _cmd_edit(
        shell.agent,
        shell.ui,
        shell.stores,
        shell.session_id,
        rest,
        shell.install_interrupt,
        shell.persona,
    )


async def _do_profile(shell: _Shell, rest: str) -> None:
    await _cmd_profile(shell.ui, shell.stores, rest)


async def _do_persona(shell: _Shell, rest: str) -> None:
    if not rest:
        shell.ui.out.write(f"Persona: {shell.persona or '(config default)'}\n")
        return
    shell.persona = rest.split()[0]
    shell.ui.out.write(f"Persona set to {shell.persona} for the next turns.\n")


COMMANDS: dict[str, CommandHandler] = {
    "/quit": _do_quit,
    "/exit": _do_quit,
    "/help": _do_help,
    "/sessions": _do_sessions,
    "/resume": _do_resume,
    "/new": _do_new,
    "/history": _do_history,
    "/tree": _do_tree,
    "/goto": _do_goto,
    "/edit": _do_edit,
    "/profile": _do_profile,
    "/persona": _do_persona,
}


async def run_cli(
    agent: Agent,
    ui: CliUI,
    read_line: ReadLine | None = None,
    stores: Stores | None = None,
    session_id: str | None = None,
    install_interrupt: InstallInterrupt | None = None,
) -> None:
    """Run the input loop until `/quit`, EOF, or Ctrl-C at the prompt.

    Each turn goes through `app.chat_turn`; the current session id lives in
    this loop's `session_id` variable and can change via `/resume`/`/new`
    (defaults: fresh in-memory stores, a new uuid4 hex session).
    `install_interrupt` defaults to a `SIGINT`-based `task.cancel()` (see
    `_default_install_interrupt`) and is injectable for tests.
    """
    read_line = read_line or _default_read_line
    stores = stores if stores is not None else InMemoryStores()
    session_id = session_id or uuid.uuid4().hex
    install_interrupt = install_interrupt or _default_install_interrupt
    shell = _Shell(agent, ui, stores, session_id, install_interrupt)

    while True:
        try:
            line = await read_line("> ")
        except (EOFError, KeyboardInterrupt):
            ui.out.write("\n")
            return

        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            command, _, rest = text.partition(" ")
            handler = COMMANDS.get(command)
            if handler is None:
                ui.out.write("Unknown command, type /help\n")
                continue
            await handler(shell, rest.strip())
            if not shell.running:
                return
            continue

        await _run_turn(
            agent, ui, stores, shell.session_id, text, None, install_interrupt, shell.persona
        )


async def amain(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sensai terminal chatbot.")
    parser.add_argument(
        "--config", default="config/agent.yaml", help="Path to the agent YAML config."
    )
    parser.add_argument(
        "--session", default=None, help="Resume this session id (default: start a new one)."
    )
    args = parser.parse_args(argv)

    ui = CliUI()
    stores = await open_stores(args.config)
    session_id = args.session or uuid.uuid4().hex
    agent = await build_agent(args.config, ui, stores=stores)

    print(f"Sensai ({agent.llm.model}) -- session {session_id} -- type /help for commands.")
    try:
        await run_cli(agent, ui, stores=stores, session_id=session_id)
    finally:
        aclose = getattr(agent.llm, "aclose", None)
        if aclose is not None:
            await aclose()
        embedder = getattr(agent.llm, "embedder", None)
        embedder_aclose = getattr(embedder, "aclose", None)
        if embedder_aclose is not None:
            await embedder_aclose()


def main(argv: list[str] | None = None) -> None:
    try:
        asyncio.run(amain(argv))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
