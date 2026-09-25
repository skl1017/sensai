"""Minimal terminal chatbot (B1, wiki §9.2).

`run_cli` is a plain input loop: read a line, feed it to the `Agent`, stream
the answer to stdout token by token via `CliUI.on_token`. `ctx.response` is
only printed directly when nothing was streamed (a cache hit or a guardrail
short-circuit, both to come with their own stories) — this keeps the CLI
silent about the mechanism and just shows the text once.

`/help` and `/quit` (alias `/exit`) are the only commands; any other `/x` is
reported as unknown. Ctrl-C or EOF at the prompt exits the loop cleanly.
Turn cancellation mid-answer (X2, e.g. Ctrl-C while the model is streaming)
is out of scope here: it belongs to the async-interruption story.

`amain`/`main` are the process entry points: `python -m ui.cli` builds the
agent from `config/agent.yaml` (or `--config`) via `app.build_agent`, then
runs the loop, closing the LLM (and its embedder) on the way out.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from collections.abc import Awaitable, Callable
from typing import TextIO

from app import build_agent, chat_turn
from core.agent import Agent
from core.llm import LLMError
from storage.stores import InMemoryStores, Stores

HELP_TEXT = """\
Available commands:
  /help    Show this message.
  /quit    Exit the chat (alias: /exit).
"""

ReadLine = Callable[[str], Awaitable[str]]


class CliUI:
    """`UserInterface` implementation that streams tokens to a text stream."""

    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out
        self.streamed = False

    def reset(self) -> None:
        """Call before each turn so `streamed` reflects only that turn."""
        self.streamed = False

    def on_token(self, text: str) -> None:
        self.streamed = True
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


async def run_cli(
    agent: Agent,
    ui: CliUI,
    read_line: ReadLine | None = None,
    stores: Stores | None = None,
    session_id: str | None = None,
) -> None:
    """Run the input loop until `/quit`, EOF, or Ctrl-C.

    Each turn goes through `app.chat_turn`, so history lives in `stores` under
    one session id generated once per run (defaults: fresh in-memory stores).
    """
    read_line = read_line or _default_read_line
    stores = stores if stores is not None else InMemoryStores()
    session_id = session_id or uuid.uuid4().hex

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
            command = text.split(maxsplit=1)[0]
            if command in ("/quit", "/exit"):
                return
            if command == "/help":
                ui.out.write(HELP_TEXT)
            else:
                ui.out.write("Unknown command, type /help\n")
            continue

        ui.reset()
        try:
            ctx = await chat_turn(agent, stores, session_id, text)
        except LLMError as exc:
            ui.out.write(f"Error: {exc}\n")
            continue

        if not ui.streamed:
            ui.out.write(ctx.response or "")
        ui.out.write("\n")


async def amain(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sensai terminal chatbot.")
    parser.add_argument(
        "--config", default="config/agent.yaml", help="Path to the agent YAML config."
    )
    args = parser.parse_args(argv)

    ui = CliUI()
    stores = InMemoryStores()
    agent = await build_agent(args.config, ui, stores=stores)

    print(f"Sensai ({agent.llm.model}) -- type /help for commands.")
    try:
        await run_cli(agent, ui, stores=stores)
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
