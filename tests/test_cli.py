"""Unit tests for `ui/cli.py`: the terminal chatbot loop."""

from __future__ import annotations

import io

from app import build_agent
from core.agent import Agent
from core.llm import ILLM, LLMError
from core.node import Next, PipelineNode
from core.types import Context, Message, ToolCall
from tests.fakes import FakeLLM
from ui.cli import CliUI, run_cli


def scripted_read_line(lines: list[str]):
    """Return an async `read_line` that yields `lines` in order, then raises EOFError."""
    remaining = list(lines)

    async def read_line(prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read_line


class _ShortCircuitNode(PipelineNode):
    """Tiny node that sets `ctx.response` without calling `next` or the LLM."""

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.response = "short-circuited answer"
        return ctx


class _RaisingLLM(ILLM):
    model = "raising"
    max_context_tokens = 8192

    def count_tokens(self, messages: list[Message]) -> int:
        return 0

    async def chat(self, messages, tools=None, schema=None):
        raise LLMError("boom")
        yield  # pragma: no cover - makes this an async generator


class _RaisingNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        async for _ in ctx.llm.chat(ctx.messages):
            pass
        return ctx  # pragma: no cover - chat() always raises first


def make_agent(node: PipelineNode, llm: ILLM | None = None) -> Agent:
    return Agent(llm or FakeLLM([]), [], [node])


# --- commands ----------------------------------------------------------------


async def test_help_command_prints_help():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line(["/help", "/quit"]))
    assert "/help" in out.getvalue()
    assert "/quit" in out.getvalue()


async def test_quit_stops_before_consuming_further_input():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    read_line = scripted_read_line(["/quit", "this should never be read"])
    await run_cli(agent, ui, read_line)
    # if /quit didn't stop the loop, the next read would raise via exhaustion
    # or the short-circuit answer would show up in the output
    assert "short-circuited" not in out.getvalue()


async def test_exit_alias_also_quits():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line(["/exit"]))
    assert out.getvalue() == ""


async def test_eof_exits_cleanly():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line([]))
    assert out.getvalue() == "\n"


async def test_unknown_command_reports_error():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line(["/frobnicate", "/quit"]))
    assert "Unknown command" in out.getvalue()
    assert "/help" in out.getvalue()


async def test_empty_line_ignored():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line(["   ", "/quit"]))
    assert "short-circuited" not in out.getvalue()


# --- responses -----------------------------------------------------------------


async def test_response_printed_when_nothing_streamed():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_ShortCircuitNode())
    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]))
    assert "short-circuited answer" in out.getvalue()


async def test_llm_error_prints_error_and_continues():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_RaisingNode(), llm=_RaisingLLM())
    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]))
    assert "Error: boom" in out.getvalue()


# --- end-to-end with real ReActLoopNode + CalculatorTool ----------------------


async def test_end_to_end_turn_with_calculator(tmp_path):
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        """
llm:
  provider: ollama
  model: fake
pipeline:
  - react
tools:
  - calculator
""".strip()
        + "\n",
        encoding="utf-8",
    )

    out = io.StringIO()
    ui = CliUI(out)
    call = ToolCall(id="1", name="calculator", arguments={"expression": "6*7"})
    llm = FakeLLM([[call], "The answer is 42."])
    agent = await build_agent(str(cfg_path), ui, llm=llm)

    await run_cli(agent, ui, scripted_read_line(["what is 6*7?", "/quit"]))

    output = out.getvalue()
    assert output.count("The answer is 42.") == 1
    assert "[calculator(" in output
    assert "expression='6*7'" in output


async def test_history_carries_previous_turn(tmp_path):
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        """
llm:
  provider: ollama
  model: fake
pipeline:
  - persist
  - react
tools:
  - calculator
""".strip()
        + "\n",
        encoding="utf-8",
    )

    from storage.stores import InMemoryStores

    out = io.StringIO()
    ui = CliUI(out)
    llm = FakeLLM(["First answer.", "Second answer."])
    # Same `stores` passed to both: `chat_turn` reads history from it, and the
    # `PersistNode` wired into `agent` via `build_agent` must write into that
    # very instance for the second turn to see the first one.
    stores = InMemoryStores()
    agent = await build_agent(str(cfg_path), ui, llm=llm, stores=stores)

    await run_cli(
        agent,
        ui,
        scripted_read_line(["first question", "second question", "/quit"]),
        stores=stores,
    )

    second_call_messages = llm.calls[-1]["messages"]
    contents = [(m.role, m.content) for m in second_call_messages]
    assert ("user", "first question") in contents
    assert ("assistant", "First answer.") in contents


def test_tool_call_event_with_non_dict_args_does_not_crash():
    out = io.StringIO()
    ui = CliUI(out)
    ui.on_event("tool_call", {"step": 1, "name": "calculator", "args": "6*7"})
    assert "[calculator('6*7')]" in out.getvalue()


async def test_cli_uses_one_session_in_stores(tmp_path):
    from core.agent import Agent
    from nodes.persist_node import PersistNode
    from storage.stores import InMemoryStores

    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    # `chat_turn` no longer appends on its own: a `PersistNode` in front of
    # the short-circuit node is what makes the turn durable.
    agent = Agent(FakeLLM([]), [], [PersistNode(stores.conversation), _ShortCircuitNode()])
    await run_cli(agent, ui, scripted_read_line(["a", "b", "/quit"]), stores, "sess")
    head = stores.conversation.head("sess")
    contents = [m.content for m in stores.conversation.branch("sess", head)]
    assert contents == ["a", "short-circuited answer", "b", "short-circuited answer"]


async def test_cli_ask_yes_no_and_eof(monkeypatch):
    answers = iter(["yes", "Y", "nope"])
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    ui = CliUI(io.StringIO())
    assert [await ui.ask("ok?") for _ in range(3)] == ["yes", "yes", "no"]

    def eof():
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert await ui.ask("ok?") == "no"
