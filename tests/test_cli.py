"""Unit tests for `ui/cli.py`: the terminal chatbot loop."""

from __future__ import annotations

import asyncio
import io

from app import build_agent
from core.agent import Agent
from core.llm import ILLM, LLMError
from core.node import Next, PipelineNode
from core.types import Context, Message, ToolCall
from storage.stores import InMemoryStores
from tests.fakes import FakeLLM
from ui.cli import CliUI, _format_citations, run_cli


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

    @property
    def options(self) -> dict:
        return {}

    def update_options(self, **kwargs) -> None:
        pass

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


class _CitingNode(PipelineNode):
    """Short-circuits with an answer and the citations a RAG turn would record."""

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.response = "Cinq semaines [1] [4]."
        ctx.state["citations"] = {
            "cited": [
                {"index": 1, "source_id": 7, "source_name": "rh.pdf", "chunk_index": 2},
            ],
            "invalid": [4],
        }
        return ctx


async def test_cited_sources_listed_under_answer_and_invalid_flagged():
    out = io.StringIO()
    agent = make_agent(_CitingNode())

    await run_cli(agent, CliUI(out), scripted_read_line(["congés ?", "/quit"]))

    text = out.getvalue()
    assert "Cinq semaines [1] [4].\nSources:\n  [1] rh.pdf (passage 3)\n" in text
    assert "Warning: [4] cite(s) no retrieved document." in text


def test_format_citations_falls_back_to_source_id():
    citations = {
        "cited": [{"index": 2, "source_id": 5, "source_name": None, "chunk_index": None}],
        "invalid": [],
    }

    assert _format_citations(citations) == "Sources:\n  [2] document 5\n"


def test_format_citations_empty_without_citations():
    assert _format_citations(None) == ""
    assert _format_citations({"cited": [], "invalid": []}) == ""


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


def test_tool_call_event_for_ask_user_is_not_logged():
    out = io.StringIO()
    ui = CliUI(out)
    ui.on_event("tool_call", {"step": 1, "name": "ask_user", "args": {"question": "Prénom ?"}})
    assert out.getvalue() == ""


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


async def test_cli_ask_text_and_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "  hello world  ")
    out = io.StringIO()
    ui = CliUI(out)
    assert await ui.ask("what's up?", kind="text") == "hello world"
    assert "what's up?" in out.getvalue()

    def eof():
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert await ui.ask("what's up?", kind="text") == ""


async def test_cli_ask_choice_valid_invalid_and_eof(monkeypatch):
    answers = iter(["nope", "5", "2"])
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    out = io.StringIO()
    ui = CliUI(out)
    assert await ui.ask("pick one", kind="choice", options=["red", "green", "blue"]) == "green"
    text = out.getvalue()
    assert "1) red" in text and "2) green" in text and "3) blue" in text
    assert "Invalid choice" in text

    def eof():
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert await ui.ask("pick one", kind="choice", options=["red"]) == ""


async def test_cli_ask_choice_without_options_raises():
    ui = CliUI(io.StringIO())
    try:
        await ui.ask("pick one", kind="choice")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


async def test_cli_ask_unknown_kind_raises():
    ui = CliUI(io.StringIO())
    try:
        await ui.ask("??", kind="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


async def test_tools_command_lists_loaded_tools():
    from tests.fakes import FakeTool

    agent = Agent(FakeLLM([]), [FakeTool()], [_ShortCircuitNode()])
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/tools", "/quit"]))
    assert f"  {FakeTool().name} — " in out.getvalue()


# --- /llm ----------------------------------------------------------------------


async def test_llm_show_prints_current_options():
    agent = make_agent(_ShortCircuitNode())
    agent.llm.update_options(temperature=0.3)
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/llm", "/quit"]))
    assert "temperature: 0.3" in out.getvalue()


async def test_llm_show_with_no_options_set():
    agent = make_agent(_ShortCircuitNode())
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/llm show", "/quit"]))
    assert "(no options set)" in out.getvalue()


async def test_llm_set_updates_the_live_llm_and_confirms():
    agent = make_agent(_ShortCircuitNode())
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/llm set temperature 0.2", "/quit"]))
    assert "Set temperature = 0.2" in out.getvalue()
    assert agent.llm.options["temperature"] == 0.2  # parsed to a real float, not "0.2"


async def test_llm_set_parses_bool_and_int_values():
    agent = make_agent(_ShortCircuitNode())
    out = io.StringIO()
    await run_cli(
        agent,
        CliUI(out=out),
        scripted_read_line(["/llm set num_predict 128", "/llm set mirostat true", "/quit"]),
    )
    assert agent.llm.options["num_predict"] == 128
    assert agent.llm.options["mirostat"] is True


async def test_llm_set_without_enough_arguments_reports_usage():
    agent = make_agent(_ShortCircuitNode())
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/llm set temperature", "/quit"]))
    assert "Usage: /llm set <key> <value>" in out.getvalue()
    assert agent.llm.options == {}


async def test_llm_unknown_subcommand_reports_error():
    agent = make_agent(_ShortCircuitNode())
    out = io.StringIO()
    await run_cli(agent, CliUI(out=out), scripted_read_line(["/llm wat", "/quit"]))
    assert "Unknown /llm subcommand" in out.getvalue()


async def test_llm_set_changes_take_effect_immediately_on_the_next_turn():
    """The same object the ReAct loop calls `chat()` on is mutated in place —
    no rebuild, no restart, the very next turn sees the new value."""

    class _RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__(["ok"])
            self.seen_options: list[dict] = []

        async def chat(self, messages, tools=None, schema=None):
            self.seen_options.append(self.options)
            async for chunk in super().chat(messages, tools, schema):
                yield chunk

    from nodes.react_loop import ReActLoopNode

    llm = _RecordingLLM()
    agent = Agent(llm, [], [ReActLoopNode()])
    out = io.StringIO()
    await run_cli(
        agent,
        CliUI(out=out),
        scripted_read_line(["/llm set temperature 0.5", "hello", "/quit"]),
    )
    assert llm.seen_options == [{"temperature": 0.5}]


class _StreamingModifiedNode(PipelineNode):
    """Streams a raw answer to the UI, then sets a filtered `ctx.response`."""

    def __init__(self, ui: CliUI, modified: bool) -> None:
        self.ui = ui
        self.modified = modified

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.ui.on_token("raw answer")
        ctx.response = "filtered answer" if self.modified else "raw answer"
        ctx.state["output_modified"] = self.modified
        return ctx


async def test_streamed_answer_reprinted_when_output_modified():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_StreamingModifiedNode(ui, modified=True))
    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]))
    text = out.getvalue()
    assert "[réponse filtrée]" in text
    assert (
        text.index("raw answer") < text.index("[réponse filtrée]") < text.index("filtered answer")
    )


async def test_streamed_answer_not_reprinted_when_unmodified():
    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_StreamingModifiedNode(ui, modified=False))
    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]))
    assert "[réponse filtrée]" not in out.getvalue()
    assert out.getvalue().count("raw answer") == 1


async def test_interrupted_partial_is_pii_masked_before_saving():
    from ui.cli import _run_turn

    class _StreamThenWait(PipelineNode):
        def __init__(self, ui: CliUI) -> None:
            self.ui = ui

        async def handle(self, ctx: Context, next: Next) -> Context:
            self.ui.on_token("Écris à jean.dupont@example.com puis")
            await asyncio.sleep(10)
            return ctx  # pragma: no cover

    out = io.StringIO()
    ui = CliUI(out)
    agent = make_agent(_StreamThenWait(ui))
    stores = InMemoryStores()

    def install(task):
        asyncio.get_running_loop().call_later(0.05, task.cancel)
        return lambda: None

    await _run_turn(agent, ui, stores, "s1", "hi", None, install)
    saved = repr(stores.conversation.branch("s1", stores.conversation.head("s1")))
    assert "[EMAIL]" in saved
    assert "jean.dupont@example.com" not in saved
