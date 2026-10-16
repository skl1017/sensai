"""Sessions, branches and interruption in `ui/cli.py` (X2, wiki §9.2/§9.3)."""

from __future__ import annotations

import asyncio
import io

from app import build_agent
from core.llm import ILLM
from core.types import LLMChunk, Message
from nodes.persist_node import PersistNode
from nodes.react_loop import ReActLoopNode
from storage.conversation import ConversationStore
from storage.stores import InMemoryStores
from tests.fakes import FakeLLM
from tests.test_cli import scripted_read_line
from ui.cli import CliUI, run_cli

# --- restart / resume ----------------------------------------------------------


async def test_restart_resumes_conversation_from_disk(tmp_path):
    sessions_dir = tmp_path / "sessions"
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"""
llm:
  provider: ollama
  model: fake
pipeline:
  - persist: {{ dir: {sessions_dir} }}
  - react
tools:
  - calculator
""".strip()
        + "\n",
        encoding="utf-8",
    )
    sid = "resumable-session"

    # First "process": one turn, persisted to disk.
    out1 = io.StringIO()
    ui1 = CliUI(out1)
    stores1 = InMemoryStores(conversation=ConversationStore(root=sessions_dir))
    agent1 = await build_agent(str(cfg_path), ui1, llm=FakeLLM(["First answer."]), stores=stores1)
    await run_cli(agent1, ui1, scripted_read_line(["first question", "/quit"]), stores1, sid)

    # Second "process": a brand new ConversationStore on the same directory.
    out2 = io.StringIO()
    ui2 = CliUI(out2)
    stores2 = InMemoryStores(conversation=ConversationStore(root=sessions_dir))
    llm2 = FakeLLM(["Second answer."])
    agent2 = await build_agent(str(cfg_path), ui2, llm=llm2, stores=stores2)
    await run_cli(agent2, ui2, scripted_read_line(["second question", "/quit"]), stores2, sid)

    contents = [(m.role, m.content) for m in llm2.calls[-1]["messages"]]
    assert ("user", "first question") in contents
    assert ("assistant", "First answer.") in contents


# --- /sessions, /new ------------------------------------------------------------


async def test_sessions_command_lists_known_sessions():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores)

    await run_cli(agent, ui, scripted_read_line(["hello", "/sessions", "/quit"]), stores, "sess-a")

    assert "sess-a" in out.getvalue()


async def test_sessions_command_empty():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores)

    await run_cli(agent, ui, scripted_read_line(["/sessions", "/quit"]), stores, "sess-a")

    assert "(no sessions yet)" in out.getvalue()


async def test_new_command_starts_a_fresh_session():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores)

    await run_cli(
        agent,
        ui,
        scripted_read_line(["hello", "/new", "hello again", "/quit"]),
        stores,
        "sess-a",
    )

    assert stores.conversation.head("sess-a") is not None
    sessions = stores.conversation.list_sessions()
    assert len(sessions) == 2
    assert "New session:" in out.getvalue()


# --- /edit, /tree, /goto ---------------------------------------------------------


async def test_edit_forks_sibling_branch_original_stays_in_tree():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    llm = FakeLLM(["first reply", "edited reply"])
    agent = await build_agent_stub(stores, llm=llm)
    sid = "sess"

    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]), stores, sid)
    roots = stores.conversation.children(sid, None)
    assert len(roots) == 1
    original_user_id = roots[0]["id"]

    await run_cli(
        agent,
        ui,
        scripted_read_line([f"/edit {original_user_id[:6]} hello v2", "/quit"]),
        stores,
        sid,
    )

    roots_after = stores.conversation.children(sid, None)
    assert len(roots_after) == 2  # the original root is untouched, plus the new fork
    contents = {node["content"] for node in roots_after}
    assert contents == {"hello", "hello v2"}

    tree_out = io.StringIO()
    tree_ui = CliUI(tree_out)
    await run_cli(agent, tree_ui, scripted_read_line(["/tree", "/quit"]), stores, sid)
    tree_text = tree_out.getvalue()
    assert "hello" in tree_text
    assert "hello v2" in tree_text
    assert "edited reply" in tree_text


async def test_edit_rejects_non_user_node():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores, llm=FakeLLM(["first reply"]))
    sid = "sess"

    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]), stores, sid)
    head = stores.conversation.head(sid)  # the assistant node

    err_out = io.StringIO()
    err_ui = CliUI(err_out)
    await run_cli(
        agent, err_ui, scripted_read_line([f"/edit {head[:6]} oops", "/quit"]), stores, sid
    )
    assert "must be a user message" in err_out.getvalue()


async def test_goto_moves_head_and_next_turn_continues_from_there():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    llm = FakeLLM(["first reply", "edited reply", "continued reply"])
    agent = await build_agent_stub(stores, llm=llm)
    sid = "sess"

    await run_cli(agent, ui, scripted_read_line(["hello", "/quit"]), stores, sid)
    roots = stores.conversation.children(sid, None)
    original_user_id = roots[0]["id"]
    original_assistant_id = stores.conversation.head(sid)

    await run_cli(
        agent,
        ui,
        scripted_read_line([f"/edit {original_user_id[:6]} hello v2", "/quit"]),
        stores,
        sid,
    )
    assert stores.conversation.head(sid) != original_assistant_id

    await run_cli(
        agent,
        ui,
        scripted_read_line([f"/goto {original_assistant_id[:6]}", "/quit"]),
        stores,
        sid,
    )
    assert stores.conversation.head(sid) == original_assistant_id

    await run_cli(agent, ui, scripted_read_line(["continue please", "/quit"]), stores, sid)
    new_head = stores.conversation.head(sid)
    path = stores.conversation.path(sid, new_head)
    assert [n["content"] for n in path] == [
        "hello",
        "first reply",
        "continue please",
        "continued reply",
    ]


async def test_goto_unknown_prefix_reports_error_without_crashing():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores, llm=FakeLLM(["reply"]))
    sid = "sess"

    await run_cli(agent, ui, scripted_read_line(["hi", "/quit"]), stores, sid)

    err_out = io.StringIO()
    err_ui = CliUI(err_out)
    await run_cli(agent, err_ui, scripted_read_line(["/goto zzzzzz", "/quit"]), stores, sid)
    assert "No node id starts with" in err_out.getvalue()


# --- /profile --------------------------------------------------------------------


async def test_profile_set_is_injected_into_next_turn():
    cfg_text = """
llm:
  provider: ollama
  model: fake
pipeline:
  - persist
  - context: { sections: [profile] }
  - react
tools:
  - calculator
"""
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    llm = FakeLLM(["ok"])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "agent.yaml"
        cfg_path.write_text(cfg_text.strip() + "\n", encoding="utf-8")
        agent = await build_agent(str(cfg_path), ui, llm=llm, stores=stores)

        await run_cli(
            agent,
            ui,
            scripted_read_line(["/profile set lang fr", "hello", "/quit"]),
            stores,
            "sess",
        )

    messages = llm.calls[-1]["messages"]
    system_messages = [m.content for m in messages if m.role == "system"]
    assert any("lang: fr" in text for text in system_messages)


async def test_profile_show_unset_instructions_clear():
    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    agent = await build_agent_stub(stores)

    await run_cli(
        agent,
        ui,
        scripted_read_line(
            [
                "/profile",
                "/profile set lang fr",
                "/profile instructions Be terse.",
                "/profile",
                "/profile unset lang",
                "/profile clear",
                "/profile",
                "/quit",
            ]
        ),
        stores,
        "sess",
    )
    text = out.getvalue()
    assert "(empty profile)" in text
    assert "lang: fr" in text
    assert "Be terse." in text


# --- interruption ----------------------------------------------------------------


class _SlowThenFastLLM(ILLM):
    """Streams one token then hangs forever on the first call; answers normally after."""

    model = "slow"
    max_context_tokens = 8192

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def count_tokens(self, messages: list[Message]) -> int:
        return 0

    async def chat(self, messages, tools=None, schema=None):
        self.calls += 1
        if self.calls == 1:
            try:
                yield LLMChunk(text="partial ")
                await asyncio.Event().wait()  # never resolves; only cancellation ends this
                yield LLMChunk(done=True, prompt_tokens=0, completion_tokens=0)  # unreachable
            finally:
                self.closed = True
        else:
            yield LLMChunk(text="done")
            yield LLMChunk(done=True, prompt_tokens=0, completion_tokens=0)


class _CancelOnFirstTokenUI(CliUI):
    """Test double: cancels the in-flight task the first time a token streams in."""

    def __init__(self, out) -> None:
        super().__init__(out)
        self.task: asyncio.Task | None = None
        self.armed = True

    def on_token(self, text: str) -> None:
        super().on_token(text)
        if self.armed and self.task is not None:
            self.armed = False
            task, self.task = self.task, None
            task.cancel()


def _install_interrupt_via_ui(ui: _CancelOnFirstTokenUI):
    def install(task: asyncio.Task):
        ui.task = task

        def remove() -> None:
            ui.task = None

        return remove

    return install


async def test_interrupt_saves_partial_and_continues_loop():
    out = io.StringIO()
    ui = _CancelOnFirstTokenUI(out)
    stores = InMemoryStores()
    llm = _SlowThenFastLLM()
    from core.agent import Agent

    agent = Agent(
        llm,
        [],
        [
            PersistNode(stores.conversation),
            ReActLoopNode(on_token=ui.on_token, on_event=ui.on_event),
        ],
    )
    sid = "sess"

    await run_cli(
        agent,
        ui,
        scripted_read_line(["hello", "next input", "/quit"]),
        stores,
        sid,
        install_interrupt=_install_interrupt_via_ui(ui),
    )

    assert "[interrupted]" in out.getvalue()
    assert llm.closed is True
    assert llm.calls == 2

    head = stores.conversation.head(sid)
    path = stores.conversation.path(sid, head)
    got = [(n["role"], n["content"], n.get("interrupted", False)) for n in path]
    assert got == [
        ("user", "hello", False),
        ("assistant", "partial ", True),
        ("user", "next input", False),
        ("assistant", "done", False),
    ]


# --- helpers ----------------------------------------------------------------------


async def build_agent_stub(stores, llm=None):
    """A minimal real `Agent` (persist + react + calculator) sharing `stores`."""
    import tempfile
    from pathlib import Path

    cfg_text = """
llm:
  provider: ollama
  model: fake
pipeline:
  - persist
  - react
tools:
  - calculator
"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "agent.yaml"
        cfg_path.write_text(cfg_text.strip() + "\n", encoding="utf-8")
        # A generously long script: these tests care about session/branch
        # bookkeeping, not particular LLM answers, so any fixed reply will do.
        return await build_agent(
            str(cfg_path), CliUI(io.StringIO()), llm=llm or FakeLLM(["ok"] * 10), stores=stores
        )


# --- /persona ---------------------------------------------------------------------


async def test_persona_command_switches_persona_and_keeps_history():
    from core.agent import Agent
    from core.node import PipelineNode

    class Recorder(PipelineNode):
        def __init__(self):
            self.seen: list[tuple[str | None, list[str]]] = []

        async def handle(self, ctx, next):
            self.seen.append((ctx.state.get("persona"), [m.content for m in ctx.messages]))
            ctx.response = f"re: {ctx.user_input}"
            return ctx

    out = io.StringIO()
    ui = CliUI(out)
    stores = InMemoryStores()
    recorder = Recorder()
    agent = Agent(FakeLLM([]), [], [PersistNode(stores.conversation), recorder])

    lines = ["one", "/persona tutor", "/persona", "two", "/quit"]
    await run_cli(agent, ui, scripted_read_line(lines), stores, "s")

    assert recorder.seen[0][0] is None
    assert recorder.seen[1] == ("tutor", ["one", "re: one"])
    assert "Persona: tutor" in out.getvalue()
