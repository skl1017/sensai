"""Unit tests for `app.py`: config parsing and agent assembly (wiki §9.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import (
    NODES,
    ROOT,
    TOOLS,
    Deps,
    build_agent,
    build_llm,
    build_reranker,
    chat_turn,
    flatten,
    normalize,
)
from core.node import PipelineNode
from llm.cross_encoder import DEFAULT_MODEL, FastEmbedReranker
from llm.ollama import Ollama, OllamaEmbedder
from nodes.context_builder import ContextBuilderNode
from nodes.persist_node import PersistNode
from nodes.react_loop import ReActLoopNode
from nodes.reranker import LLMReranker
from storage.stores import InMemoryStores
from tests.fakes import FakeEmbedder, FakeLLM, FakeTool
from tools.agent_tool import AgentTool, SubAgentUI


class _NoopUI:
    def on_token(self, text: str) -> None:
        pass

    def on_event(self, kind: str, data: dict) -> None:
        pass

    async def ask(self, question: str, kind: str = "confirm") -> str:
        return "no"


# --- normalize -------------------------------------------------------------


def test_normalize_bare_string():
    assert normalize("calculator") == ("calculator", {})


def test_normalize_dict_with_params():
    assert normalize({"react": {"max_iterations": 3}}) == ("react", {"max_iterations": 3})


def test_normalize_dict_with_none_params():
    assert normalize({"react": None}) == ("react", {})


@pytest.mark.parametrize(
    "entry",
    [
        {"react": {}, "cache": {}},  # two keys
        {},  # zero keys
        42,
        ["react"],
    ],
)
def test_normalize_invalid_entry_raises(entry):
    with pytest.raises(ValueError):
        normalize(entry)


# --- build_llm ---------------------------------------------------------------


async def test_build_llm_returns_ollama_with_embedder():
    llm = build_llm(
        {
            "provider": "ollama",
            "model": "qwen2.5:3b",
            "embed_model": "nomic-embed-text",
            "host": "http://localhost:11434",
            "max_context_tokens": 8192,
        }
    )
    try:
        assert isinstance(llm, Ollama)
        assert llm.model == "qwen2.5:3b"
        assert isinstance(llm.embedder, OllamaEmbedder)
        assert llm.embedder.model == "nomic-embed-text"
        assert llm.max_context_tokens == 8192
    finally:
        await llm.embedder.aclose()
        await llm.aclose()


def test_build_llm_rejects_non_ollama_provider():
    with pytest.raises(ValueError):
        build_llm({"provider": "openai", "model": "gpt-4"})


def test_build_llm_rejects_missing_provider():
    with pytest.raises(ValueError):
        build_llm({"model": "qwen2.5:3b"})


# --- build_reranker ------------------------------------------------------------


def test_build_reranker_absent_section_returns_none():
    assert build_reranker(None, FakeLLM([])) is None


def test_build_reranker_defaults_to_fastembed_jina():
    reranker = build_reranker({}, FakeLLM([]))

    assert isinstance(reranker, FastEmbedReranker)
    assert reranker.model == DEFAULT_MODEL


def test_build_reranker_forwards_fastembed_params():
    reranker = build_reranker(
        {"provider": "fastembed", "model": "BAAI/bge-reranker-base", "cache_dir": "m"},
        FakeLLM([]),
    )

    assert reranker.model == "BAAI/bge-reranker-base"
    assert reranker.cache_dir == "m"


def test_build_reranker_llm_provider_uses_given_llm():
    llm = FakeLLM([])

    reranker = build_reranker({"provider": "llm"}, llm)

    assert isinstance(reranker, LLMReranker)
    assert reranker.llm is llm


def test_build_reranker_llm_provider_rejects_params():
    with pytest.raises(ValueError, match="model"):
        build_reranker({"provider": "llm", "model": "x"}, FakeLLM([]))


def test_build_reranker_rejects_unknown_provider():
    with pytest.raises(ValueError, match="cohere"):
        build_reranker({"provider": "cohere"}, FakeLLM([]))


# --- build_agent -------------------------------------------------------------


def _write_config(tmp_path, pipeline="react", tools="calculator"):
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"""
llm:
  provider: ollama
  model: fake
pipeline:
  - {pipeline}
tools:
  - {tools}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


async def test_build_agent_from_temp_config():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(Path(tmp))
        ui = _NoopUI()
        llm = FakeLLM(["hello"])
        agent = await build_agent(str(cfg_path), ui, llm=llm)

        assert "calculator" in agent.tools
        assert len(agent.pipeline) == 1
        node = agent.pipeline[0]
        assert isinstance(node, ReActLoopNode)
        assert node.on_token == ui.on_token
        assert node.on_event == ui.on_event


async def test_build_agent_unknown_node_raises():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(Path(tmp), pipeline="not_a_node")
        with pytest.raises(ValueError, match="Unknown node"):
            await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))


async def test_build_agent_unknown_tool_raises():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(Path(tmp), tools="not_a_tool")
        with pytest.raises(ValueError, match="Unknown tool"):
            await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))


@pytest.fixture
def no_mcp(monkeypatch):
    """The real config declares an `npx` MCP server: never spawn it in unit tests."""

    async def _no_servers(configs, stack, **_):
        return []

    monkeypatch.setattr("app.connect_servers", _no_servers)


@pytest.mark.usefixtures("no_mcp")
async def test_build_agent_real_config_with_fake_llm():
    # Explicit in-memory stores: the real config's `persist`/`context` entries
    # point at `sessions/`/`data/profile.yaml`, and this test must never write there.
    # The real config's `context` entry enables the `fewshot` section, which
    # requires an embedder: `FakeLLM` has none by default, so one is attached
    # here the same way `Ollama.embedder` would be (`build_agent` falls back
    # to `getattr(llm, "embedder", None)`).
    ui = _NoopUI()
    llm = FakeLLM(["hi"])
    llm.embedder = FakeEmbedder()
    agent = await build_agent("config/agent.yaml", ui, llm=llm, stores=InMemoryStores())

    assert "calculator" in agent.tools
    assert len(agent.pipeline) == 3
    assert isinstance(agent.pipeline[0], PersistNode)
    assert isinstance(agent.pipeline[1], ContextBuilderNode)
    assert isinstance(agent.pipeline[2], ReActLoopNode)


@pytest.mark.usefixtures("no_mcp")
async def test_build_agent_real_config_end_to_end_chat_turn():
    """One full turn against the real `config/agent.yaml`: persona, profile and
    fewshot sections, budget pass, react loop -- everything wired, nothing real."""
    ui = _NoopUI()
    llm = FakeLLM(["hi there"])
    llm.embedder = FakeEmbedder()
    stores = InMemoryStores()
    agent = await build_agent("config/agent.yaml", ui, llm=llm, stores=stores)

    ctx = await chat_turn(agent, stores, "s1", "hello")

    assert ctx.response == "hi there"
    assert ctx.state["prompt"]["version"] == "v2"


def test_deps_dataclass_fields():
    deps = Deps(llm=FakeLLM([]), ui=_NoopUI())
    assert deps.llm is not None
    assert deps.ui is not None


async def test_build_agent_flattens_tool_lists(tmp_path):
    """Custom multi-tool factory result (a list) is flattened into `agent.tools`."""
    from app import TOOLS

    def multi_tool_factory(params, deps):
        return [FakeTool(name="a"), FakeTool(name="b")]

    TOOLS["multi"] = multi_tool_factory
    try:
        cfg_path = _write_config(tmp_path, tools="multi")
        agent = await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))
        assert "a" in agent.tools
        assert "b" in agent.tools
    finally:
        del TOOLS["multi"]


# --- Deps / flatten / registries ---------------------------------------------


def test_deps_stores_defaults_to_none():
    assert Deps(llm=FakeLLM([]), ui=_NoopUI()).stores is None


def test_flatten_single_tool():
    a = FakeTool(name="a")
    assert flatten([a]) == [a]


def test_flatten_list_of_tools():
    a, b = FakeTool(name="a"), FakeTool(name="b")
    assert flatten([[a, b]]) == [a, b]


def test_flatten_mixed_preserves_order():
    a, b, c = FakeTool(name="a"), FakeTool(name="b"), FakeTool(name="c")
    assert flatten([a, [b, c]]) == [a, b, c]


def test_flatten_empty():
    assert flatten([]) == []


async def test_unknown_node_error_lists_available_names(tmp_path):
    cfg_path = _write_config(tmp_path, pipeline="nope")
    with pytest.raises(ValueError) as exc:
        await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))
    assert all(name in str(exc.value) for name in NODES)


async def test_unknown_tool_error_lists_available_names(tmp_path):
    cfg_path = _write_config(tmp_path, tools="nope")
    with pytest.raises(ValueError) as exc:
        await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))
    assert all(name in str(exc.value) for name in TOOLS)


class _DummyNode(PipelineNode):
    @classmethod
    def from_config(cls, params, deps):
        return cls()

    async def handle(self, ctx, next):
        return await next(ctx)


async def test_new_node_is_one_registry_entry_and_one_config_line(monkeypatch, tmp_path):
    monkeypatch.setitem(NODES, "dummy", _DummyNode.from_config)
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "llm:\n  provider: ollama\n  model: fake\n"
        "pipeline:\n  - dummy\n  - react\n"
        "tools:\n  - calculator\n",
        encoding="utf-8",
    )
    agent = await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))
    assert isinstance(agent.pipeline[0], _DummyNode)
    assert isinstance(agent.pipeline[1], ReActLoopNode)


# --- approval wiring, pipeline from config, chat_turn ------------------------


def _cfg(tmp_path, pipeline=("react",), approval=()):
    lines = ["llm:", "  provider: ollama", "  model: fake", "pipeline:"]
    lines += [f"  - {p}" for p in pipeline]
    lines += ["tools:", "  - fakes"]
    if approval:
        lines += ["approval:"] + [f'  - "{a}"' for a in approval]
    path = tmp_path / "agent.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


async def test_removing_react_line_shortens_pipeline(monkeypatch, tmp_path):
    monkeypatch.setitem(TOOLS, "fakes", lambda p, d: FakeTool())
    full = await build_agent(_cfg(tmp_path), _NoopUI(), llm=FakeLLM([]))
    empty = await build_agent(_cfg(tmp_path, pipeline=()), _NoopUI(), llm=FakeLLM([]))
    assert len(full.pipeline) - len(empty.pipeline) == 1


async def test_tools_matching_approval_are_wrapped(monkeypatch, tmp_path):
    from tools.approved_tool import ApprovedTool

    monkeypatch.setitem(
        TOOLS, "fakes", lambda p, d: [FakeTool(name="write_file"), FakeTool(name="read_file")]
    )
    agent = await build_agent(_cfg(tmp_path, approval=["write_*"]), _NoopUI(), llm=FakeLLM([]))
    assert isinstance(agent.tools["write_file"], ApprovedTool)
    assert not isinstance(agent.tools["read_file"], ApprovedTool)


async def test_no_approval_section_wraps_nothing(monkeypatch, tmp_path):
    from tools.approved_tool import ApprovedTool

    monkeypatch.setitem(TOOLS, "fakes", lambda p, d: FakeTool(name="write_file"))
    agent = await build_agent(_cfg(tmp_path), _NoopUI(), llm=FakeLLM([]))
    assert not isinstance(agent.tools["write_file"], ApprovedTool)


async def test_build_agent_gives_deps_default_stores(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setitem(TOOLS, "fakes", lambda p, d: seen.append(d) or FakeTool())
    await build_agent(_cfg(tmp_path), _NoopUI(), llm=FakeLLM([]))
    assert seen[0].stores.conversation is not None


class _RecordingNode(PipelineNode):
    def __init__(self):
        self.states: list[dict] = []
        self.histories: list[list] = []

    async def handle(self, ctx, next):
        self.states.append(dict(ctx.state))
        self.histories.append([(m.role, m.content) for m in ctx.messages])
        ctx.response = f"reply to {ctx.user_input}"
        return ctx


async def test_chat_turn_state_and_history_threading():
    from core.agent import Agent
    from storage.stores import InMemoryStores

    node, stores = _RecordingNode(), InMemoryStores()
    agent = Agent(FakeLLM([]), [], [PersistNode(stores.conversation), node])

    await chat_turn(agent, stores, "s1", "one")
    await chat_turn(agent, stores, "s1", "two", persona="pirate")

    assert node.states[0] == {
        "session_id": "s1",
        "parent_id": None,
        "artifact": None,
        "summary": None,
    }
    assert node.states[1]["persona"] == "pirate"
    assert node.states[1]["parent_id"] is not None
    assert node.histories[1] == [("user", "one"), ("assistant", "reply to one")]


async def test_chat_turn_parent_id_creates_sibling_branch():
    from core.agent import Agent
    from storage.stores import InMemoryStores

    node, stores = _RecordingNode(), InMemoryStores()
    agent = Agent(FakeLLM([]), [], [PersistNode(stores.conversation), node])
    conv = stores.conversation

    await chat_turn(agent, stores, "s", "one")
    first_reply = conv.head("s")
    await chat_turn(agent, stores, "s", "two")
    await chat_turn(agent, stores, "s", "two edited", parent_id=first_reply)
    await chat_turn(agent, stores, "s", "three")  # continues the new head branch

    assert node.histories[2] == [("user", "one"), ("assistant", "reply to one")]
    assert [c for _, c in node.histories[3]] == [
        "one",
        "reply to one",
        "two edited",
        "reply to two edited",
    ]


async def test_chat_turn_root_sentinel_forks_off_no_parent():
    from core.agent import Agent
    from storage.stores import InMemoryStores

    node, stores = _RecordingNode(), InMemoryStores()
    agent = Agent(FakeLLM([]), [], [PersistNode(stores.conversation), node])

    await chat_turn(agent, stores, "s", "one")
    await chat_turn(agent, stores, "s", "one bis", parent_id=ROOT)

    assert node.states[1]["parent_id"] is None
    assert node.histories[1] == []


async def test_chat_turn_without_persist_node_does_not_append():
    """`chat_turn` itself never writes: without a `PersistNode`, nothing is saved."""
    from core.agent import Agent
    from storage.stores import InMemoryStores

    node, stores = _RecordingNode(), InMemoryStores()
    agent = Agent(FakeLLM([]), [], [node])

    await chat_turn(agent, stores, "s", "one")

    assert stores.conversation.head("s") is None


# --- open_stores ---------------------------------------------------------------


async def test_open_stores_persist_dir_uses_on_disk_conversation_store(tmp_path):
    from app import open_stores
    from core.types import Message
    from storage.conversation import ConversationStore
    from storage.profile import ProfileStore

    sessions_dir = tmp_path / "sessions"
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"""
llm:
  provider: ollama
  model: fake
pipeline:
  - persist: {{ dir: {sessions_dir} }}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    stores = await open_stores(str(cfg_path))

    assert isinstance(stores.conversation, ConversationStore)
    assert isinstance(stores.profile, ProfileStore)
    assert stores.profile.load() == {"preferences": {}, "instructions": ""}

    # actually persists to that dir
    stores.conversation.append("s1", None, [Message("user", "hi")])
    assert sessions_dir.exists()
    assert (sessions_dir / "s1.json").exists()


async def test_open_stores_without_persist_is_in_memory(tmp_path):
    from app import open_stores

    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "llm:\n  provider: ollama\n  model: fake\npipeline:\n  - react\n",
        encoding="utf-8",
    )

    stores = await open_stores(str(cfg_path))

    assert stores.conversation._root is None
    assert stores.profile._path is None


async def test_open_stores_context_profile_path_honored(tmp_path):
    from app import open_stores

    profile_path = tmp_path / "custom_profile.yaml"
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"""
llm:
  provider: ollama
  model: fake
pipeline:
  - context: {{ profile_path: {profile_path} }}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    stores = await open_stores(str(cfg_path))

    assert stores.profile._path == profile_path


async def test_open_stores_context_without_profile_path_uses_default(tmp_path, monkeypatch):
    from app import open_stores

    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "llm:\n  provider: ollama\n  model: fake\npipeline:\n  - context\n",
        encoding="utf-8",
    )

    stores = await open_stores(str(cfg_path))

    assert str(stores.profile._path) == "data/profile.yaml"


async def test_open_stores_without_context_profile_is_in_memory(tmp_path):
    from app import open_stores

    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "llm:\n  provider: ollama\n  model: fake\npipeline:\n  - react\n",
        encoding="utf-8",
    )

    stores = await open_stores(str(cfg_path))

    assert stores.profile._path is None


async def test_build_agent_uses_open_stores_equivalent_without_reloading(monkeypatch, tmp_path):
    """`build_agent` doesn't call `open_stores` (which would reparse the config again)."""
    import app as app_module

    calls = []
    real_load_config = app_module.load_config

    def counting_load_config(path):
        calls.append(path)
        return real_load_config(path)

    monkeypatch.setattr(app_module, "load_config", counting_load_config)

    cfg_path = _write_config(tmp_path)
    await build_agent(str(cfg_path), _NoopUI(), llm=FakeLLM([]))

    assert len(calls) == 1


# --- sub-agents (A2) -----------------------------------------------------------

_SUB_LLM = "llm: {provider: ollama, model: m}\n"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_build_agent_exposes_sub_agent_as_ask_tool(tmp_path):
    _write(
        tmp_path / "agents" / "helper.yaml", _SUB_LLM + "pipeline: [react]\ntools: [calculator]\n"
    )
    cfg = _write(
        tmp_path / "agent.yaml",
        _SUB_LLM
        + "pipeline: [react]\n"
        + "agents:\n"
        + "  - {name: helper, config: agents/helper.yaml, description: Aide., max_calls: 5}\n",
    )
    ui = _NoopUI()

    agent = await build_agent(str(cfg), ui, llm=FakeLLM([]), stores=InMemoryStores())
    try:
        tool = agent.tools["ask_helper"]
        assert isinstance(tool, AgentTool)
        assert tool.max_calls == 5
        assert tool.max_depth == 2
        assert set(tool.agent.tools) == {"calculator"}
        react = tool.agent.pipeline[0]
        assert isinstance(react.on_event.__self__, SubAgentUI)
        assert react.on_event.__self__.parent is ui
    finally:
        await agent.aclose()


async def test_build_agent_closes_sub_agents_on_aclose(tmp_path):
    _write(tmp_path / "helper.yaml", _SUB_LLM + "pipeline: [react]\n")
    cfg = _write(
        tmp_path / "agent.yaml",
        _SUB_LLM + "agents: [{name: helper, config: helper.yaml, description: d}]\n",
    )

    agent = await build_agent(str(cfg), _NoopUI(), llm=FakeLLM([]), stores=InMemoryStores())
    sub = agent.tools["ask_helper"].agent
    await agent.aclose()

    assert sub._stack is None


@pytest.mark.parametrize("self_cycle", [True, False])
async def test_build_agent_rejects_sub_agent_config_cycle(tmp_path, self_cycle):
    if self_cycle:
        cfg = _write(
            tmp_path / "a.yaml", _SUB_LLM + "agents: [{name: a, config: a.yaml, description: d}]\n"
        )
    else:
        cfg = _write(
            tmp_path / "a.yaml", _SUB_LLM + "agents: [{name: b, config: b.yaml, description: d}]\n"
        )
        _write(
            tmp_path / "b.yaml", _SUB_LLM + "agents: [{name: a, config: a.yaml, description: d}]\n"
        )

    with pytest.raises(ValueError, match="cycle"):
        await build_agent(str(cfg), _NoopUI(), llm=FakeLLM([]), stores=InMemoryStores())


async def test_build_agent_rejects_duplicate_tool_names(tmp_path):
    _write(tmp_path / "helper.yaml", _SUB_LLM)
    cfg = _write(
        tmp_path / "agent.yaml",
        _SUB_LLM
        + "agents:\n"
        + "  - {name: helper, config: helper.yaml, description: d}\n"
        + "  - {name: helper, config: helper.yaml, description: d}\n",
    )

    with pytest.raises(ValueError, match="Duplicate tool name 'ask_helper'"):
        await build_agent(str(cfg), _NoopUI(), llm=FakeLLM([]), stores=InMemoryStores())


async def test_sub_agent_shares_parent_stores_and_config_dir(tmp_path):
    # The sub-agent config sits in agents/, which has no personas/: its persona
    # section only builds if it resolves against the parent's config_dir.
    _write(
        tmp_path / "agents" / "helper.yaml",
        _SUB_LLM + "pipeline: [{context: {sections: [persona]}}]\n",
    )
    cfg = _write(
        tmp_path / "agent.yaml",
        _SUB_LLM + "agents: [{name: helper, config: agents/helper.yaml, description: d}]\n",
    )
    stores = InMemoryStores()

    agent = await build_agent(
        str(cfg), _NoopUI(), llm=FakeLLM([]), stores=stores, config_dir=Path("config")
    )
    try:
        assert "ask_helper" in agent.tools
    finally:
        await agent.aclose()


async def test_tutor_sub_agent_builds_from_real_config(tmp_path):
    cfg = _write(
        tmp_path / "agent.yaml",
        _SUB_LLM
        + "agents: [{name: tutor, config: "
        + str(Path("config/agents/tutor.yaml").resolve())
        + ", description: d}]\n",
    )

    agent = await build_agent(
        str(cfg), _NoopUI(), llm=FakeLLM([]), stores=InMemoryStores(), config_dir=Path("config")
    )
    try:
        assert set(agent.tools["ask_tutor"].agent.tools) == {"calculator"}
    finally:
        await agent.aclose()
