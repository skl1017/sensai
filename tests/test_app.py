"""Unit tests for `app.py`: config parsing and agent assembly (wiki §9.1)."""

from __future__ import annotations

import pytest

from app import Deps, build_agent, build_llm, normalize
from llm.ollama import Ollama, OllamaEmbedder
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeLLM, FakeTool


class _NoopUI:
    def on_token(self, text: str) -> None:
        pass

    def on_event(self, kind: str, data: dict) -> None:
        pass


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


async def test_build_agent_real_config_with_fake_llm():
    ui = _NoopUI()
    llm = FakeLLM(["hi"])
    agent = await build_agent("config/agent.yaml", ui, llm=llm)

    assert "calculator" in agent.tools
    assert len(agent.pipeline) == 1
    assert isinstance(agent.pipeline[0], ReActLoopNode)


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
