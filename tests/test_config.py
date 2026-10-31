from pathlib import Path

import pytest

from config_loader import AgentConfig, ConfigError, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(tmp_path, text):
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_full_example_loads():
    cfg = load_config(CONFIG_DIR / "agent.full.example.yaml")
    assert isinstance(cfg, AgentConfig)
    assert cfg.llm.model == "llama3.1"
    assert cfg.persona == "default"
    assert cfg.prompt_version == "v3"
    assert [name for name, _ in cfg.pipeline][0] == "logging"
    assert ("calculator", {}) in cfg.tools
    assert cfg.approval == ["write_file", "run_python", "memory_forget"]
    assert [s.name for s in cfg.mcp_servers] == ["fs", "sensai"]
    assert cfg.mcp_servers[1].read_only == []


def test_default_config_loads():
    cfg = load_config(CONFIG_DIR / "agent.yaml")
    assert [name for name, _ in cfg.pipeline] == [
        "persist",
        "input_guardrail",
        "context",
        "react",
        "output_guardrail",
    ]


def test_minimal_config_uses_defaults(tmp_path):
    cfg = _load(tmp_path, "llm:\n  model: m\n")
    assert cfg.llm.model == "m"
    assert cfg.persona == "default"
    assert cfg.pipeline == cfg.tools == cfg.approval == cfg.mcp_servers == []


def test_missing_llm(tmp_path):
    with pytest.raises(ConfigError, match=r"llm: required"):
        _load(tmp_path, "persona: x\n")


def test_missing_llm_model(tmp_path):
    with pytest.raises(ConfigError, match=r"llm\.model: required"):
        _load(tmp_path, "llm:\n  provider: ollama\n")


def test_bad_entry(tmp_path):
    with pytest.raises(ConfigError, match=r"tools\[1\]"):
        _load(tmp_path, "llm: {model: m}\ntools:\n  - calculator\n  - 42\n")


def test_two_key_entry(tmp_path):
    text = "llm: {model: m}\npipeline:\n  - a\n  - b\n  - {x: {}, y: {}}\n"
    with pytest.raises(ConfigError, match=r"pipeline\[2\].*got 2 keys"):
        _load(tmp_path, text)


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match=r"foo: unknown key"):
        _load(tmp_path, "llm: {model: m}\nfoo: 1\n")


def test_mcp_server_requires_command(tmp_path):
    with pytest.raises(ConfigError, match=r"mcp_servers\[0\]\.command: required"):
        _load(tmp_path, "llm: {model: m}\nmcp_servers:\n  - name: fs\n")


def test_mcp_server_env(tmp_path):
    cfg = _load(
        tmp_path,
        "llm: {model: m}\nmcp_servers:\n  - name: fs\n    command: npx\n    env: {TOKEN: abc}\n",
    )
    assert cfg.mcp_servers[0].env == {"TOKEN": "abc"}


def test_mcp_server_env_defaults_empty(tmp_path):
    cfg = _load(tmp_path, "llm: {model: m}\nmcp_servers:\n  - name: fs\n    command: npx\n")
    assert cfg.mcp_servers[0].env == {}


def test_mcp_server_env_must_be_string_mapping(tmp_path):
    with pytest.raises(ConfigError, match=r"mcp_servers\[0\]\.env: expected a mapping of strings"):
        _load(
            tmp_path,
            "llm: {model: m}\nmcp_servers:\n  - name: fs\n    command: npx\n    env: {A: 1}\n",
        )


def test_reranker_section_defaults_to_none(tmp_path):
    cfg = _load(tmp_path, "llm: {model: m}\n")
    assert cfg.reranker is None


def test_reranker_section_is_kept_as_mapping(tmp_path):
    cfg = _load(tmp_path, "llm: {model: m}\nreranker: {provider: fastembed, model: x}\n")
    assert cfg.reranker == {"provider": "fastembed", "model": "x"}


def test_reranker_section_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match="reranker"):
        _load(tmp_path, "llm: {model: m}\nreranker: fastembed\n")


def test_agents_default_to_empty(tmp_path):
    assert _load(tmp_path, "llm: {model: m}\n").agents == []


def test_agents_entry_parsed(tmp_path):
    cfg = _load(
        tmp_path,
        "llm: {model: m}\n"
        "agents:\n"
        "  - {name: researcher, config: agents/r.yaml, description: Cherche., max_depth: 1,"
        " max_calls: 2, timeout: 60}\n",
    )
    (entry,) = cfg.agents
    assert (entry.name, entry.config, entry.description) == (
        "researcher",
        "agents/r.yaml",
        "Cherche.",
    )
    assert (entry.max_depth, entry.max_calls, entry.timeout) == (1, 2, 60)


def test_agents_guards_default_to_none(tmp_path):
    cfg = _load(tmp_path, "llm: {model: m}\nagents: [{name: r, config: r.yaml, description: d}]\n")
    assert (cfg.agents[0].max_depth, cfg.agents[0].max_calls, cfg.agents[0].timeout) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("missing", ["name", "config", "description"])
def test_agents_entry_requires_key(tmp_path, missing):
    entry = {"name": "r", "config": "r.yaml", "description": "d"}
    del entry[missing]
    body = ", ".join(f"{k}: {v}" for k, v in entry.items())
    with pytest.raises(ConfigError, match=rf"agents\[0\]\.{missing}: required"):
        _load(tmp_path, f"llm: {{model: m}}\nagents: [{{{body}}}]\n")


@pytest.mark.parametrize(
    "bad", ["max_depth: 0", "max_calls: -1", "timeout: true", "max_depth: 1.5"]
)
def test_agents_guards_must_be_positive(tmp_path, bad):
    with pytest.raises(ConfigError, match="positive number"):
        _load(
            tmp_path,
            f"llm: {{model: m}}\nagents: [{{name: r, config: r.yaml, description: d, {bad}}}]\n",
        )


def test_agents_entry_rejects_unknown_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        _load(
            tmp_path, "llm: {model: m}\nagents: [{name: r, config: r.yaml, description: d, x: 1}]\n"
        )


def test_full_example_declares_tutor_sub_agent():
    cfg = load_config(CONFIG_DIR / "agent.full.example.yaml")
    assert [a.name for a in cfg.agents] == ["tutor"]
    assert (CONFIG_DIR / cfg.agents[0].config).is_file()


def test_tutor_sub_agent_config_loads():
    cfg = load_config(CONFIG_DIR / "agents" / "tutor.yaml")
    assert cfg.persona == "tutor"
    assert [name for name, _ in cfg.pipeline] == ["context", "react"]
    assert cfg.tools == [("calculator", {})]
