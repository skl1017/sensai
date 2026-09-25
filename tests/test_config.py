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
    assert [name for name, _ in cfg.pipeline] == ["persist", "context", "react"]


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
