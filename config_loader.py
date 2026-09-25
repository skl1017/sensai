"""Loading and validation of `config/agent.yaml` (wiki §9).

`load_config` parses the YAML file into plain dataclasses and raises a
`ConfigError` naming the faulty key path (`llm.model`, `pipeline[2]`, ...) on
any structural problem. It only checks the *shape* of the file: whether a node
or tool name exists is `app.build_agent`'s job (it owns the registries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The config file is malformed; the message names the faulty key path."""


@dataclass
class LLMConfig:
    """`llm:` section. `params` holds every other key, forwarded to the adapter."""

    model: str
    provider: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict as consumed by `app.build_llm`."""
        out = dict(self.params)
        out["model"] = self.model
        if self.provider is not None:
            out["provider"] = self.provider
        return out


@dataclass
class McpServerConfig:
    """One `mcp_servers:` entry."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    read_only: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Validated content of `agent.yaml`. `pipeline`/`tools` are `(name, params)` pairs."""

    llm: LLMConfig
    persona: str = "default"
    prompt_version: str | None = None
    pipeline: list[tuple[str, dict]] = field(default_factory=list)
    tools: list[tuple[str, dict]] = field(default_factory=list)
    approval: list[str] = field(default_factory=list)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)


def normalize(entry: Any, where: str = "entry") -> tuple[str, dict]:
    """Turn one `pipeline`/`tools` YAML entry into `(name, params)`.

    An entry is either a bare string (`"calculator"` -> `("calculator", {})`)
    or a single-key mapping (`{"react": {...}}` -> `("react", {...})`, with
    `None` params normalized to `{}`). `where` is the key path used in errors.
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ConfigError(
                f"{where}: expected a string or a single-key mapping, got {len(entry)} keys"
            )
        ((name, params),) = entry.items()
        if not isinstance(name, str):
            raise ConfigError(f"{where}: name must be a string, got {name!r}")
        if params is not None and not isinstance(params, dict):
            raise ConfigError(f"{where}.{name}: expected a mapping, got {type(params).__name__}")
        return name, params or {}
    raise ConfigError(
        f"{where}: expected a string or a single-key mapping, got {type(entry).__name__}"
    )


def _check_keys(data: dict, allowed: set[str], where: str) -> None:
    for key in data:
        if key not in allowed:
            path = f"{where}.{key}" if where else str(key)
            raise ConfigError(f"{path}: unknown key")


def _list(data: dict, key: str) -> list:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{key}: expected a list, got {type(value).__name__}")
    return value


def _str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: expected a list of strings")
    return list(value)


def _parse_llm(raw: Any) -> LLMConfig:
    if raw is None:
        raise ConfigError("llm: required")
    if not isinstance(raw, dict):
        raise ConfigError(f"llm: expected a mapping, got {type(raw).__name__}")
    model = raw.get("model")
    if not model:
        raise ConfigError("llm.model: required")
    if not isinstance(model, str):
        raise ConfigError("llm.model: expected a string")
    provider = raw.get("provider")
    if provider is not None and not isinstance(provider, str):
        raise ConfigError("llm.provider: expected a string")
    params = {k: v for k, v in raw.items() if k not in ("model", "provider")}
    return LLMConfig(model=model, provider=provider, params=params)


def _parse_mcp_server(raw: Any, where: str) -> McpServerConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    _check_keys(raw, {"name", "command", "args", "read_only"}, where)
    for key in ("name", "command"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ConfigError(f"{where}.{key}: required")
    return McpServerConfig(
        name=raw["name"],
        command=raw["command"],
        args=_str_list(raw.get("args"), f"{where}.args"),
        read_only=_str_list(raw.get("read_only"), f"{where}.read_only"),
    )


def parse_config(data: Any) -> AgentConfig:
    """Validate an already-parsed YAML document."""
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping at the top level")
    _check_keys(
        data,
        {"llm", "persona", "prompt_version", "pipeline", "tools", "approval", "mcp_servers"},
        "",
    )
    persona = data.get("persona") or "default"
    if not isinstance(persona, str):
        raise ConfigError("persona: expected a string")
    prompt_version = data.get("prompt_version")
    if prompt_version is not None:
        prompt_version = str(prompt_version)

    return AgentConfig(
        llm=_parse_llm(data.get("llm")),
        persona=persona,
        prompt_version=prompt_version,
        pipeline=[normalize(e, f"pipeline[{i}]") for i, e in enumerate(_list(data, "pipeline"))],
        tools=[normalize(e, f"tools[{i}]") for i, e in enumerate(_list(data, "tools"))],
        approval=_str_list(data.get("approval"), "approval"),
        mcp_servers=[
            _parse_mcp_server(e, f"mcp_servers[{i}]")
            for i, e in enumerate(_list(data, "mcp_servers"))
        ],
    )


def load_config(path: str | Path) -> AgentConfig:
    """Read, parse and validate the YAML config file at `path` (synchronous read)."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_config(data)
