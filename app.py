"""Assembly root: builds an `Agent` from `config/agent.yaml` (wiki §9.1).

This is where every layer of the project meets: `core/` interfaces, `llm/`
adapters, `tools/` and `nodes/` implementations. Nothing else in the codebase
is allowed to import `llm/` — only this module, the composition root, wires
a concrete `ILLM` backend into the abstract pipeline.

`Deps` is the single object carrying shared dependencies into the `from_config`
factories (no globals, per `CLAUDE.md`). It is deliberately small for now:
`stores` (storage layer) and the `approval`/`mcp_servers` wiring come with
their own stories, and `chat_turn` (the CLI/Web/scheduler shared entry point)
comes with persistence. `build_agent` alone is enough to drive a CLI turn.

Adding a feature is meant to stay a one-line change: register a new node or
tool factory in `NODES`/`TOOLS`, then add one line to `config/agent.yaml` —
no change to `build_agent` itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.agent import Agent
from core.llm import ILLM
from core.node import PipelineNode
from core.tool import ITool
from core.ui import UserInterface
from llm.ollama import Ollama, OllamaEmbedder
from nodes.react_loop import ReActLoopNode
from tools.calculator import CalculatorTool


@dataclass
class Deps:
    """Shared dependencies injected into node/tool factories.

    `stores` (conversation tree, memory DB, vector store) is not here yet:
    it arrives with the storage layer's own story.
    """

    llm: ILLM
    ui: UserInterface


NODES: dict[str, Callable[[dict, Deps], PipelineNode]] = {
    "react": ReActLoopNode.from_config,
}

TOOLS: dict[str, Callable[[dict, Deps], ITool | list[ITool]]] = {
    "calculator": CalculatorTool.from_config,
}


def normalize(entry: Any) -> tuple[str, dict]:
    """Turn one `pipeline`/`tools` YAML entry into `(name, params)`.

    An entry is either a bare string (`"calculator"` -> `("calculator", {})`)
    or a single-key mapping (`{"react": {...}}` -> `("react", {...})`, with
    `None` params normalized to `{}`).
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(
                f"Malformed config entry {entry!r}: expected exactly one key, got {len(entry)}"
            )
        ((name, params),) = entry.items()
        return name, params or {}
    raise ValueError(f"Malformed config entry {entry!r}: expected a string or a single-key dict")


def load_config(path: str) -> dict:
    """Load and parse the YAML config file at `path`."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_llm(llm_cfg: dict) -> Ollama:
    """Build the `Ollama` adapter (and its embedder) from the `llm:` config section."""
    cfg = dict(llm_cfg)
    provider = cfg.pop("provider", None)
    if provider != "ollama":
        raise ValueError(f"Unsupported llm provider {provider!r}: only 'ollama' is supported")

    embed_model = cfg.pop("embed_model", None)
    embedder_kwargs = {"host": cfg.get("host", "http://localhost:11434")}
    if embed_model is not None:
        embedder_kwargs["model"] = embed_model
    embedder = OllamaEmbedder(**embedder_kwargs)
    return Ollama(embedder=embedder, **cfg)


async def build_agent(cfg_path: str, ui: UserInterface, llm: ILLM | None = None) -> Agent:
    """Build an `Agent` from the YAML config at `cfg_path`.

    `llm`, when given, overrides the config's `llm:` section (tests inject a
    `FakeLLM` this way and never touch the network/`Ollama`).
    """
    cfg = await asyncio.to_thread(load_config, cfg_path)
    resolved_llm = llm if llm is not None else build_llm(cfg["llm"])
    deps = Deps(llm=resolved_llm, ui=ui)

    tools: list[ITool] = []
    for name, params in map(normalize, cfg.get("tools", [])):
        factory = TOOLS.get(name)
        if factory is None:
            raise ValueError(f"Unknown tool {name!r}. Available: {', '.join(sorted(TOOLS))}")
        result = factory(params, deps)
        if isinstance(result, list):
            tools.extend(result)
        else:
            tools.append(result)

    pipeline: list[PipelineNode] = []
    for name, params in map(normalize, cfg.get("pipeline", [])):
        factory = NODES.get(name)
        if factory is None:
            raise ValueError(f"Unknown node {name!r}. Available: {', '.join(sorted(NODES))}")
        pipeline.append(factory(params, deps))

    return Agent(resolved_llm, tools, pipeline)
