"""Assembly root: builds an `Agent` from `config/agent.yaml` (wiki §9.1).

This is where every layer of the project meets: `core/` interfaces, `llm/`
adapters, `tools/` and `nodes/` implementations. Nothing else in the codebase
is allowed to import `llm/` — only this module, the composition root, wires
a concrete `ILLM` backend into the abstract pipeline.

`Deps` is the single object carrying shared dependencies into the `from_config`
factories (no globals, per `CLAUDE.md`). It is deliberately small for now:
`stores` are in-memory for now and the `mcp_servers` wiring comes with
their own stories. `chat_turn` is the CLI/Web/scheduler shared entry point.

Adding a feature is meant to stay a one-line change: register a new node or
tool factory in `NODES`/`TOOLS`, then add one line to `config/agent.yaml` —
no change to `build_agent` itself.
"""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from config_loader import AgentConfig, load_config, normalize  # noqa: F401
from core.agent import Agent
from core.llm import ILLM
from core.node import PipelineNode
from core.tool import ITool
from core.types import Context, Message
from core.ui import UserInterface
from llm.ollama import Ollama, OllamaEmbedder
from nodes.react_loop import ReActLoopNode
from storage.stores import InMemoryStores, Stores
from tools.approved_tool import ApprovedTool
from tools.calculator import CalculatorTool


@dataclass
class Deps:
    """Shared dependencies injected into node/tool factories.

    `stores` (conversation tree, memory DB, vector store, journal) is optional
    until the storage layer is wired in.
    """

    llm: ILLM
    ui: UserInterface
    stores: Stores | None = None


def flatten(items: Iterable[ITool | list[ITool]]) -> list[ITool]:
    """Flatten factory results (a tool or a list of tools) into one list of tools."""
    flat: list[ITool] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


NODES: dict[str, Callable[[dict, Deps], PipelineNode]] = {
    "react": ReActLoopNode.from_config,
}

TOOLS: dict[str, Callable[[dict, Deps], ITool | list[ITool]]] = {
    "calculator": CalculatorTool.from_config,
}


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


def needs_approval(tool: ITool, cfg: AgentConfig) -> bool:
    """True if `tool.name` matches a pattern of the config's `approval` section (fnmatch).

    Wiki §8.9 also requires approval for MCP tools absent from `read_only`;
    that rule arrives with the MCP story, once MCP tools exist.
    """
    return any(fnmatch.fnmatchcase(tool.name, pattern) for pattern in cfg.approval)


async def build_agent(
    cfg_path: str,
    ui: UserInterface,
    llm: ILLM | None = None,
    stores: Stores | None = None,
) -> Agent:
    """Build an `Agent` from the YAML config at `cfg_path`.

    `llm`, when given, overrides the config's `llm:` section (tests inject a
    `FakeLLM` this way and never touch the network/`Ollama`). `stores` defaults
    to fresh `InMemoryStores`. Tools matching `approval` are wrapped in
    `ApprovedTool`. `mcp_servers` are validated by the loader but not
    connected yet (MCP story).
    """
    cfg = await asyncio.to_thread(load_config, cfg_path)
    resolved_llm = llm if llm is not None else build_llm(cfg.llm.as_dict())
    deps = Deps(llm=resolved_llm, ui=ui, stores=stores if stores is not None else InMemoryStores())

    built: list[ITool | list[ITool]] = []
    for name, params in cfg.tools:
        factory = TOOLS.get(name)
        if factory is None:
            raise ValueError(f"Unknown tool {name!r}. Available: {', '.join(sorted(TOOLS))}")
        built.append(factory(params, deps))
    tools = [ApprovedTool(t, ui) if needs_approval(t, cfg) else t for t in flatten(built)]

    pipeline: list[PipelineNode] = []
    for name, params in cfg.pipeline:
        factory = NODES.get(name)
        if factory is None:
            raise ValueError(f"Unknown node {name!r}. Available: {', '.join(sorted(NODES))}")
        pipeline.append(factory(params, deps))

    return Agent(resolved_llm, tools, pipeline)


async def chat_turn(
    agent: Agent,
    stores: Stores,
    session_id: str,
    text: str,
    persona: str = "default",
    parent_id: str | None = None,
) -> Context:
    """Shared entry point for CLI / Web / scheduler (wiki §9.2).

    Loads the branch ending at `parent_id` (default: the session head) as
    history, seeds `ctx.state`, runs the agent, then appends the exchange
    under that parent. Passing an older `parent_id` creates a sibling branch.
    """
    conv = stores.conversation
    parent = parent_id or conv.head(session_id)
    history = conv.branch(session_id, parent)
    state = {
        "session_id": session_id,
        "persona": persona,
        "parent_id": parent,
        "artifact": conv.load_artifact(session_id),
        "summary": conv.load_summary(session_id),
    }
    ctx = await agent.run(text, history, state)
    conv.append(
        session_id, parent, [Message("user", text), Message("assistant", ctx.response or "")]
    )
    return ctx
