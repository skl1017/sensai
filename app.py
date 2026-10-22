"""Assembly root: builds an `Agent` from `config/agent.yaml` (wiki §9.1).

This is where every layer of the project meets: `core/` interfaces, `llm/`
adapters, `tools/` and `nodes/` implementations. Nothing else in the codebase
is allowed to import `llm/` — only this module, the composition root, wires
a concrete `ILLM` backend into the abstract pipeline.

`Deps` is the single object carrying shared dependencies into the `from_config`
factories (no globals, per `CLAUDE.md`). `stores` default to on-disk-backed
`InMemoryStores` built by `open_stores`/`_stores_from_config` from the
`persist`/`context` pipeline entries (X2, #23-#25). `mcp_servers` are
connected by `build_agent` (their tools join the others; the `AsyncExitStack`
owning the connections is handed to the `Agent`, whose `aclose()` releases it).
`chat_turn` is the CLI/Web/scheduler shared entry point: it loads
history/artifact/summary and runs the agent, but no longer appends the exchange
itself — `PersistNode`, sitting in the configured pipeline, owns that write.

Adding a feature is meant to stay a one-line change: register a new node or
tool factory in `NODES`/`TOOLS`, then add one line to `config/agent.yaml` —
no change to `build_agent` itself.
"""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from config.db import db
from config_loader import AgentConfig, load_config, normalize  # noqa: F401
from core.agent import Agent
from core.llm import ILLM, IEmbedder, IReranker
from core.node import PipelineNode
from core.tool import ITool
from core.types import Context
from core.ui import UserInterface
from llm.cross_encoder import FastEmbedReranker
from llm.ollama import Ollama, OllamaEmbedder
from nodes.context_builder import ContextBuilderNode
from nodes.persist_node import PersistNode
from nodes.react_loop import ReActLoopNode
from nodes.reranker import LLMReranker
from storage.chunk_store import ChunkStore
from storage.conversation import ConversationStore
from storage.profile import ProfileStore
from storage.stores import InMemoryStores, Stores
from storage.vector_storage import SqliteVectorStorage
from tools.approved_tool import ApprovedTool
from tools.ask_user_tool import AskUserTool
from tools.calculator import CalculatorTool
from tools.mcp_adapter import MCPToolAdapter, connect_servers

ROOT = "root"  # sentinel `parent_id`: fork under the session root (no parent)


@dataclass
class Deps:
    """Shared dependencies injected into node/tool factories.

    `stores` (conversation tree, memory DB, vector store, journal) is optional
    until the storage layer is wired in. `embedder` is the LLM's embedder
    when it has one (`FewShotSection`); `reranker` is built from the
    config's `reranker:` section (`RAGSection` with `rerank: true`);
    `config_dir` is the directory of the loaded `agent.yaml` (personas,
    prompts, few-shot bank are resolved against it);
    `persona`/`prompt_version` are the config's root keys.
    """

    llm: ILLM
    ui: UserInterface
    stores: Stores | None = None
    embedder: IEmbedder | None = None
    reranker: IReranker | None = None
    config_dir: Path = Path("config")
    persona: str = "default"
    prompt_version: str | None = None


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
    "persist": PersistNode.from_config,
    "context": ContextBuilderNode.from_config,
    "react": ReActLoopNode.from_config,
}

TOOLS: dict[str, Callable[[dict, Deps], ITool | list[ITool]]] = {
    "calculator": CalculatorTool.from_config,
    "ask_user": AskUserTool.from_config,
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


def build_reranker(reranker_cfg: dict | None, llm: ILLM) -> IReranker | None:
    """Build the `IReranker` from the `reranker:` config section (`None` when absent).

    `provider: fastembed` (default) is a local cross-encoder (`model`,
    `cache_dir` forwarded); `provider: llm` grades passages with `llm`.
    """
    if reranker_cfg is None:
        return None
    cfg = dict(reranker_cfg)
    provider = cfg.pop("provider", "fastembed")
    if provider == "fastembed":
        return FastEmbedReranker(**cfg)
    if provider == "llm":
        if cfg:
            raise ValueError(
                f"Unknown reranker param(s) for provider 'llm': {', '.join(sorted(cfg))}"
            )
        return LLMReranker(llm)
    raise ValueError(f"Unsupported reranker provider {provider!r}: expected 'fastembed' or 'llm'")


def needs_approval(tool: ITool, cfg: AgentConfig) -> bool:
    """True if `tool` must go through `ApprovedTool`.

    An `MCPToolAdapter` needs approval unless its `remote_name` is in the
    `read_only` list of its server's `mcp_servers` entry (wiki §8.9; deny by
    default, including for an unknown server). Any other tool needs it when
    its name matches a pattern of the config's `approval` section (fnmatch).
    """
    if isinstance(tool, MCPToolAdapter):
        server = next((s for s in cfg.mcp_servers if s.name == tool.server), None)
        return server is None or tool.remote_name not in server.read_only
    return any(fnmatch.fnmatchcase(tool.name, pattern) for pattern in cfg.approval)


def _stores_from_config(cfg: AgentConfig) -> InMemoryStores:
    """Build the default `InMemoryStores` from the `persist`/`context` pipeline entries.

    `ConversationStore(root=...)` is used when a `persist` entry carries a
    `dir` param, in-memory otherwise. `ProfileStore` is backed by the
    `context` entry's `profile_path` (default `"data/profile.yaml"`) when a
    `context` entry is present at all, in-memory (`ProfileStore(None)`)
    otherwise. Kept deliberately simple: no validation beyond what
    `PersistNode.from_config`/`ContextBuilderNode.from_config` already do.
    """
    persist_params = next((params for name, params in cfg.pipeline if name == "persist"), None)
    context_params = next((params for name, params in cfg.pipeline if name == "context"), None)

    conversation = None
    if persist_params is not None and "dir" in persist_params:
        conversation = ConversationStore(root=persist_params["dir"])

    profile = None
    if context_params is not None:
        profile = ProfileStore(context_params.get("profile_path", "data/profile.yaml"))

    chunk_store = ChunkStore(
        db.engine,
        SqliteVectorStorage(db.engine),
    )
    return InMemoryStores(conversation=conversation, profile=profile, chunk_store=chunk_store)


async def open_stores(cfg_path: str) -> InMemoryStores:
    """Build the `Stores` a session needs, straight from `cfg_path` (CLI/Web entry point).

    Callers that also call `build_agent` on the same config should pass the
    result in as `stores=` so the config isn't parsed a third time.
    """
    cfg = await asyncio.to_thread(load_config, cfg_path)
    return _stores_from_config(cfg)


async def build_agent(
    cfg_path: str,
    ui: UserInterface,
    llm: ILLM | None = None,
    stores: Stores | None = None,
) -> Agent:
    """Build an `Agent` from the YAML config at `cfg_path`.

    `llm`, when given, overrides the config's `llm:` section (tests inject a
    `FakeLLM` this way and never touch the network/`Ollama`). `stores`
    defaults to `_stores_from_config(cfg)` (built from the already-parsed
    config, so it isn't loaded twice just to get here). Tools matching
    `approval` are wrapped in `ApprovedTool`. `mcp_servers` are connected here
    and their tools added (wrapped unless `read_only`, see `needs_approval`);
    the connections live in an `AsyncExitStack` owned by the returned agent,
    so callers must `await agent.aclose()` when done. If assembly fails, the
    stack is closed before the error propagates.
    """
    cfg = await asyncio.to_thread(load_config, cfg_path)
    resolved_llm = llm if llm is not None else build_llm(cfg.llm.as_dict())
    resolved_stores = stores if stores is not None else _stores_from_config(cfg)
    deps = Deps(
        llm=resolved_llm,
        ui=ui,
        stores=resolved_stores,
        embedder=getattr(resolved_llm, "embedder", None),
        reranker=build_reranker(cfg.reranker, resolved_llm),
        config_dir=Path(cfg_path).parent,
        persona=cfg.persona,
        prompt_version=cfg.prompt_version,
    )

    stack = AsyncExitStack()
    try:
        built: list[ITool | list[ITool]] = []
        for name, params in cfg.tools:
            factory = TOOLS.get(name)
            if factory is None:
                raise ValueError(f"Unknown tool {name!r}. Available: {', '.join(sorted(TOOLS))}")
            built.append(factory(params, deps))
        all_tools = flatten(built) + await connect_servers(cfg.mcp_servers, stack)
        tools = [ApprovedTool(t, ui) if needs_approval(t, cfg) else t for t in all_tools]

        pipeline: list[PipelineNode] = []
        for name, params in cfg.pipeline:
            factory = NODES.get(name)
            if factory is None:
                raise ValueError(f"Unknown node {name!r}. Available: {', '.join(sorted(NODES))}")
            pipeline.append(factory(params, deps))

        return Agent(resolved_llm, tools, pipeline, stack)
    except BaseException:
        await stack.aclose()
        raise


async def chat_turn(
    agent: Agent,
    stores: Stores,
    session_id: str,
    text: str,
    persona: str | None = None,
    parent_id: str | None = None,
) -> Context:
    """Shared entry point for CLI / Web / scheduler (wiki §9.2, X2).

    Loads the branch ending at `parent_id` as history, seeds `ctx.state`, and
    runs the agent. `parent_id`: `None` (default) means "the current session
    head"; `app.ROOT` means "no parent" (fork off the very first message);
    anything else is used as-is (an older node id, to fork a sibling branch
    there). `persona`, when given, seeds `state["persona"]` for this turn
    only; otherwise `PersonaSection` falls back to the config's root
    `persona`. The history is the same either way, so switching persona
    between turns never loses it (A5). All store reads are synchronous I/O, so they go through
    `asyncio.to_thread`.

    This function no longer appends the exchange: `PersistNode`, wired into
    the configured pipeline, owns that write (it runs after the rest of the
    chain, so a raised exception or a cancelled turn saves nothing).
    """
    conv = stores.conversation
    if parent_id == ROOT:
        parent = None
    elif parent_id:
        parent = parent_id
    else:
        parent = await asyncio.to_thread(conv.head, session_id)

    history = await asyncio.to_thread(conv.branch, session_id, parent)
    artifact = await asyncio.to_thread(conv.load_artifact, session_id)
    summary = await asyncio.to_thread(conv.load_summary, session_id)
    state = {
        "session_id": session_id,
        "parent_id": parent,
        "artifact": artifact,
        "summary": summary,
    }
    if persona is not None:
        state["persona"] = persona
    return await agent.run(text, history, state)
