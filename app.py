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
`agents` entries are sub-agents built recursively from their own config file
and exposed as `ask_<name>` tools (A2); their `aclose()` is registered on the
same stack.
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
import os
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from config.db import db
from config_loader import AgentConfig, SubAgentConfig, load_config, normalize  # noqa: F401
from core.agent import Agent
from core.llm import ILLM, IEmbedder, IReranker
from core.node import PipelineNode
from core.tool import ITool
from core.types import Context
from core.ui import UserInterface
from llm.cross_encoder import FastEmbedReranker
from llm.ollama import Ollama, OllamaEmbedder
from nodes.cache_node import CacheNode
from nodes.context_builder import ContextBuilderNode
from nodes.input_guardrail import InputGuardrailNode
from nodes.output_guardrail import OutputGuardrailNode
from nodes.persist_node import PersistNode
from nodes.react_loop import ReActLoopNode
from nodes.reranker import LLMReranker
from storage.chunk_store import ChunkStore
from storage.conversation import ConversationStore
from storage.memory_db import MemoryDB
from storage.profile import ProfileStore
from storage.semantic_cache import SemanticCache
from storage.stores import InMemoryStores, Stores
from storage.vector_storage import SqliteVectorStorage
from tools.agent_tool import AgentTool, SubAgentUI
from tools.approved_tool import ApprovedTool
from tools.ask_user_tool import AskUserTool
from tools.calculator import CalculatorTool
from tools.code_sandbox import CodeSandboxTool
from tools.file_tool import ListDirTool, PermissionPolicy, ReadFileTool, WriteFileTool
from tools.mcp_adapter import MCPToolAdapter, connect_servers
from tools.memory_tool import from_config as memory_tools_from_config
from tools.web_search import build_web_tools

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
    `persona`/`prompt_version` are the config's root keys. `stream` is
    `False` when an `output_guardrail` entry is `strict` (EV2): the ReAct node
    then streams nothing and the UI shows the checked answer only.
    """

    llm: ILLM
    ui: UserInterface
    stores: Stores | None = None
    embedder: IEmbedder | None = None
    reranker: IReranker | None = None
    config_dir: Path = Path("config")
    persona: str = "default"
    prompt_version: str | None = None
    permissions: PermissionPolicy | None = None
    stream: bool = True


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
    "input_guardrail": InputGuardrailNode.from_config,
    "cache": CacheNode.from_config,
    "context": ContextBuilderNode.from_config,
    "react": ReActLoopNode.from_config,
    "output_guardrail": OutputGuardrailNode.from_config,
}

TOOLS: dict[str, Callable[[dict, Deps], ITool | list[ITool]]] = {
    "calculator": CalculatorTool.from_config,
    "ask_user": AskUserTool.from_config,
    "memory": memory_tools_from_config,
    "read_file": ReadFileTool.from_config,
    "list_dir": ListDirTool.from_config,
    "write_file": WriteFileTool.from_config,
    "web_search": build_web_tools,
    "run_python": CodeSandboxTool.from_config,
}


def build_llm(llm_cfg: dict) -> Ollama:
    """Build the `Ollama` adapter (and its embedder) from the `llm:` config section.

    `api_key_env` names the environment variable holding the chat API key (the
    key itself never sits in the config); `embed_host` lets embeddings stay on
    another server than the chat, and defaults to `host`. The embedder never
    receives the key.
    """
    cfg = dict(llm_cfg)
    provider = cfg.pop("provider", None)
    if provider != "ollama":
        raise ValueError(f"Unsupported llm provider {provider!r}: only 'ollama' is supported")

    api_key_env = cfg.pop("api_key_env", None)
    if api_key_env is not None:
        cfg["api_key"] = os.environ.get(api_key_env)
        if not cfg["api_key"]:
            raise ValueError(f"llm.api_key_env: environment variable {api_key_env} is not set")

    embed_model = cfg.pop("embed_model", None)
    default_host = cfg.get("host", "http://localhost:11434")
    embedder_kwargs = {"host": cfg.pop("embed_host", default_host)}
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
    """Build the default `InMemoryStores` from the `persist`/`cache`/`context` pipeline entries.

    `ConversationStore(root=...)` is used when a `persist` entry carries a
    `dir` param, in-memory otherwise. `ProfileStore` is backed by the
    `context` entry's `profile_path` (default `"data/profile.yaml"`) when a
    `context` entry is present at all, in-memory (`ProfileStore(None)`)
    otherwise. A `cache` entry adds a `SemanticCache` in the shared engine, its
    vector table sized by the entry's `embedding_dim` (default 768,
    nomic-embed-text). Kept deliberately simple: no validation beyond what
    `PersistNode.from_config`/`ContextBuilderNode.from_config` already do.
    """
    persist_params = next((params for name, params in cfg.pipeline if name == "persist"), None)
    context_params = next((params for name, params in cfg.pipeline if name == "context"), None)
    cache_params = next((params for name, params in cfg.pipeline if name == "cache"), None)

    conversation = None
    if persist_params is not None and "dir" in persist_params:
        conversation = ConversationStore(root=persist_params["dir"])

    profile = None
    if context_params is not None:
        profile = ProfileStore(context_params.get("profile_path", "data/profile.yaml"))

    # `memory` tool entry: the long-term memory lives in the shared engine (config/db/db.py).
    memory_db = MemoryDB(db.engine) if any(name == "memory" for name, _ in cfg.tools) else None

    chunk_store = ChunkStore(
        db.engine,
        SqliteVectorStorage(db.engine),
    )
    semantic_cache = None
    if cache_params is not None:
        semantic_cache = SemanticCache(
            db.engine,
            SqliteVectorStorage(
                db.engine,
                embedding_dim=cache_params.get("embedding_dim", 768),
                table="cache_vectors",
                distance_metric="cosine",
            ),
        )

    return InMemoryStores(
        conversation=conversation,
        profile=profile,
        memory_db=memory_db,
        chunk_store=chunk_store,
        semantic_cache=semantic_cache,
    )


async def open_stores(cfg_path: str) -> InMemoryStores:
    """Build the `Stores` a session needs, straight from `cfg_path` (CLI/Web entry point).

    Callers that also call `build_agent` on the same config should pass the
    result in as `stores=` so the config isn't parsed a third time.
    """
    cfg = await asyncio.to_thread(load_config, cfg_path)
    return _stores_from_config(cfg)


async def _build_sub_agents(
    entries: list[SubAgentConfig],
    cfg_path: str,
    ui: UserInterface,
    stores: Stores,
    config_dir: Path,
    ancestors: tuple[Path, ...],
    stack: AsyncExitStack,
) -> list[ITool]:
    """Build one `AgentTool` per `agents:` entry, each around a recursively built `Agent`.

    A sub-agent shares the parent's `stores` and `config_dir` (personas,
    prompts, few-shot bank) and gets the parent's UI through `SubAgentUI`.
    `ancestors` holds the config files being built up the chain: a sub-agent
    config already in it is a cycle (A -> B -> A) and raises `ValueError`
    instead of recursing forever.
    """
    tools: list[ITool] = []
    for entry in entries:
        sub_path = (Path(cfg_path).parent / entry.config).resolve()
        if sub_path in ancestors:
            chain = " -> ".join(str(p) for p in (*ancestors, sub_path))
            raise ValueError(f"agents: sub-agent config cycle: {chain}")
        sub = await build_agent(
            str(sub_path),
            SubAgentUI(ui, entry.name),
            stores=stores,
            config_dir=config_dir,
            _ancestors=ancestors,
        )
        stack.push_async_callback(sub.aclose)
        guards = {
            key: value
            for key in ("max_depth", "max_calls", "timeout")
            if (value := getattr(entry, key)) is not None
        }
        tools.append(AgentTool(entry.name, entry.description, sub, **guards))
    return tools


def _check_unique_names(tools: list[ITool]) -> None:
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        seen.add(tool.name)


async def build_agent(
    cfg_path: str,
    ui: UserInterface,
    llm: ILLM | None = None,
    stores: Stores | None = None,
    config_dir: Path | None = None,
    _ancestors: tuple[Path, ...] = (),
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
    stack is closed before the error propagates. `config_dir` defaults to
    the config file's directory (sub-agents receive the parent's). Tool
    names, sub-agents' `ask_<name>` included, must be unique.
    `_ancestors` is internal to sub-agent recursion (cycle detection).
    """
    ancestors = (*_ancestors, Path(cfg_path).resolve())
    cfg = await asyncio.to_thread(load_config, cfg_path)
    resolved_llm = llm if llm is not None else build_llm(cfg.llm.as_dict())
    resolved_stores = stores if stores is not None else _stores_from_config(cfg)
    deps = Deps(
        llm=resolved_llm,
        ui=ui,
        stores=resolved_stores,
        embedder=getattr(resolved_llm, "embedder", None),
        reranker=build_reranker(cfg.reranker, resolved_llm),
        config_dir=config_dir if config_dir is not None else Path(cfg_path).parent,
        persona=cfg.persona,
        prompt_version=cfg.prompt_version,
        stream=not any(
            name == "output_guardrail" and params.get("strict") for name, params in cfg.pipeline
        ),
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
        all_tools += await _build_sub_agents(
            cfg.agents, cfg_path, ui, resolved_stores, deps.config_dir, ancestors, stack
        )
        _check_unique_names(all_tools)
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
    rag_filters: dict | None = None,
) -> Context:
    """Shared entry point for CLI / Web / scheduler (wiki §9.2, X2).

    Loads the branch ending at `parent_id` as history, seeds `ctx.state`, and
    runs the agent. `parent_id`: `None` (default) means "the current session
    head"; `app.ROOT` means "no parent" (fork off the very first message);
    anything else is used as-is (an older node id, to fork a sibling branch
    there). `persona`, when given, seeds `state["persona"]` for this turn
    only; otherwise `PersonaSection` falls back to the config's root
    `persona`. The history is the same either way, so switching persona
    between turns never loses it (A5). `rag_filters`, when given, seeds
    `state["rag_filters"]` for this turn only (`file_name`, `date_from`, `date_to`,
    see `RAGSection`). All store reads are synchronous I/O, so they go through
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
    if rag_filters is not None:
        state["rag_filters"] = rag_filters
    return await agent.run(text, history, state)
