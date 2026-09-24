# CLAUDE.md

Sensai: a pure-Python AI agent, no framework. A minimal core (`Agent`, `ILLM`, `ITool`, `PipelineNode`) onto which every feature plugs in as a node or a tool. Source of truth: private wiki `skl1017/sensai` → "Documentation d'architecture" (`git clone https://github.com/skl1017/sensai.wiki.git`; the wiki is written in French).

> The core (`core/`), test doubles (`tests/fakes.py`) and CI exist; the rest describes the target architecture: check that files exist before referring to them.

## Design principles

- **Dependency injection**: the `Agent` receives its LLM, tools and pipeline; nothing is hard-imported. Shared dependencies go through a `Deps` object, never globals.
- **Middleware**: each node receives the `Context` and `next`, and acts before, after, or instead of the rest of the chain.
- **Single shared state**: the `Context` flows through the whole pipeline.
- **Async everywhere** (streaming, interruption, Web UI). Synchronous I/O goes through `asyncio.to_thread`.
- **Fail safe**: a tool error becomes an observation; a critical tool runs only after user approval; deny by default (permissions, approvals).

## Structure and dependency rules

```
core/        types.py, llm.py (ILLM, IEmbedder), tool.py (ITool), node.py (PipelineNode), runtime.py (ContextVars), agent.py
llm/         ollama.py (ILLM + IEmbedder adapter, httpx)
tools/       ITool implementations (calculator, file_tool, web_search, code_sandbox, mcp_adapter, ask_user_tool, approved_tool, memory_tool, agent_tool)
nodes/       PipelineNode implementations (logging, persist, input_guardrail, cache, context_builder, react_loop, output_guardrail, eval, artifact)
storage/     conversation (tree), memory_db (SQLite), vector_store, ingest, prompt_registry
mcp_server/  server.py (exposes ITools over MCP)
eval/        run.py (evaluation bench)
config/      agent.yaml, permissions.yaml, schedules.yaml, personas/, prompts/
ui/          cli.py, web/, export.py, dashboard.py, headless.py
app.py       build_agent, chat_turn
tests/       unit + adversarial/
```

- `core/` depends on nothing; everything depends on `core/`, never the reverse.
- `nodes/` depends on `core/` and `storage/`; `llm/`, `tools/`, `storage/`, `ui/` depend on `core/`.
- Nodes never know concrete tools: they go through `ctx.tools`.

## Core types and interfaces

- Dataclasses: `ToolCall`, `Message`, `LLMChunk`, `Context(user_input, messages, llm, tools, response, state)`.
- `Context.messages`: nodes append, never remove (except `ContextBuilderNode`). `response` stays `None` until a final answer exists. `state`: each key has a single owner node; other nodes only read it.
- `ILLM.chat`: declared `def` in the interface, implemented as an `async def` generator. The last chunk has `done=True` plus token counts. With `schema`, the streamed text forms valid JSON.
- `ITool.run(**kwargs) -> str`: always returns a `str`, never blocks the event loop. `name` unique snake_case, `description` says *when* to use it, `parameters` is a JSON Schema `object` with `additionalProperties: false`. Tools don't receive the `Context`: request info comes via `ContextVar`s (`core/runtime.py`).
- `PipelineNode.handle(ctx, next)`: calls `next` at most once and returns the `Context`. Not calling `next` = short-circuit, so `ctx.response` must be set. **Never catch `asyncio.CancelledError`.**

## Pipeline (order = list order in `agent.yaml`)

1. `LoggingNode`: wraps everything, LLM/tool measurement proxies, JSONL/SQLite event
2. `PersistNode`: saves the exchange on the way back (conversation tree, atomic writes)
3. `InputGuardrailNode`: blocks before the model (short-circuit)
4. `CacheNode`: semantic cache (short-circuits on hit)
5. `ContextBuilderNode`: pluggable `ContextSection`s (persona, profile, memory, artifact, rag, fewshot) + token budget/summary
6. `ReActLoopNode`: LLM ↔ tools loop, errors = observations, `max_iterations`, `tool_timeout`
7. `OutputGuardrailNode`: masks/refuses PII before caching
8. `EvalNode`: LLM judge, schema-constrained
9. `ArtifactNode`: living-document edits, validated by code

Order is structural: what a node does after `next` runs once the following nodes are done. Don't reorder without re-reading section 9 of the wiki.

## Tools

All implement `ITool` and are called uniformly by the ReAct node. Common rules: description is read by the model, short return values (truncate at the source), errors = exceptions with explicit messages, never secrets in returns/errors. Tools with side effects (`write_file`, `run_python`, `memory_forget`, MCP tools not in `read_only`) are wrapped by `ApprovedTool` (human-in-the-loop) per the `approval` section of `agent.yaml`. `FileTools` go through `PermissionPolicy` (resolve the path first, deny by default, `deny` wins).

## Configuration and assembly

- `config/agent.yaml` describes llm, persona, pipeline, tools, approval, mcp_servers. Adding/removing a feature = adding/removing a line, no core change.
- `app.py`: `build_agent(cfg_path, ui)` builds the agent from the `NODES` / `TOOLS` factory registries (`from_config(params, deps)`). `chat_turn(...)` is the shared entry point for CLI / Web / scheduler.
- Switching persona = one YAML file in `config/personas/`, no code.

## Adding a component

- **New tool**: `ITool` class in `tools/` (template: `CalculatorTool`), register in `TOOLS`, enable in `agent.yaml`.
- **New node**: `PipelineNode` class in `nodes/` with `from_config`, register in `NODES`, pick its position based on its before/after-`next` behavior.
- **New `ctx.state` key**: single owner; document it in the wiki.

## Testing

`pytest` + `pytest-asyncio`. Everything is testable without Ollama thanks to `FakeLLM(ILLM)` (scripted responses/tool calls), `FakeTool` and `FakeUI` (answers `yes`/`no`). Levels: unit (nodes, security, compression), integration (marked `slow`, with Ollama), adversarial (`tests/adversarial/cases.yaml`), evaluation (`eval/run.py`).

## Environment and CI

- Local LLM via Ollama (`http://localhost:11434`, `/api/chat`, `/api/embed`); always send `num_ctx` explicitly.
- `.env*` and `__pycache__` are gitignored: never commit secrets.
- `.github/workflows/CD.yml`: a push to `main` on `skl1017/sensai` is mirrored to the Epitech repo. Work on `feat/*` branches, PR into `main`.
- `.github/workflows/CI.yml`: on push to `main`/`feat/**` and PRs into `main`, installs `.[dev]` from `pyproject.toml`, runs `ruff check`, `ruff format --check` and `pytest -m "not slow"`.
