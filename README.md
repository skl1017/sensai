# Sensai

Sensai is a pure-Python AI agent built with no framework. It is organized around a minimal
core — `Agent`, `ILLM`, `ITool`, `PipelineNode` — onto which every feature plugs in as a
pipeline node or a tool. The full architecture is documented in the project wiki
(`skl1017/sensai` → "Documentation d'architecture").

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

Runtime and dev dependencies are declared in `pyproject.toml` (the `dev` extra adds
`pytest`, `pytest-asyncio` and `ruff`).

## Running tests

```bash
.venv/bin/pytest
```

## Lint / format

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## Layout

- `core/` — core types and interfaces: `types.py`, `llm.py` (`ILLM`, `IEmbedder`), `tool.py`
  (`ITool`), `node.py` (`PipelineNode`), `runtime.py`, `agent.py`.
- `llm/` — `ILLM`/`IEmbedder` adapters (e.g. Ollama over `httpx`).
- `tools/` — `ITool` implementations.
- `nodes/` — `PipelineNode` implementations that make up the pipeline.
- `storage/` — conversation tree, memory DB, vector store, ingest, prompt registry.
- `mcp_server/` — exposes `ITool`s over MCP.
- `eval/` — evaluation bench.
- `config/` — `agent.yaml`, `permissions.yaml`, `schedules.yaml`, personas, prompts.
- `ui/` — CLI, web UI, export, dashboard, headless mode.
- `tests/` — unit and adversarial tests.

## Environment

Sensai talks to a local LLM via Ollama at `http://localhost:11434` (`/api/chat`, `/api/embed`).
Copy `.env.example` to `.env` and fill in any local configuration needed; `.env*` files are
gitignored and must never be committed.
