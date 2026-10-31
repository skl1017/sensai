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
.venv/bin/pip install -e '.[rerank]'
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

To run Ollama in Docker and pull the chat and embedding models set in `.env`:

```bash
docker compose up -d        # ollama on :11434 + one-shot model pull + searxng on :8080
docker compose logs -f ollama-pull
```

### Web search (SearXNG)

The `web_search` tool queries a SearXNG instance at `http://localhost:8080` (the `url` of
`web_search` in `config/agent.yaml`), or Google via SerpApi when `SERPAPI_API_KEY` is set. The
`searxng` service of `docker-compose.yml` provides that instance, published on `127.0.0.1` only:

```bash
docker compose up -d searxng
curl "http://localhost:8080/search?q=test&format=json"   # JSON results = ready
```

Its settings live in `config/searxng/settings.yml`: the JSON format is enabled (SearXNG answers
`403` without it) and the anti-bot limiter is disabled (it would block the agent's repeated
queries). Set `SEARXNG_SECRET` in `.env` to replace the default secret key.

## Chat from the terminal

Once Ollama is up (`docker compose up -d` above), start the minimal CLI chatbot:

```bash
.venv/bin/python -m ui.cli               # uses config/agent.yaml
.venv/bin/python -m ui.cli --config path/to/other.yaml
```

Type a message and press Enter; the answer streams token by token. Commands:

- `/help` — list available commands.
- `/quit` (or `/exit`) — leave the chat. Ctrl-C/Ctrl-D also exit.

Example:

```
> what is 12*(3+4)**2?
[calculator(expression='12*(3+4)**2')]
12 * (3 + 4) ** 2 = 588
```

## Limitations

- **Answer quality**: answers depend on the local model; a small model can be wrong or inconsistent.
- **Performance**: response times depend on the hardware and can be long without a GPU.
- **External services**: Ollama is required; `web_search` needs SearXNG or a SerpApi key,
  `run_python` needs Docker and the filesystem MCP server needs `npx`.
- **Documents**: scanned PDFs are not supported (no OCR) and some PDFs extract poorly.
- **Guardrails** (`config/guardrails.yaml`): input checks (length, forbidden patterns,
  injection heuristics, optional LLM scope check) can be bypassed by rephrasing, encoding or
  another language, and only cover the user's message, not RAG chunks or tool observations.
  PII masking is regex-based (email, phone, IBAN, card); names or addresses need an extra
  recognizer. Answers are streamed before the output check: a masked answer is reprinted after
  `[réponse filtrée]`, so the raw text shows briefly (set `strict: true` to disable streaming).
- **Security**: side-effecting tools rely on human approval.
- **Interface**: the terminal is the only interface for now.
