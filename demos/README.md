# Feature demos

One terminal recording per feature, scripted with [VHS](https://github.com/charmbracelet/vhs):
`<name>.tape` is the script, `<name>.gif` / `<name>.mp4` the result. They are the keynote's
fallback for anything that depends on the network or a slow model.

| Feature | Demo |
|---|---|
| B1 CLI chatbot (streaming, history, errors, model from config) | `b1-cli-chatbot` |
| M1 Session persistence & profile | `m1-sessions-profile` |
| M2 Token budget & summarization | `m2-token-budget` |
| M3 Structured memory (SQLite) | `m3-structured-memory` |
| T1 MCP (filesystem server + our own server) | `t1-mcp` |
| T2 Sandboxed code execution (Docker) | `t2-code-sandbox` |
| T3 Web search (SearXNG) | `t3-web-search` |
| T4 File access within permissions | `t4-file-permissions` |
| R1 Basic RAG with citations | `r1-basic-rag` |
| R2 Advanced RAG (rerank, threshold, filters from the question) | `r2-advanced-rag` |
| A1 ReAct loop | `a1-react-loop` |
| A2 Multi-agent (tutor sub-agent) | `a2-multi-agent` |
| A3 Human-in-the-loop | `a3-human-in-the-loop` |
| A5 Persona switch mid-session | `a5-persona` |
| A6 Semantic cache | `a6-semantic-cache` |
| A7 Few-shot selection & query rewrite | `a7-prompt-optimization` |
| EV2 Input & output guardrails | `ev2-guardrails` |
| X2 Branching & interrupt | `x2-branching-interrupt` |
| X4 Questions to the user | `x4-ask-user` |

## Re-recording

```bash
demos/record.sh                   # every demo
demos/record.sh a6-semantic-cache # just this one
```

Requirements: `vhs` (with `ttyd` and `ffmpeg`) on the `PATH`, the project's `.venv`,
Ollama and SearXNG running (`docker compose up -d ollama searxng`), Docker for `t2`, `npx` for
`t1`. The demos use your `config/agent.yaml` (model, host): answers vary from one run to the
next, so check the result before replacing a video.

## How it works

- Each demo runs in a fresh `demos/.work/` (gitignored): a copy of `config/`, its own
  `data.db`, `sessions/` and profile. Your real state is never read or written.
- `<name>.setup.sh` (optional) prepares that workspace: config tweaks through `cfg` from
  `_lib.sh`, fixture documents from `fixtures/rag/` (a fictional company), a seeded session.
- `_settings.tape` holds the shared look and the hidden shell setup.
- `sensai` in the recordings is `demos/sensai_demo.py`: the real `ui.cli`, plus, when
  `SENSAI_SHOW_STATE` is set, a dim `[state]` block under each answer showing `ctx.state`
  keys that the CLI doesn't print (cache hit, context stats, RAG filters and scores, few-shot
  examples). Nothing in the application is modified.
