"""Local model benchmark: latency, tokens/s, RAM/VRAM (issue #118).

Measures the models themselves, straight against Ollama's HTTP API, without
going through the `Agent` pipeline (context building, RAG, ReAct would blur
the numbers). Each group member runs it on their own machine; every run
writes `bench/<hostname>.json`, and `bench/results.md` is rebuilt from all
the JSON files found, so the machines end up in one comparison table.

Usage
-----
    python -m eval.bench                          # model from config/agent.yaml
    python -m eval.bench -m qwen2.5:3b -m qwen2.5:7b --runs 5 --pull
    python -m eval.bench --report-only            # rebuild results.md only

Protocol
--------
For each model:

1. Unload every loaded model (`keep_alive: 0`) so the first call starts cold.
2. Cold call: its `load_duration` is the model load time.
3. `GET /api/ps`: total size (`size`), part in VRAM (`size_vram`), % GPU.
4. For each prompt (`short`, `long` ~2,000 tokens, `tool` with the calculator
   spec), `--runs` measured calls with `temperature: 0`, a fixed `seed`,
   `num_ctx` and `num_predict`. Each call starts with a system message
   carrying a different run tag, so Ollama's prompt prefix cache can't skip
   the prompt evaluation (including the tool specs, rendered before the
   user message).
5. Unload the model.

Per call: TTFT (client clock, first chunk carrying text or tool calls), total
latency (client clock), generation tokens/s (`eval_count / eval_duration`)
and prompt tokens/s (`prompt_eval_count / prompt_eval_duration`). The JSON
keeps every raw sample; the report shows median and p95.

RAM/VRAM come from `/api/ps`, which works whether Ollama runs natively or in
Docker. GPU details (`nvidia-smi`) are only reported when the tool is on the
host's PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from tools.calculator import CalculatorTool

DEFAULT_CONFIG = Path("config/agent.yaml")
DEFAULT_OUT = Path("bench")
METRICS = ("ttft_ms", "latency_ms", "gen_tok_s", "prompt_tok_s")

_LONG_PARAGRAPH = (
    "Sensai is a pure-Python AI agent built around a minimal core: an Agent, "
    "an LLM interface, a tool interface and pipeline nodes. Every feature, "
    "from persistence to retrieval-augmented generation, plugs in as a node "
    "or a tool configured from a YAML file. "
)


def _prompts() -> dict[str, dict]:
    calculator = CalculatorTool()
    tool_spec = {
        "type": "function",
        "function": {
            "name": calculator.name,
            "description": calculator.description,
            "parameters": calculator.parameters,
        },
    }
    return {
        "short": {"content": "Explain in three sentences what a hash table is."},
        "long": {
            # ~50 tokens per paragraph, so ~2,000 tokens of context.
            "content": _LONG_PARAGRAPH * 40 + "\nSummarize the text above in five bullet points.",
        },
        "tool": {"content": "How much is 1234 * 5678 - 91?", "tools": [tool_spec]},
    }


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return count / duration_ns * 1e9


async def run_once(
    client: httpx.AsyncClient, model: str, prompt: dict, options: dict, tag: str
) -> dict:
    """One streamed `/api/chat` call; returns its raw measurements."""
    payload: dict = {
        "model": model,
        "stream": True,
        "options": options,
        # The tag goes in a system message: templates render it first, before
        # the tool specs, so no part of the prompt is served from the cache.
        "messages": [
            {"role": "system", "content": f"Benchmark {tag}."},
            {"role": "user", "content": prompt["content"]},
        ],
    }
    if prompt.get("tools"):
        payload["tools"] = prompt["tools"]

    start = time.perf_counter()
    ttft = None
    last: dict = {}
    async with client.stream("POST", "/api/chat", json=payload) as r:
        if r.status_code >= 400:
            body = (await r.aread()).decode("utf-8", errors="replace")
            raise RuntimeError(f"{model}: HTTP {r.status_code}: {body[:200]}")
        async for line in r.aiter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(f"{model}: {data['error']}")
            msg = data.get("message", {})
            if ttft is None and (msg.get("content") or msg.get("tool_calls")):
                ttft = time.perf_counter() - start
            if data.get("done"):
                last = data
    latency = time.perf_counter() - start

    return {
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "latency_ms": latency * 1000,
        "load_ms": last.get("load_duration", 0) / 1e6,
        "prompt_tokens": last.get("prompt_eval_count"),
        "completion_tokens": last.get("eval_count"),
        "prompt_tok_s": _rate(last.get("prompt_eval_count"), last.get("prompt_eval_duration")),
        "gen_tok_s": _rate(last.get("eval_count"), last.get("eval_duration")),
    }


async def memory(client: httpx.AsyncClient, model: str) -> dict:
    """Size of `model` once loaded, from `/api/ps`."""
    r = await client.get("/api/ps")
    r.raise_for_status()
    for m in r.json().get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            return {
                "size_gb": size / 1e9,
                "vram_gb": vram / 1e9,
                "gpu_pct": round(vram / size * 100) if size else 0,
            }
    return {"size_gb": None, "vram_gb": None, "gpu_pct": None}


async def unload_all(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/ps")
    r.raise_for_status()
    for m in r.json().get("models", []):
        await client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})


async def ensure_model(client: httpx.AsyncClient, model: str, pull: bool) -> bool:
    r = await client.post("/api/show", json={"model": model})
    if r.status_code == 200:
        return True
    if not pull:
        print(f"  {model}: not installed, skipped (run `ollama pull {model}` or use --pull)")
        return False
    print(f"  {model}: pulling…")
    r = await client.post("/api/pull", json={"model": model, "stream": False}, timeout=None)
    return r.status_code == 200


async def bench_model(
    client: httpx.AsyncClient, model: str, prompts: dict[str, dict], runs: int, options: dict
) -> dict:
    await unload_all(client)
    cold = await run_once(client, model, prompts["short"], options, tag="cold")
    result: dict = {
        "model": model,
        "load_ms": cold["load_ms"],
        "memory": await memory(client, model),
        "prompts": {},
    }
    for name, prompt in prompts.items():
        samples = []
        for i in range(runs):
            samples.append(await run_once(client, model, prompt, options, tag=f"run {i}"))
            print(f"  {model} {name} {i + 1}/{runs}", end="\r", flush=True)
        result["prompts"][name] = {"samples": samples, "summary": summarize(samples)}
    print(" " * 60, end="\r")
    await unload_all(client)
    return result


# --------------------------------------------------------------------------
# Aggregation and machine info
# --------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (`p` in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(samples: list[dict]) -> dict:
    """Median and p95 of each metric, ignoring missing (`None`) values."""
    summary = {}
    for metric in METRICS:
        values = [s[metric] for s in samples if s.get(metric) is not None]
        summary[metric] = (
            {"p50": statistics.median(values), "p95": percentile(values, 95)} if values else None
        )
    return summary


def _ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _gpu() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or None


async def machine_info(client: httpx.AsyncClient, label: str | None) -> dict:
    r = await client.get("/api/version")
    return {
        "hostname": label or socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "ollama": r.json().get("version") if r.status_code == 200 else None,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(results: list[dict]) -> str:
    """Markdown comparison table built from every machine's JSON result."""
    lines = [
        "# Benchmarks locaux des modèles",
        "",
        "Généré par `python -m eval.bench`. Médianes sur les runs mesurés (p95 entre parenthèses).",
        "",
        "## Machines",
        "",
        "| Machine | CPU | Cœurs | RAM (Go) | GPU | Ollama | Date |",
        "|---|---|---|---|---|---|---|",
    ]
    for res in results:
        m = res["machine"]
        lines.append(
            f"| {m['hostname']} | {m['cpu']} | {m['cores']} | {_fmt(m['ram_gb'], 1)} "
            f"| {m['gpu'] or '—'} | {m['ollama'] or '—'} | {res['date'][:10]} |"
        )
    lines += [
        "",
        "## Résultats",
        "",
        "| Machine | Modèle | Prompt | Load (s) | TTFT (ms) | Latence (ms) | Gen tok/s "
        "| Prompt tok/s | Taille (Go) | VRAM (Go) | % GPU |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        host = res["machine"]["hostname"]
        for model in res["models"]:
            mem = model["memory"]
            for name, prompt in model["prompts"].items():
                s = prompt["summary"]
                cells = []
                for metric in METRICS:
                    v = s.get(metric)
                    cells.append("—" if v is None else f"{_fmt(v['p50'])} ({_fmt(v['p95'])})")
                lines.append(
                    f"| {host} | {model['model']} | {name} | {_fmt(model['load_ms'] / 1000, 1)} "
                    f"| {' | '.join(cells)} | {_fmt(mem['size_gb'], 1)} "
                    f"| {_fmt(mem['vram_gb'], 1)} | {_fmt(mem['gpu_pct'])} |"
                )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path) -> Path:
    results = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    path = out_dir / "results.md"
    path.write_text(render_report(results), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _default_llm(config: Path) -> dict:
    try:
        return yaml.safe_load(config.read_text()).get("llm", {})
    except (OSError, AttributeError):
        return {}


async def main(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        print(f"report: {write_report(args.out)}")
        return 0

    llm = _default_llm(args.config)
    host = args.host or llm.get("host", "http://localhost:11434")
    models = args.model or [llm.get("model", "qwen2.5:3b")]
    options = {
        "temperature": 0,
        "seed": 42,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }
    prompts = _prompts()

    async with httpx.AsyncClient(base_url=host, timeout=None) as client:
        try:
            machine = await machine_info(client, args.label)
        except httpx.TransportError:
            print(f"Ollama unreachable at {host}: is `ollama serve` running?", file=sys.stderr)
            return 1
        print(f"machine: {machine['hostname']} — {machine['cpu']} — GPU: {machine['gpu']}")
        results = []
        for model in models:
            if not await ensure_model(client, model, args.pull):
                continue
            print(f"benchmarking {model}…")
            results.append(await bench_model(client, model, prompts, args.runs, options))

    if not results:
        print("no model benchmarked", file=sys.stderr)
        return 1
    out = args.out / f"{machine['hostname']}.json"
    out.write_text(
        json.dumps(
            {
                "date": datetime.now(UTC).isoformat(),
                "machine": machine,
                "options": options,
                "runs": args.runs,
                "models": results,
            },
            indent=2,
        )
    )
    print(f"results: {out}\nreport: {write_report(args.out)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-m", "--model", action="append", help="model to bench (repeatable)")
    p.add_argument("--host", help="Ollama URL (default: llm.host from the config)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--runs", type=int, default=5, help="measured calls per prompt")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--num-predict", type=int, default=128, help="max generated tokens")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", help="machine name in the report (default: hostname)")
    p.add_argument("--pull", action="store_true", help="pull missing models")
    p.add_argument("--report-only", action="store_true", help="only rebuild results.md")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
