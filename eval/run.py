"""Offline evaluation bench: compare agent configurations with an LLM judge (issue #70).

Each configuration of `eval/configs.yaml` (agent yaml + optional `prompt_version`
and `model` overrides) answers every question of `eval/questions.yaml`; the
answers are scored by the judge (`nodes.eval_node.judge`) on relevance,
coherence and faithfulness. Per configuration the report gives the mean of each
score, the number of judge failures, the mean latency and the mean tokens.

Usage
-----
    python -m eval.run --questions eval/questions.yaml --configs eval/configs.yaml \\
        --judge-model qwen2.5:7b --out eval/results

Writes `<out>/<timestamp>.json` (raw results) and `<out>/latest.md` (tables).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml

from core.llm import ILLM

SCORES = ("relevance", "coherence", "faithfulness")
DEFAULT_QUESTIONS = Path("eval/questions.yaml")
DEFAULT_CONFIGS = Path("eval/configs.yaml")
DEFAULT_OUT = Path("eval/results")

JudgeFn = Callable[[ILLM, str, str, list[str]], Awaitable[dict | None]]
AgentFactory = Callable[[dict], Awaitable["object"]]


class BenchUI:
    """Non-interactive UI: drops tokens/events, refuses every approval."""

    def on_token(self, text: str) -> None:
        return None

    def on_event(self, kind: str, data: dict) -> None:
        return None

    async def ask(self, question: str, kind: str = "confirm", options: list[str] | None = None):
        return "no"


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for item in data:
        if isinstance(item, str):
            item = {"question": item}
        if not item.get("question"):
            raise ValueError(f"{path}: every entry needs a 'question'")
        out.append({"question": item["question"], "sources": list(item.get("sources") or [])})
    return out


def load_configs(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    for item in data:
        if not item.get("name") or not item.get("config"):
            raise ValueError(f"{path}: every entry needs 'name' and 'config'")
    return data


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    """Per-configuration summary of per-question rows (None scores are ignored)."""
    summary: dict = {"n": len(rows)}
    for key in SCORES:
        summary[key] = _mean(
            [r["scores"][key] for r in rows if r.get("scores") and r["scores"].get(key) is not None]
        )
    summary["judge_failures"] = sum(1 for r in rows if not r.get("scores"))
    summary["errors"] = sum(1 for r in rows if r.get("error"))
    summary["latency_ms"] = _mean([r["latency_ms"] for r in rows])
    summary["tokens"] = _mean([r["tokens"] for r in rows if r.get("tokens") is not None])
    return summary


def _tokens(state: dict) -> int | None:
    metrics = state.get("metrics")
    if not metrics:
        return None
    return (metrics.get("prompt_tokens") or 0) + (metrics.get("completion_tokens") or 0)


async def run_config(
    entry: dict,
    questions: list[dict],
    agent_factory: AgentFactory,
    judge_fn: JudgeFn,
    judge_llm: ILLM,
) -> dict:
    """Run every question through the agent built for `entry`, judge each answer."""
    agent = await agent_factory(entry)
    rows: list[dict] = []
    try:
        for q in questions:
            row: dict = {"question": q["question"], "response": None, "error": None}
            sources = list(q["sources"])
            start = time.perf_counter()
            try:
                ctx = await agent.run(q["question"], [], state={})
            except Exception as exc:  # one failing question must not sink the run
                row.update(latency_ms=(time.perf_counter() - start) * 1000, tokens=None)
                row.update(error=f"{type(exc).__name__}: {exc}", scores=None)
                rows.append(row)
                continue
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            row["response"] = ctx.response
            row["tokens"] = _tokens(ctx.state)
            sources += [
                c["text"]
                for c in ctx.state.get("retrieved_chunks") or []
                if isinstance(c, dict) and c.get("text")
            ]
            row["sources"] = sources
            try:
                verdict = await judge_fn(judge_llm, q["question"], ctx.response or "", sources)
            except Exception:  # judge failure counts, CancelledError still propagates
                verdict = None
            row["scores"] = verdict
            rows.append(row)
    finally:
        close = getattr(agent, "aclose", None)
        if close is not None:
            await close()
    return {"name": entry["name"], "entry": entry, "rows": rows, "summary": aggregate(rows)}


async def run_bench(
    questions: list[dict],
    configs: list[dict],
    agent_factory: AgentFactory,
    judge_fn: JudgeFn,
    judge_llm: ILLM,
) -> dict:
    results = [
        await run_config(entry, questions, agent_factory, judge_fn, judge_llm) for entry in configs
    ]
    return {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "configs": results}


def _fmt(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(result: dict) -> str:
    """Markdown: one row per configuration, then one table per question."""
    lines = [
        "# Offline evaluation",
        "",
        f"Generated by `python -m eval.run` ({result['timestamp']}). "
        "Scores 0-5 judged by an LLM; faithfulness ignores questions without sources.",
        "",
        "## Configurations",
        "",
        "| config | n | relevance | coherence | faithfulness | judge failures "
        "| errors | latency ms | tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result["configs"]:
        s = c["summary"]
        lines.append(
            f"| {c['name']} | {s['n']} | {_fmt(s['relevance'], 2)} | {_fmt(s['coherence'], 2)} "
            f"| {_fmt(s['faithfulness'], 2)} | {s['judge_failures']} | {s['errors']} "
            f"| {_fmt(s['latency_ms'])} | {_fmt(s['tokens'])} |"
        )
    cfgs = result["configs"]
    if len(cfgs) == 2:
        a, b = cfgs[0]["summary"], cfgs[1]["summary"]
        diffs = []
        for key in SCORES:
            d = None if a[key] is None or b[key] is None else b[key] - a[key]
            diffs.append(f"{key} {'—' if d is None else f'{d:+.2f}'}")
        lines += ["", f"Difference ({cfgs[1]['name']} - {cfgs[0]['name']}): " + ", ".join(diffs)]

    lines += ["", "## Per question", ""]
    names = [c["name"] for c in cfgs]
    lines += [
        "| question | " + " | ".join(f"{n} (rel/coh/faith)" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for i, row in enumerate(cfgs[0]["rows"] if cfgs else []):
        cells = []
        for c in cfgs:
            r = c["rows"][i]
            sc = r.get("scores")
            if r.get("error"):
                cells.append("error")
            elif not sc:
                cells.append("judge failed")
            else:
                cells.append("/".join(_fmt(sc.get(k)) for k in SCORES))
        q = row["question"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_results(result: dict, out_dir: Path) -> tuple[Path, Path]:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_dir / "latest.md"
    md_path.write_text(render_report(result), encoding="utf-8")
    return json_path, md_path


def _llm_section(config_path: str) -> dict:
    from config_loader import load_config

    return load_config(config_path).llm.as_dict()


def make_judge_llm(config_path: str, judge_model: str | None) -> ILLM:
    from app import build_llm

    cfg = _llm_section(config_path)
    if judge_model:
        cfg["model"] = judge_model
    cfg["options"] = {**(cfg.get("options") or {}), "temperature": 0}
    return build_llm(cfg)


async def default_agent_factory(entry: dict):
    from app import build_agent, build_llm

    llm = None
    if entry.get("model"):
        cfg = _llm_section(entry["config"])
        cfg["model"] = entry["model"]
        llm = build_llm(cfg)
    return await build_agent(
        entry["config"], BenchUI(), llm=llm, prompt_version=entry.get("prompt_version")
    )


async def main(args: argparse.Namespace) -> int:
    from nodes.eval_node import judge

    questions = load_questions(args.questions)
    configs = load_configs(args.configs)
    judge_llm = make_judge_llm(configs[0]["config"], args.judge_model)
    result = await run_bench(questions, configs, default_agent_factory, judge, judge_llm)
    json_path, md_path = write_results(result, args.out)
    print(render_report(result))
    print(f"Wrote {json_path} and {md_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    p.add_argument("--judge-model", help="judge model (default: llm.model of the first config)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
