"""Calibrates the LLM judge against human annotations (issue #72).

The judge (`nodes/eval_node.py::judge`) scores each answer on relevance,
coherence and faithfulness (0-5). Before trusting its scores, annotate a
small sample by hand and check how far the judge drifts from you.

Usage
-----
    python -m eval.calibration --sample eval/calibration_sample.yaml \
        --judge-model qwen2.5:7b --out eval/calibration.md

Protocol
--------
1. Load the sample (`items:` with `question`, `answer`, `sources` and
   `human: {relevance, coherence, faithfulness}`). Human scores are integers
   0-5, or `null` when not annotated.
2. Keep the annotated items (at least one human score not null).
3. Run the judge on each of them. A judge returning `None` or raising is a
   failure: it is counted, and the item is left out of the metrics.
4. Per criterion, compare judge and human on the pairs where both values are
   present: MAE, bias (mean judge - human), exact agreement, agreement within
   one point, and Spearman rho (stdlib, average ranks for ties).
5. Write a Markdown report (French) to `--out` and print it.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SAMPLE = Path(__file__).resolve().parent / "calibration_sample.yaml"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration.md"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "agent.yaml"
CRITERIA: tuple[str, ...] = ("relevance", "coherence", "faithfulness")

JudgeFn = Callable[[Any, str, str, list[str]], Awaitable[dict[str, Any] | None]]
Pair = tuple[dict[str, Any], dict[str, Any] | None]


def _score_or_none(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise ValueError(f"{where}: human score must be an integer 0-5 or null, got {value!r}")
    return value


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate `items:` from the sample YAML. Raises `ValueError` on bad shape."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("calibration sample must be a mapping with an 'items' list")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(data["items"], start=1):
        where = f"item {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: must be a mapping")
        question, answer = item.get("question"), item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{where}: needs a non-empty string 'question'")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{where}: needs a non-empty string 'answer'")
        sources = item.get("sources") or []
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError(f"{where}: 'sources' must be a list of strings")
        human_raw = item.get("human")
        if not isinstance(human_raw, dict):
            raise ValueError(f"{where}: needs a 'human' mapping")
        human = {
            name: _score_or_none(human_raw.get(name), f"{where}.human.{name}") for name in CRITERIA
        }
        items.append(
            {"question": question, "answer": answer, "sources": list(sources), "human": human}
        )
    return items


def annotated(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the items where at least one human score is not null."""
    return [item for item in items if any(item["human"].get(name) is not None for name in CRITERIA)]


async def run(items: list[dict[str, Any]], judge_llm: Any, judge_fn: JudgeFn) -> list[Pair]:
    """Run the judge on each item. Returns `(human, judge)` per item, judge `None` on failure."""
    pairs: list[Pair] = []
    for item in items:
        try:
            result = await judge_fn(judge_llm, item["question"], item["answer"], item["sources"])
        except Exception:  # a judge error is a failure, never a crash (CancelledError escapes)
            result = None
        pairs.append((item["human"], result))
    return pairs


def _average_ranks(values: list[float]) -> list[float]:
    """1-based ranks, tied values share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho: Pearson correlation of the average ranks. `None` if undefined."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs two sequences of the same length")
    if len(xs) < 2:
        return None
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def metrics(pairs: list[Pair]) -> dict[str, dict[str, Any]]:
    """Per criterion: n, MAE, bias, exact and within-1 agreement, Spearman.

    A pair is used for a criterion only when the judge returned a result and
    both the human and the judge scores for that criterion are present.
    """
    result: dict[str, dict[str, Any]] = {}
    for name in CRITERIA:
        human_scores: list[float] = []
        judge_scores: list[float] = []
        for human, judge in pairs:
            if judge is None:
                continue
            h, j = human.get(name), judge.get(name)
            if h is None or j is None:
                continue
            human_scores.append(h)
            judge_scores.append(j)

        n = len(human_scores)
        if n == 0:
            result[name] = {
                "n": 0,
                "mae": None,
                "bias": None,
                "exact": None,
                "within1": None,
                "spearman": None,
            }
            continue
        diffs = [j - h for h, j in zip(human_scores, judge_scores, strict=True)]
        result[name] = {
            "n": n,
            "mae": sum(abs(d) for d in diffs) / n,
            "bias": sum(diffs) / n,
            "exact": sum(1 for d in diffs if d == 0) / n,
            "within1": sum(1 for d in diffs if abs(d) <= 1) / n,
            "spearman": spearman(judge_scores, human_scores),
        }
    return result


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def _hint(name: str, stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return f"- **{name}** : aucune paire exploitable."
    bias = stats["bias"]
    if bias > 0.5:
        direction = "juge indulgent (il surnote par rapport à vous)"
    elif bias < -0.5:
        direction = "juge sévère (il sous-note par rapport à vous)"
    else:
        direction = "pas de biais systématique notable"
    spearman_part = (
        "corrélation de rang non calculable (variance nulle ou n < 2)"
        if stats["spearman"] is None
        else f"Spearman {stats['spearman']:.2f}"
    )
    return (
        f"- **{name}** : biais {bias:+.2f} ({direction}), "
        f"accord ±1 {_pct(stats['within1'])}, {spearman_part}."
    )


def render_report(
    metrics_by_criterion: dict[str, dict[str, Any]], failures: int, model: str
) -> str:
    """Markdown report in French: table per criterion plus interpretation hints."""
    lines = [
        "# Calibration du juge",
        "",
        f"Modèle juge : `{model}`",
        f"Échecs du juge (exclus des métriques) : {failures}",
        "",
        "| critère | n | MAE | biais (juge − humain) | accord exact | accord ±1 | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CRITERIA:
        stats = metrics_by_criterion[name]
        lines.append(
            f"| {name} | {stats['n']} | {_fmt(stats['mae'])} | {_fmt(stats['bias'])} "
            f"| {_pct(stats['exact'])} | {_pct(stats['within1'])} | {_fmt(stats['spearman'])} |"
        )
    lines += [
        "",
        "## Lecture",
        "",
        "- biais > 0 : le juge est indulgent ; biais < 0 : le juge est sévère.",
        "- MAE : écart moyen en points (0 à 5), plus il est bas, mieux c'est.",
        "- Spearman : 1 = même ordre de notes que vous, 0 = pas de lien, -1 = ordre inversé.",
        "",
    ]
    lines += [_hint(name, metrics_by_criterion[name]) for name in CRITERIA]
    return "\n".join(lines) + "\n"


def _default_llm_config(config: Path) -> dict[str, Any]:
    try:
        return dict(yaml.safe_load(config.read_text(encoding="utf-8")).get("llm", {}))
    except (OSError, AttributeError):
        return {}


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human scores.")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="annotated sample YAML")
    parser.add_argument(
        "--judge-model", default=None, help="judge model (default: config llm.model)"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="agent config (llm section)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Markdown report path")
    args = parser.parse_args(argv)

    items = load_items(args.sample)
    to_judge = annotated(items)
    if not to_judge:
        print(
            f"Aucun item annoté dans {args.sample} : remplissez les champs `human` "
            "(scores 0-5) avant de lancer la calibration."
        )
        return 1

    # Imported here so the module (and its tests) do not need the judge or the adapters.
    from app import build_llm
    from nodes.eval_node import judge

    llm_cfg = _default_llm_config(Path(args.config))
    if args.judge_model:
        llm_cfg["model"] = args.judge_model
    model = llm_cfg.get("model", "unknown")
    llm = build_llm(llm_cfg)
    async with llm:
        pairs = await run(to_judge, llm, judge)

    failures = sum(1 for _, judged in pairs if judged is None)
    report = render_report(metrics(pairs), failures, model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
