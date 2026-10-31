"""Calibrates the cosine threshold of the semantic response cache (issue #61).

The cache serves a stored answer when the embedding of a new question is at
least `threshold` similar (cosine) to a previously answered question. A false
positive (a different question served the wrong answer) is much worse than a
false negative (one extra LLM call), so the recommendation favours zero FP.

Usage
-----
    python -m eval.cache_threshold
    python -m eval.cache_threshold --model nomic-embed-text --host http://localhost:11434
    python -m eval.cache_threshold --pairs eval/cache_pairs.yaml

Protocol
--------
1. Load the labelled pairs (`eval/cache_pairs.yaml`: `{a, b, same}`).
2. Embed every question in one batch with `OllamaEmbedder` and score each pair
   with the cosine similarity (`_cosine`, shared with the few-shot section).
3. For each threshold from 0.80 to 0.99 (step 0.01), a pair is predicted
   "same" when `score >= threshold`. Count false positives (predicted same,
   labelled different) and false negatives (predicted different, labelled
   same). Precision is 1.0 when nothing is predicted same; recall is 0.0 when
   there is no `same: true` pair.
4. Recommend the lowest threshold with zero false positives (ties broken by
   fewest false negatives). If no threshold reaches zero FP, take the one with
   the fewest FP, then the fewest FN.
5. Print the sweep as a markdown table, every pair's score sorted from highest
   to lowest (to see the overlap between the two classes), and the
   recommendation.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

from core.llm import IEmbedder
from llm.ollama import OllamaEmbedder
from nodes.fewshot_section import _cosine

DEFAULT_PAIRS = Path(__file__).resolve().parent / "cache_pairs.yaml"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
THRESHOLDS: list[float] = [round(0.80 + i * 0.01, 2) for i in range(20)]


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate `pairs:` from a YAML file. Raises `ValueError` on bad shape."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
        raise ValueError("cache pairs file must be a mapping with a 'pairs' list")

    pairs: list[dict[str, Any]] = []
    for item in data["pairs"]:
        if not isinstance(item, dict):
            raise ValueError("each pair must be a mapping")
        a, b, same = item.get("a"), item.get("b"), item.get("same")
        if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip():
            raise ValueError("each pair needs non-empty string 'a' and 'b'")
        if not isinstance(same, bool):
            raise ValueError("each pair needs a boolean 'same'")
        pairs.append({"a": a, "b": b, "same": same})
    return pairs


async def score_pairs(embedder: IEmbedder, pairs: list[dict[str, Any]]) -> list[float]:
    """Cosine similarity of each pair, embedding all questions in one batch."""
    if not pairs:
        return []
    texts: list[str] = []
    for pair in pairs:
        texts.extend((pair["a"], pair["b"]))
    vectors = await embedder.embed(texts)
    return [_cosine(vectors[2 * i], vectors[2 * i + 1]) for i in range(len(pairs))]


def sweep(scores: list[float], labels: list[bool], thresholds: list[float]) -> list[dict[str, Any]]:
    """Confusion counts per threshold: a pair is 'same' when `score >= threshold`."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        fp = fn = tp = 0
        for score, label in zip(scores, labels, strict=True):
            predicted = score >= threshold
            if predicted and not label:
                fp += 1
            elif predicted and label:
                tp += 1
            elif not predicted and label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": threshold,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest threshold with zero FP (ties: fewest FN); else fewest FP, then fewest FN."""
    if not rows:
        raise ValueError("cannot recommend a threshold from an empty sweep")

    zero_fp = [row for row in rows if row["fp"] == 0]
    if zero_fp:
        return min(zero_fp, key=lambda row: (row["threshold"], row["fn"]))
    return min(rows, key=lambda row: (row["fp"], row["fn"], row["threshold"]))


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| threshold | FP | FN | precision | recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['fp']} | {row['fn']} "
            f"| {row['precision']:.3f} | {row['recall']:.3f} |"
        )
    return "\n".join(lines)


def _report(pairs: list[dict[str, Any]], scores: list[float], rows: list[dict[str, Any]]) -> str:
    out = [_markdown_table(rows), "", "Scores, highest first:", ""]
    out.append("| score | same | a | b |")
    out.append("|---:|:---:|---|---|")
    ranked = sorted(zip(scores, pairs, strict=True), key=lambda item: item[0], reverse=True)
    for score, pair in ranked:
        out.append(
            f"| {score:.4f} | {'yes' if pair['same'] else 'no'} | {pair['a']} | {pair['b']} |"
        )

    same_scores = [s for s, p in zip(scores, pairs, strict=True) if p["same"]]
    diff_scores = [s for s, p in zip(scores, pairs, strict=True) if not p["same"]]
    out.append("")
    if same_scores:
        out.append(f"min score, same pairs: {min(same_scores):.4f}")
    if diff_scores:
        out.append(f"max score, different pairs: {max(diff_scores):.4f}")

    best = recommend(rows)
    out.append(
        f"recommended threshold: {best['threshold']:.2f} "
        f"(FP={best['fp']}, FN={best['fn']}, precision={best['precision']:.3f}, "
        f"recall={best['recall']:.3f})"
    )
    return "\n".join(out)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the semantic cache threshold.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama base URL")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="labelled pairs YAML")
    args = parser.parse_args(argv)

    pairs = load_pairs(args.pairs)
    async with OllamaEmbedder(model=args.model, host=args.host) as embedder:
        scores = await score_pairs(embedder, pairs)

    rows = sweep(scores, [p["same"] for p in pairs], THRESHOLDS)
    print(f"model={args.model} pairs={len(pairs)}")
    print(_report(pairs, scores, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
