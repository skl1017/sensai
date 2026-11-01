"""Metrics dashboard over the request event log (issue #71).

`compute` turns a list of event dicts (see the event contract in `storage/event_log.py`)
into a plain stats dict. It is pure: it never reads files, and it tolerates missing keys.
`render_text` and `render_html` only format those stats.

CLI
---
    python -m ui.dashboard [logs/events.jsonl]              # text tables on stdout
    python -m ui.dashboard logs/events.jsonl --html out.html

The events are read through `storage.event_log.read_events`, imported lazily so that
`compute` and the renderers stay usable without the storage layer.
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = "logs/events.jsonl"
EVAL_DIMENSIONS = ("relevance", "coherence", "faithfulness")
UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------


def percentile(values: list[float], p: int) -> float | None:
    """Nearest-rank percentile (`p` in 0..100) of `values`; `None` when empty."""
    if not values:
        return None
    ordered = sorted(values)
    # Integer arithmetic avoids float error on exact ranks (e.g. 95% of 20 = 19).
    rank = -(-p * len(ordered) // 100)  # ceil(p * n / 100)
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def _num(value: object) -> float | None:
    """`value` as a float when it is a real number, otherwise `None` (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p50_p95(values: list[float]) -> dict:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def compute(events: list[dict]) -> dict:
    """Indicators over `events`. Empty input gives zero counts, rates and `None` latencies."""
    total = len(events)
    status = Counter(e.get("status") or UNKNOWN for e in events)
    cache_hits = sum(1 for e in events if e.get("cache_hit"))
    blocked = sum(1 for e in events if e.get("blocked"))

    latencies = [v for e in events if (v := _num(e.get("latency_ms"))) is not None]
    ttfts = [
        v
        for e in events
        for call in (e.get("llm_calls") or [])
        if (v := _num(call.get("ttft_ms"))) is not None
    ]
    tools = Counter(
        call.get("name") or UNKNOWN for e in events for call in (e.get("tool_calls") or [])
    )

    versions: dict[str, dict] = {}
    for e in events:
        key = e.get("prompt_version") or UNKNOWN
        row = versions.setdefault(key, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        row["requests"] += 1
        row["prompt_tokens"] += int(_num(e.get("prompt_tokens")) or 0)
        row["completion_tokens"] += int(_num(e.get("completion_tokens")) or 0)
    for row in versions.values():
        row["mean_tokens_per_request"] = (row["prompt_tokens"] + row["completion_tokens"]) / row[
            "requests"
        ]

    scored = [e["eval_scores"] for e in events if isinstance(e.get("eval_scores"), dict)]
    evaluation: dict = {"evaluated": len(scored)}
    for dim in EVAL_DIMENSIONS:
        values = [v for s in scored if (v := _num(s.get(dim))) is not None]
        evaluation[dim] = _mean(values)

    return {
        "total": total,
        "status": dict(status),
        "error_rate": _rate(status.get("error", 0), total),
        "cancelled_rate": _rate(status.get("cancelled", 0), total),
        "cache_hit_rate": _rate(cache_hits, total),
        "blocked_rate": _rate(blocked, total),
        "latency_ms": _p50_p95(latencies),
        "llm_ttft_ms": _p50_p95(ttfts),
        "tool_calls": dict(tools),
        "prompt_versions": versions,
        "eval": evaluation,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _overview_rows(stats: dict) -> list[tuple[str, str]]:
    latency, ttft = stats["latency_ms"], stats["llm_ttft_ms"]
    return [
        ("requests", str(stats["total"])),
        ("error rate", _pct(stats["error_rate"])),
        ("cancelled rate", _pct(stats["cancelled_rate"])),
        ("cache hit rate", _pct(stats["cache_hit_rate"])),
        ("blocked rate", _pct(stats["blocked_rate"])),
        ("latency p50 (ms)", _ms(latency["p50"])),
        ("latency p95 (ms)", _ms(latency["p95"])),
        ("llm ttft p50 (ms)", _ms(ttft["p50"])),
        ("llm ttft p95 (ms)", _ms(ttft["p95"])),
    ]


def _text_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Aligned plain-text table: first column left-aligned, the others right-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]

    def line(cells: list[str]) -> str:
        parts = [cells[0].ljust(widths[0])] + [
            cell.rjust(width) for cell, width in zip(cells[1:], widths[1:], strict=True)
        ]
        return "  ".join(parts).rstrip()

    out = [title, line(headers), line(["-" * w for w in widths])]
    out += [line(row) for row in rows]
    return "\n".join(out)


def _sections(stats: dict) -> list[tuple[str, list[str], list[list[str]]]]:
    """(title, headers, rows) for every table, shared by the text and HTML renderers."""
    sections = [
        ("Overview", ["indicator", "value"], [list(r) for r in _overview_rows(stats)]),
        (
            "Status",
            ["status", "count"],
            [[name, str(count)] for name, count in sorted(stats["status"].items())],
        ),
        (
            "Tool calls",
            ["tool", "calls"],
            [[name, str(count)] for name, count in sorted(stats["tool_calls"].items())],
        ),
        (
            "Prompt versions",
            ["prompt_version", "requests", "prompt tokens", "completion tokens", "mean tok/req"],
            [
                [
                    version,
                    str(row["requests"]),
                    str(row["prompt_tokens"]),
                    str(row["completion_tokens"]),
                    f"{row['mean_tokens_per_request']:.1f}",
                ]
                for version, row in sorted(stats["prompt_versions"].items())
            ],
        ),
        (
            "Eval (mean scores)",
            ["metric", "value"],
            [["evaluated", str(stats["eval"]["evaluated"])]]
            + [[dim, _score(stats["eval"][dim])] for dim in EVAL_DIMENSIONS],
        ),
    ]
    return sections


def render_text(stats: dict) -> str:
    """Aligned plain-text tables, one per section."""
    return "\n\n".join(
        _text_table(title, headers, rows) for title, headers, rows in _sections(stats)
    )


_HTML_STYLE = """
:root { --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --line: #ddd; --head: #efefef; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #161616; --fg: #eaeaea; --muted: #999;
    --line: #333; --head: #222; }
}
:root[data-theme="dark"] { --bg: #161616; --fg: #eaeaea; --muted: #999; --line: #333;
  --head: #222; }
body { background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif;
  margin: 2rem; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; }
table { border-collapse: collapse; min-width: 320px; }
th, td { border: 1px solid var(--line); padding: 0.35rem 0.7rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: var(--head); }
"""


def _html_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"
    )


def render_html(stats: dict) -> str:
    """Single standalone HTML page with inline CSS; every value is escaped."""
    sections = "\n".join(
        _html_table(title, headers, rows) for title, headers, rows in _sections(stats)
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Sensai metrics</title>\n"
        f"<style>{_HTML_STYLE}</style>\n</head>\n<body>\n"
        "<h1>Sensai metrics</h1>\n"
        f"{sections}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ui.dashboard", description="Sensai metrics dashboard."
    )
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="events file (.jsonl/.db)")
    parser.add_argument("--html", type=Path, metavar="OUT", help="write a standalone HTML page")
    args = parser.parse_args(argv)

    source = Path(args.path)
    if not source.is_file():
        print(f"error: events file not found: {source}", file=sys.stderr)
        return 1

    # Lazy import: only the CLI needs the storage layer.
    from storage.event_log import read_events

    stats = compute(read_events(str(source)))
    if args.html:
        args.html.write_text(render_html(stats), encoding="utf-8")
        print(f"wrote {args.html}")
    else:
        print(render_text(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
