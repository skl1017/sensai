"""Unit tests for `ui/dashboard.py`, on synthetic event dicts (no storage layer needed)."""

from __future__ import annotations

import json

import pytest

from ui.dashboard import compute, main, percentile, render_html, render_text


def event(**fields) -> dict:
    base = {
        "status": "ok",
        "latency_ms": 100,
        "llm_calls": [],
        "tool_calls": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_hit": False,
        "blocked": False,
        "prompt_version": "tpl@v1",
        "eval_scores": None,
    }
    base.update(fields)
    return base


def sample_events() -> list[dict]:
    return [
        event(
            latency_ms=10,
            prompt_version="tpl@v1",
            prompt_tokens=100,
            completion_tokens=20,
            llm_calls=[{"ttft_ms": 5}, {"ttft_ms": 7}],
            tool_calls=[{"name": "calculator"}],
            eval_scores={"relevance": 4, "coherence": 5, "faithfulness": None},
        ),
        event(
            latency_ms=20,
            prompt_version="tpl@v1",
            prompt_tokens=50,
            completion_tokens=10,
            llm_calls=[{"ttft_ms": 9}],
            cache_hit=True,
            eval_scores={"relevance": 2, "coherence": None, "faithfulness": 3},
        ),
        event(
            latency_ms=30,
            status="error",
            prompt_version="tpl@v2",
            prompt_tokens=200,
            completion_tokens=40,
            blocked=True,
            tool_calls=[{"name": "calculator"}, {"name": "web_search"}],
            eval_scores="not a dict",
        ),
        event(
            latency_ms=40,
            status="cancelled",
            prompt_version=None,
            prompt_tokens=10,
            completion_tokens=None,
        ),
    ]


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile([3, 1, 2], 50) == 2
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None


def test_compute_volume_status_and_rates():
    stats = compute(sample_events())

    assert stats["total"] == 4
    assert stats["status"] == {"ok": 2, "error": 1, "cancelled": 1}
    assert stats["error_rate"] == pytest.approx(0.25)
    assert stats["cancelled_rate"] == pytest.approx(0.25)
    assert stats["cache_hit_rate"] == pytest.approx(0.25)
    assert stats["blocked_rate"] == pytest.approx(0.25)


def test_compute_latency_and_ttft_percentiles():
    stats = compute(sample_events())

    assert stats["latency_ms"] == {"p50": 20, "p95": 40}
    # ttft values: 5, 7, 9 (nearest-rank p50 = 7, p95 = 9)
    assert stats["llm_ttft_ms"] == {"p50": 7, "p95": 9}


def test_compute_tool_calls_by_name():
    assert compute(sample_events())["tool_calls"] == {"calculator": 2, "web_search": 1}


def test_compute_tokens_per_prompt_version():
    versions = compute(sample_events())["prompt_versions"]

    assert versions["tpl@v1"] == {
        "requests": 2,
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "mean_tokens_per_request": 90.0,
    }
    assert versions["tpl@v2"]["requests"] == 1
    assert versions["tpl@v2"]["mean_tokens_per_request"] == 240.0
    # None prompt_version becomes "unknown"; None completion_tokens counts as 0.
    assert versions["unknown"] == {
        "requests": 1,
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "mean_tokens_per_request": 10.0,
    }


def test_compute_eval_means_ignore_none_and_non_dict():
    evaluation = compute(sample_events())["eval"]

    # Only the two events with a dict eval_scores count; None values are skipped.
    assert evaluation["evaluated"] == 2
    assert evaluation["relevance"] == pytest.approx(3.0)
    assert evaluation["coherence"] == pytest.approx(5.0)
    assert evaluation["faithfulness"] == pytest.approx(3.0)


def test_compute_empty_list_gives_zeros_and_none():
    stats = compute([])

    assert stats["total"] == 0
    assert stats["status"] == {}
    assert stats["error_rate"] == 0.0
    assert stats["cache_hit_rate"] == 0.0
    assert stats["latency_ms"] == {"p50": None, "p95": None}
    assert stats["llm_ttft_ms"] == {"p50": None, "p95": None}
    assert stats["tool_calls"] == {}
    assert stats["prompt_versions"] == {}
    assert stats["eval"] == {
        "evaluated": 0,
        "relevance": None,
        "coherence": None,
        "faithfulness": None,
    }
    assert "requests" in render_text(stats)
    assert "<table>" in render_html(stats)


def test_compute_tolerates_missing_keys():
    stats = compute([{}, {"status": "error"}, {"llm_calls": [{}], "tool_calls": [{}]}])

    assert stats["total"] == 3
    assert stats["status"] == {"unknown": 2, "error": 1}
    assert stats["error_rate"] == pytest.approx(1 / 3)
    assert stats["latency_ms"] == {"p50": None, "p95": None}
    assert stats["llm_ttft_ms"] == {"p50": None, "p95": None}
    assert stats["tool_calls"] == {"unknown": 1}
    assert stats["prompt_versions"]["unknown"]["requests"] == 3
    assert stats["eval"]["evaluated"] == 0


def test_render_text_contains_key_numbers():
    text = render_text(compute(sample_events()))

    assert "requests" in text
    assert "25.0%" in text  # error / cancelled / cache / blocked rates
    assert "tpl@v1" in text
    assert "calculator" in text
    assert "web_search" in text
    assert "90.0" in text  # mean tokens per request for tpl@v1
    assert "unknown" in text


def test_render_html_escapes_prompt_version():
    stats = compute([event(prompt_version="<script>alert(1)</script>")])

    page = render_html(stats)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert page.startswith("<!doctype html>")
    assert "<style>" in page


def test_main_missing_file_exits_1(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl"

    assert main([str(missing)]) == 1
    assert "not found" in capsys.readouterr().err


def test_main_reads_events_file(tmp_path, capsys):
    read_events_module = pytest.importorskip("storage.event_log")
    assert hasattr(read_events_module, "read_events")

    source = tmp_path / "events.jsonl"
    lines = [
        json.dumps(e)
        for e in (
            event(latency_ms=10, prompt_version="tpl@v1", prompt_tokens=5, completion_tokens=5),
            event(latency_ms=30, status="error", prompt_version="tpl@v1"),
        )
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert main([str(source)]) == 0
    assert "requests" in capsys.readouterr().out

    out = tmp_path / "dashboard.html"
    assert main([str(source), "--html", str(out)]) == 0
    assert "<!doctype html>" in out.read_text(encoding="utf-8")
