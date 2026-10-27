"""Unit tests for `eval/bench.py`, against `httpx.MockTransport` (no Ollama)."""

from __future__ import annotations

import json

import httpx
import pytest

from eval.bench import memory, percentile, render_report, run_once, summarize

HOST = "http://ollama.test"


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=HOST, transport=httpx.MockTransport(handler))


def ndjson(*lines: dict) -> bytes:
    return "\n".join(json.dumps(line) for line in lines).encode()


async def test_run_once_reads_ollama_counters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "Hel"}, "done": False},
                {"message": {"content": "lo"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "load_duration": 2_000_000_000,
                    "prompt_eval_count": 50,
                    "prompt_eval_duration": 500_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                },
            ),
        )

    async with client_for(handler) as client:
        result = await run_once(client, "m", {"content": "hi"}, {"num_ctx": 2048}, tag="run 0")

    assert seen["options"] == {"num_ctx": 2048}
    assert seen["messages"] == [
        {"role": "system", "content": "Benchmark run 0."},
        {"role": "user", "content": "hi"},
    ]
    assert result["load_ms"] == pytest.approx(2000)
    assert result["prompt_tok_s"] == pytest.approx(100)
    assert result["gen_tok_s"] == pytest.approx(20)
    assert result["completion_tokens"] == 20
    assert result["ttft_ms"] is not None
    assert result["ttft_ms"] <= result["latency_ms"]


async def test_run_once_counts_tool_call_as_first_token():
    def handler(request: httpx.Request) -> httpx.Response:
        call = {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "", "tool_calls": [call]}, "done": False},
                {"message": {"content": ""}, "done": True, "eval_count": 5},
            ),
        )

    async with client_for(handler) as client:
        result = await run_once(client, "m", {"content": "1+1", "tools": [{}]}, {}, tag="t")

    assert result["ttft_ms"] is not None
    assert result["gen_tok_s"] is None  # no eval_duration reported


async def test_run_once_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    async with client_for(handler) as client:
        with pytest.raises(RuntimeError, match="404"):
            await run_once(client, "absent", {"content": "hi"}, {}, tag="t")


async def test_memory_reports_vram_share():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"models": [{"name": "m", "size": 4_000_000_000, "size_vram": 1_000_000_000}]}
        )

    async with client_for(handler) as client:
        assert await memory(client, "m") == {"size_gb": 4.0, "vram_gb": 1.0, "gpu_pct": 25}
        assert (await memory(client, "other"))["size_gb"] is None


def test_percentile_nearest_rank():
    assert percentile([5.0], 95) == 5.0
    assert percentile([float(i) for i in range(1, 21)], 95) == 19.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_summarize_ignores_missing_values():
    samples = [
        {"ttft_ms": 10, "latency_ms": 100, "gen_tok_s": None, "prompt_tok_s": 50},
        {"ttft_ms": 30, "latency_ms": 300, "gen_tok_s": None, "prompt_tok_s": 70},
    ]
    summary = summarize(samples)
    assert summary["ttft_ms"]["p50"] == 20
    assert summary["gen_tok_s"] is None


def test_render_report_has_one_row_per_prompt():
    result = {
        "date": "2026-10-27T10:00:00+00:00",
        "machine": {
            "hostname": "pc1",
            "cpu": "cpu",
            "cores": 8,
            "ram_gb": 16.0,
            "gpu": None,
            "ollama": "0.34.4",
        },
        "models": [
            {
                "model": "m",
                "load_ms": 1500,
                "memory": {"size_gb": 2.0, "vram_gb": 0.0, "gpu_pct": 0},
                "prompts": {
                    name: {"summary": summarize([{"ttft_ms": 1, "latency_ms": 2}])}
                    for name in ("short", "long")
                },
            }
        ],
    }
    report = render_report([result])
    assert "| pc1 | m | short | 1.5 |" in report
    assert "| pc1 | m | long |" in report
    assert "| pc1 | cpu | 8 | 16.0 | — | 0.34.4 | 2026-10-27 |" in report
