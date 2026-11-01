"""Tests for `eval/run.py` (fake agents, fake judge, no Ollama)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import build_agent
from eval.run import (
    BenchUI,
    aggregate,
    load_configs,
    load_questions,
    render_report,
    run_bench,
    write_results,
)
from tests.fakes import FakeEmbedder, FakeLLM

QUESTIONS = [
    {"question": "q1", "sources": ["src"]},
    {"question": "q2", "sources": []},
]


class FakeAgent:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.closed = False

    async def run(self, text, history, state):
        return SimpleNamespace(
            response=f"{self.tag}:{text}",
            state={
                "metrics": {"prompt_tokens": 10, "completion_tokens": 5},
                "retrieved_chunks": [{"text": "chunk"}],
            },
        )

    async def aclose(self):
        self.closed = True


async def factory(entry):
    return FakeAgent(entry["name"])


async def fake_judge(llm, question, answer, sources):
    base = 4 if answer.startswith("good") else 2
    return {
        "relevance": base,
        "coherence": base,
        "faithfulness": base if sources else None,
        "unsupported_claims": [],
    }


def test_aggregate_ignores_none_faithfulness_and_counts_failures():
    rows = [
        {
            "scores": {"relevance": 4, "coherence": 2, "faithfulness": None},
            "latency_ms": 10,
            "tokens": 100,
        },
        {
            "scores": {"relevance": 2, "coherence": 4, "faithfulness": 3},
            "latency_ms": 30,
            "tokens": None,
        },
        {"scores": None, "latency_ms": 20, "tokens": 50},
    ]
    s = aggregate(rows)
    assert s["n"] == 3
    assert s["relevance"] == 3
    assert s["coherence"] == 3
    assert s["faithfulness"] == 3
    assert s["judge_failures"] == 1
    assert s["latency_ms"] == 20
    assert s["tokens"] == 75


def test_aggregate_all_none_gives_none():
    s = aggregate(
        [
            {
                "scores": {"relevance": 1, "coherence": 1, "faithfulness": None},
                "latency_ms": 1,
                "tokens": None,
            }
        ]
    )
    assert s["faithfulness"] is None
    assert s["tokens"] is None


async def test_end_to_end_two_configs_report_shows_difference(tmp_path):
    configs = [{"name": "good", "config": "x"}, {"name": "bad", "config": "x"}]

    async def fac(entry):
        return FakeAgent("good" if entry["name"] == "good" else "bad")

    result = await run_bench(QUESTIONS, configs, fac, fake_judge, FakeLLM([]))
    by_name = {c["name"]: c["summary"] for c in result["configs"]}
    assert by_name["good"]["relevance"] == 4
    assert by_name["bad"]["relevance"] == 2
    assert by_name["good"]["faithfulness"] == 4  # retrieved chunk counts as a source
    assert by_name["good"]["tokens"] == 15

    report = render_report(result)
    assert "| good |" in report and "| bad |" in report
    assert "Difference (bad - good): relevance -2.00" in report
    assert "4/4/4" in report and "2/2/2" in report

    json_path, md_path = write_results(result, tmp_path)
    assert json.loads(json_path.read_text())["configs"][0]["name"] == "good"
    assert md_path.name == "latest.md" and "Offline evaluation" in md_path.read_text()


async def test_sources_include_retrieved_chunks_and_agent_closed():
    seen = []

    async def judge(llm, q, a, sources):
        seen.append(sources)
        return {"relevance": 1, "coherence": 1, "faithfulness": 1}

    agents = []

    async def fac(entry):
        agents.append(FakeAgent("a"))
        return agents[-1]

    await run_bench(QUESTIONS[:1], [{"name": "a", "config": "x"}], fac, judge, FakeLLM([]))
    assert seen == [["src", "chunk"]]
    assert agents[0].closed


async def test_judge_failure_and_agent_error_are_counted():
    class Boom(FakeAgent):
        async def run(self, text, history, state):
            if text == "q2":
                raise RuntimeError("boom")
            return await super().run(text, history, state)

    async def fac(entry):
        return Boom("a")

    async def bad_judge(*a):
        return None

    result = await run_bench(QUESTIONS, [{"name": "a", "config": "x"}], fac, bad_judge, FakeLLM([]))
    s = result["configs"][0]["summary"]
    assert s["judge_failures"] == 2 and s["errors"] == 1
    assert "error" in render_report(result)


def test_load_yaml_files():
    qs = load_questions("eval/questions.yaml")
    assert len(qs) >= 10 and all(q["question"] for q in qs)
    cfgs = load_configs("eval/configs.yaml")
    assert [c["prompt_version"] for c in cfgs] == ["v1", "v2"]


async def test_bench_ui_refuses_approvals():
    ui = BenchUI()
    assert ui.on_token("x") is None
    assert await ui.ask("ok?") == "no"


@pytest.mark.usefixtures("no_mcp")
async def test_build_agent_prompt_version_override():
    outcomes = {}
    for version in ("v1", "v2", None):
        llm = FakeLLM(["hi"])
        llm.embedder = FakeEmbedder()
        agent = await build_agent(
            "config/agent.eval.yaml", BenchUI(), llm=llm, prompt_version=version
        )
        ctx = await agent.run("hello", [], state={})
        outcomes[version] = ctx.state["prompt"]["version"]
        await agent.aclose()
    assert outcomes == {"v1": "v1", "v2": "v2", None: "v2"}  # None keeps the config's v2
