import asyncio
import json
import random
from types import SimpleNamespace

import pytest

from core.types import Context
from nodes.eval_node import JUDGE_SCHEMA, EvalNode, judge
from tests.fakes import FakeLLM

GOOD = {
    "relevance": 5,
    "coherence": 4,
    "faithfulness": 3,
    "unsupported_claims": [{"claim": "x", "reason": "contradicted"}],
}


def js(d) -> str:
    return json.dumps(d)


class FakeEventLog:
    def __init__(self):
        self.updates = []

    async def update(self, trace_id, patch):
        self.updates.append((trace_id, patch))


def make_ctx(response="réponse", **state):
    return Context("question", [], FakeLLM([]), {}, response=response, state=dict(state))


async def passthrough(ctx):
    return ctx


async def test_judge_valid_scores():
    llm = FakeLLM([js(GOOD)])
    result = await judge(llm, "q", "a", ["src"])
    assert result == {**GOOD, "unsupported_claims": [{"claim": "x", "reason": "contradicted"}]}
    assert llm.calls[0]["schema"] == JUDGE_SCHEMA


async def test_judge_clamps_and_cleans():
    raw = {
        "relevance": 9,
        "coherence": -2,
        "faithfulness": 4,
        "unsupported_claims": [
            {"claim": "a", "reason": "weird"},
            {"claim": "", "reason": "unverifiable"},
            "junk",
            {"reason": "unverifiable"},
        ],
    }
    result = await judge(FakeLLM([js(raw)]), "q", "a", ["s"])
    assert result["relevance"] == 5
    assert result["coherence"] == 0
    assert result["unsupported_claims"] == [{"claim": "a", "reason": "unverifiable"}]


async def test_judge_invalid_json_replayed_once():
    llm = FakeLLM(["not json", js(GOOD)])
    assert (await judge(llm, "q", "a", ["s"]))["relevance"] == 5
    assert len(llm.calls) == 2


async def test_judge_two_invalid_returns_none():
    llm = FakeLLM(["nope", "still nope"])
    assert await judge(llm, "q", "a", ["s"]) is None


async def test_judge_missing_scores_returns_none():
    assert await judge(FakeLLM([js({"relevance": 3})]), "q", "a", ["s"]) is None


async def test_judge_faithfulness_none_without_sources():
    llm = FakeLLM([js(GOOD)])
    result = await judge(llm, "q", "a", [])
    assert result["faithfulness"] is None
    assert "aucune" in llm.calls[0]["messages"][-1].content


async def test_judge_exception_returns_none():
    class Boom(FakeLLM):
        def chat(self, *a, **k):
            raise RuntimeError("down")

    assert await judge(Boom([]), "q", "a", ["s"]) is None


async def test_judge_does_not_swallow_cancellation():
    class Cancelling(FakeLLM):
        def chat(self, *a, **k):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await judge(Cancelling([]), "q", "a", ["s"])


async def test_inline_sets_scores_and_uses_sources():
    llm = FakeLLM([js(GOOD)])
    node = EvalNode(llm, mode="inline")
    ctx = make_ctx(
        retrieved_chunks=[{"index": 1, "text": "chunk text"}, {"index": 2}],
        react_trace=[{"step": 1, "observation": "obs text"}],
    )
    out = await node.handle(ctx, passthrough)
    assert out.state["eval_scores"]["relevance"] == 5
    prompt = llm.calls[0]["messages"][-1].content
    assert "chunk text" in prompt and "obs text" in prompt


@pytest.mark.parametrize("state", [{"cache_hit": True}, {"blocked": "length"}])
async def test_skip_on_cache_hit_or_blocked(state):
    llm = FakeLLM([js(GOOD)])
    out = await EvalNode(llm, mode="inline").handle(make_ctx(**state), passthrough)
    assert "eval_scores" not in out.state
    assert llm.calls == []


async def test_skip_when_no_response():
    llm = FakeLLM([js(GOOD)])
    out = await EvalNode(llm, mode="inline").handle(make_ctx(response=None), passthrough)
    assert "eval_scores" not in out.state and llm.calls == []


async def test_sample_rate_zero_and_one():
    llm = FakeLLM([js(GOOD)])
    node = EvalNode(llm, mode="inline", sample_rate=0.0, rng=random.Random(1))
    out = await node.handle(make_ctx(), passthrough)
    assert "eval_scores" not in out.state and llm.calls == []

    node = EvalNode(llm, mode="inline", sample_rate=1.0, rng=random.Random(1))
    out = await node.handle(make_ctx(), passthrough)
    assert out.state["eval_scores"] is not None


async def test_annotate_appends_claims():
    node = EvalNode(FakeLLM([js(GOOD)]), mode="inline", annotate=True)
    out = await node.handle(make_ctx(), passthrough)
    assert out.response.startswith("réponse")
    assert "x (contradicted)" in out.response


async def test_no_annotate_leaves_response():
    node = EvalNode(FakeLLM([js(GOOD)]), mode="inline")
    out = await node.handle(make_ctx(), passthrough)
    assert out.response == "réponse"


async def test_inline_judge_failure_does_not_fail_request():
    node = EvalNode(FakeLLM(["bad", "bad"]), mode="inline")
    out = await node.handle(make_ctx(), passthrough)
    assert out.state["eval_scores"] is None
    assert out.response == "réponse"


async def test_background_updates_event_log():
    log = FakeEventLog()
    node = EvalNode(FakeLLM([js(GOOD)]), mode="background", event_log=log)
    out = await node.handle(make_ctx(trace_id="t1"), passthrough)
    assert "eval_scores" not in out.state
    await node.drain()
    assert log.updates[0][0] == "t1"
    assert log.updates[0][1]["eval_scores"]["relevance"] == 5
    assert not node._pending


async def test_background_errors_are_contained():
    class BadLog:
        async def update(self, *a):
            raise RuntimeError("disk")

    node = EvalNode(FakeLLM([js(GOOD)]), event_log=BadLog())
    await node.handle(make_ctx(trace_id="t1"), passthrough)
    await node.drain()


async def test_background_without_trace_id_does_not_update():
    log = FakeEventLog()
    node = EvalNode(FakeLLM([js(GOOD)]), event_log=log)
    await node.handle(make_ctx(), passthrough)
    await node.drain()
    assert log.updates == []


def test_from_config_defaults_and_fallback_llm():
    llm = FakeLLM([])
    log = FakeEventLog()
    node = EvalNode.from_config({}, SimpleNamespace(llm=llm, event_log=log))
    assert node.judge_llm is llm and node.event_log is log
    assert node.mode == "background" and node.sample_rate == 1.0 and not node.annotate


def test_from_config_judge_model_uses_make_llm():
    made = []

    def make_llm(model, temperature=0):
        made.append((model, temperature))
        return FakeLLM([], model=model)

    deps = SimpleNamespace(llm=FakeLLM([]), make_llm=make_llm)
    node = EvalNode.from_config({"judge_model": "big", "mode": "inline", "annotate": True}, deps)
    assert made == [("big", 0)]
    assert node.judge_llm.model == "big" and node.mode == "inline" and node.annotate


@pytest.mark.parametrize(
    "params",
    [
        {"nope": 1},
        {"mode": "weird"},
        {"sample_rate": 2},
        {"sample_rate": "x"},
        {"annotate": "yes"},
        {"temperature": "hot"},
        {"judge_model": 3},
    ],
)
def test_from_config_rejects_bad_params(params):
    with pytest.raises(ValueError):
        EvalNode.from_config(params, SimpleNamespace(llm=FakeLLM([])))
