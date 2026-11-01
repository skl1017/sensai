"""Unit tests for `CacheNode` (#59/#60): semantic cache in front of the model.

Most tests run a real `core.agent.Agent` with `[CacheNode, ReActLoopNode]` over a
real `SemanticCache` (in-memory SQLite) and a bag-of-words `FakeEmbedder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.agent import Agent
from core.llm import IEmbedder
from core.types import Context, Message, ToolCall
from nodes.cache_node import CacheNode, _same_discriminators
from nodes.react_loop import ReActLoopNode
from storage.profile import ProfileStore
from storage.semantic_cache import SemanticCache
from tests.fakes import FakeEmbedder, FakeLLM, FakeTool, make_semantic_cache

DIM = 256


class FakeClock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class ConstantEmbedder(IEmbedder):
    """Every text gets the same vector: similarity is always 1 (isolates the lexical guard)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


class BrokenEmbedder(IEmbedder):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("ollama down")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    return make_semantic_cache(DIM, clock)


def make_scope(value: str = "scope-a"):
    async def scope_fn(ctx: Context) -> str:
        return value

    return scope_fn


def make_agent(llm, cache, embedder=None, tools=(), scope_fn=None, **node_kwargs):
    node = CacheNode(
        embedder if embedder is not None else FakeEmbedder(DIM),
        cache,
        scope_fn or make_scope(),
        **node_kwargs,
    )
    return Agent(llm, list(tools), [node, ReActLoopNode()])


async def stored_count(cache: SemanticCache) -> int:
    # Large ttl so every entry counts as fresh.
    return len(cache._candidate_ids("scope-a", None))


# -- hit / miss ------------------------------------------------------------------


async def test_miss_then_identical_question_hits_without_calling_llm(cache):
    llm = FakeLLM(["Paris."])
    agent = make_agent(llm, cache)

    first = await agent.run("What is the capital of France?", [])
    assert first.response == "Paris."
    assert first.state["cache_hit"] is False
    assert len(llm.calls) == 1

    second = await agent.run("What is the capital of France?", [])

    assert second.response == "Paris."
    assert second.state["cache_hit"] is True
    assert len(llm.calls) == 1  # script is exhausted: a second call would raise


async def test_below_threshold_is_a_miss(cache):
    llm = FakeLLM(["A", "B"])
    agent = make_agent(llm, cache, threshold=0.99)
    await agent.run("alpha beta gamma delta", [])

    result = await agent.run("alpha beta gamma epsilon", [])  # cosine 0.75

    assert result.state["cache_hit"] is False
    assert result.response == "B"


async def test_ttl_expired_entry_is_a_miss(cache, clock):
    llm = FakeLLM(["A", "B"])
    agent = make_agent(llm, cache, ttl_hours=1)
    await agent.run("hello there", [])

    clock.now += 3601
    result = await agent.run("hello there", [])

    assert result.state["cache_hit"] is False
    assert result.response == "B"


async def test_different_scope_is_a_miss(cache):
    llm = FakeLLM(["A", "B"])
    await make_agent(llm, cache, scope_fn=make_scope("scope-a")).run("hello there", [])

    result = await make_agent(llm, cache, scope_fn=make_scope("scope-b")).run("hello there", [])

    assert result.state["cache_hit"] is False
    assert result.response == "B"


async def test_rag_filters_neither_read_nor_write(cache):
    llm = FakeLLM(["A", "B"])
    agent = make_agent(llm, cache)
    await agent.run("hello there", [])  # populates the cache

    result = await agent.run("hello there", [], {"rag_filters": {"file_name": "rh.pdf"}})

    assert result.state["cache_hit"] is False
    assert result.response == "B"  # no read
    assert await stored_count(cache) == 1  # no write


# -- mode --------------------------------------------------------------------------


HISTORY = [Message("user", "hi"), Message("assistant", "hello")]


async def test_first_turn_only_with_history_neither_reads_nor_writes(cache):
    llm = FakeLLM(["A", "B"])
    agent = make_agent(llm, cache)
    await agent.run("hello there", [])  # populates the cache

    result = await agent.run("hello there", HISTORY)

    assert result.state["cache_hit"] is False
    assert result.response == "B"  # no read
    assert await stored_count(cache) == 1  # no write


async def test_first_turn_only_with_history_does_not_write(cache):
    llm = FakeLLM(["A"])
    await make_agent(llm, cache).run("hello there", HISTORY)

    assert await stored_count(cache) == 0


async def test_mode_always_reads_with_history(cache):
    llm = FakeLLM(["A"])
    agent = make_agent(llm, cache, mode="always")
    await agent.run("hello there", [])

    result = await agent.run("hello there", HISTORY)

    assert result.state["cache_hit"] is True
    assert result.response == "A"


# -- failures ------------------------------------------------------------------------


async def test_embedding_failure_is_a_miss_and_stores_nothing(cache):
    llm = FakeLLM(["A"])
    agent = make_agent(llm, cache, embedder=BrokenEmbedder())

    result = await agent.run("hello there", [])

    assert result.response == "A"
    assert result.state["cache_hit"] is False
    assert len(llm.calls) == 1
    assert await stored_count(cache) == 0


async def test_lookup_failure_is_a_miss(cache):
    async def broken_lookup(*args, **kwargs):
        raise RuntimeError("db down")

    cache.lookup = broken_lookup
    result = await make_agent(FakeLLM(["A"]), cache).run("hello there", [])

    assert result.response == "A"
    assert result.state["cache_hit"] is False


async def test_write_failure_is_logged_not_raised(cache, caplog):
    async def broken_store(*args, **kwargs):
        raise RuntimeError("disk full")

    cache.store = broken_store
    result = await make_agent(FakeLLM(["A"]), cache).run("hello there", [])

    assert result.response == "A"
    assert "Cache: write failed" in caplog.text


# -- write exclusions ----------------------------------------------------------------


async def run_with_next(cache, next_fn, question="hello there"):
    node = CacheNode(FakeEmbedder(DIM), cache, make_scope())
    ctx = Context(question, [], FakeLLM([]), {})
    return await node.handle(ctx, next_fn)


async def test_empty_response_is_not_stored(cache):
    async def next_fn(ctx):
        ctx.response = "   "
        return ctx

    await run_with_next(cache, next_fn)
    assert await stored_count(cache) == 0


async def test_blocked_is_not_stored(cache):
    async def next_fn(ctx):
        ctx.response = "Refused."
        ctx.state["blocked"] = True
        return ctx

    await run_with_next(cache, next_fn)
    assert await stored_count(cache) == 0


async def test_react_truncated_is_not_stored(cache):
    async def next_fn(ctx):
        ctx.response = "Partial."
        ctx.state["react_truncated"] = True
        return ctx

    await run_with_next(cache, next_fn)
    assert await stored_count(cache) == 0


async def test_answer_after_tool_use_is_not_stored(cache):
    tool = FakeTool(name="clock", result="12:00")
    call = ToolCall(id="1", name="clock", arguments={})
    llm = FakeLLM([[call], "It is 12:00."])
    agent = make_agent(llm, cache, tools=[tool])

    result = await agent.run("what time is it", [])

    assert result.response == "It is 12:00."
    assert len(tool.calls) == 1
    assert await stored_count(cache) == 0


async def test_answer_from_a_filtered_search_is_not_stored(cache):
    async def next_fn(ctx):
        ctx.response = "A"
        ctx.state["rag_applied_filters"] = {"date_to": "2023-12-31"}
        return ctx

    await run_with_next(cache, next_fn)

    assert await stored_count(cache) == 0


async def test_plain_answer_is_stored(cache):
    await make_agent(FakeLLM(["A"]), cache).run("hello there", [])
    assert await stored_count(cache) == 1


# -- lexical guard ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Je veux un conge", "Je ne veux pas un conge"),
        ("Is the door open?", "Is the door not open?"),
        ("It works", "It doesn't work"),
        ("Combien font 2 + 2 ?", "Combien font 2 + 3 ?"),
        ("Price is 1.5 euros", "Price is 15 euros"),
    ],
)
async def test_guard_blocks_even_with_similarity_one(cache, first, second):
    llm = FakeLLM(["A", "B"])
    agent = make_agent(llm, cache, embedder=ConstantEmbedder())
    await agent.run(first, [])

    result = await agent.run(second, [])

    assert result.state["cache_hit"] is False
    assert result.response == "B"


async def test_guard_allows_same_discriminators(cache):
    llm = FakeLLM(["A"])
    agent = make_agent(llm, cache, embedder=ConstantEmbedder())
    await agent.run("Combien font 2 + 2 ?", [])

    result = await agent.run("Dis moi combien font 2 + 2", [])

    assert result.state["cache_hit"] is True


def test_same_discriminators_basics():
    assert _same_discriminators("Hello", "hi there")
    assert not _same_discriminators("n'est pas", "est")
    assert _same_discriminators("1,5 kg", "1,5 kg")
    assert not _same_discriminators("2 and 2", "2")


# -- from_config -------------------------------------------------------------------------


@dataclass
class FakeStores:
    semantic_cache: object = None
    profile: object = None


@dataclass
class FakeDeps:
    embedder: object = None
    stores: object = None
    config_dir: Path = Path("config")
    persona: str = "default"
    prompt_version: str | None = None


def test_from_config_unknown_param_raises(cache):
    deps = FakeDeps(FakeEmbedder(), FakeStores(cache))
    with pytest.raises(ValueError, match="Unknown CacheNode param"):
        CacheNode.from_config({"thresold": 0.9}, deps)


def test_from_config_requires_embedder(cache):
    with pytest.raises(ValueError, match="embedder"):
        CacheNode.from_config({}, FakeDeps(None, FakeStores(cache)))


def test_from_config_requires_semantic_cache():
    with pytest.raises(ValueError, match="semantic cache"):
        CacheNode.from_config({}, FakeDeps(FakeEmbedder(), FakeStores(None)))
    with pytest.raises(ValueError, match="semantic cache"):
        CacheNode.from_config({}, FakeDeps(FakeEmbedder(), None))


def test_from_config_rejects_bad_mode(cache):
    deps = FakeDeps(FakeEmbedder(), FakeStores(cache))
    with pytest.raises(ValueError, match="mode"):
        CacheNode.from_config({"mode": "sometimes"}, deps)


def test_from_config_accepts_params(cache):
    deps = FakeDeps(FakeEmbedder(), FakeStores(cache))
    node = CacheNode.from_config(
        {"threshold": 0.9, "ttl_hours": 2, "mode": "always", "embedding_dim": 768}, deps
    )
    assert node.threshold == 0.9
    assert node.ttl_seconds == 7200
    assert node.mode == "always"


@pytest.fixture
def config_dir(tmp_path):
    personas = tmp_path / "personas"
    prompts = tmp_path / "prompts"
    personas.mkdir()
    prompts.mkdir()
    for name in ("default", "tutor"):
        (personas / f"{name}.yaml").write_text(f"name: {name}\nprompt_template: base\n")
    (prompts / "base_v1.md").write_text("v1")
    (prompts / "base_v2.md").write_text("v2")
    return tmp_path


def default_scope(config_dir, cache, profile=None, **deps_kwargs):
    profile_store = ProfileStore(None)
    if profile is not None:
        profile_store._memory = profile
    deps = FakeDeps(
        FakeEmbedder(), FakeStores(cache, profile_store), config_dir=config_dir, **deps_kwargs
    )
    return CacheNode.from_config({}, deps).scope_fn


def make_ctx(**state) -> Context:
    return Context("hi", [], FakeLLM([]), {}, state=state)


async def test_default_scope_is_stable(config_dir, cache):
    scope_fn = default_scope(config_dir, cache)
    assert await scope_fn(make_ctx()) == await scope_fn(make_ctx())


async def test_default_scope_changes_with_pinned_prompt_version(config_dir, cache):
    v1 = await default_scope(config_dir, cache, prompt_version="v1")(make_ctx())
    v2 = await default_scope(config_dir, cache, prompt_version="v2")(make_ctx())
    assert v1 != v2


async def test_default_scope_changes_when_a_new_latest_prompt_appears(config_dir, cache):
    scope_fn = default_scope(config_dir, cache)
    before = await scope_fn(make_ctx())

    (config_dir / "prompts" / "base_v3.md").write_text("v3")

    assert await scope_fn(make_ctx()) != before


async def test_default_scope_changes_with_persona(config_dir, cache):
    scope_fn = default_scope(config_dir, cache)
    assert await scope_fn(make_ctx()) != await scope_fn(make_ctx(persona="tutor"))


async def test_default_scope_changes_with_profile(config_dir, cache):
    a = await default_scope(config_dir, cache, profile={"preferences": {"lang": "fr"}})(make_ctx())
    b = await default_scope(config_dir, cache, profile={"preferences": {"lang": "en"}})(make_ctx())
    assert a != b
