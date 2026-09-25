"""Unit tests for `ContextBuilderNode` (#25, Story 6).

`FakeProfileStore` mirrors `ProfileStore.load()`'s shape so these tests don't
depend on the storage layer (built in parallel). One integration test at the
bottom exercises the real `ProfileStore` and is skipped if it doesn't exist
yet.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from core.agent import Agent
from core.types import Context, Message
from nodes.context_builder import ContextBuilderNode, ContextSection, ProfileSection
from nodes.fewshot_section import FewShotSection
from nodes.persona_section import PersonaSection
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeEmbedder, FakeLLM


class FakeProfileStore:
    def __init__(self, profile: dict | None = None) -> None:
        self.profile = profile if profile is not None else {}

    def load(self) -> dict:
        return self.profile


class StaticSection(ContextSection):
    def __init__(self, text: str | None, priority: int) -> None:
        self.text = text
        self.priority = priority

    async def build(self, ctx: Context) -> str | None:
        return self.text


class FailingSection(ContextSection):
    priority = 5

    async def build(self, ctx: Context) -> str | None:
        raise RuntimeError("section exploded")


class CancellingSection(ContextSection):
    priority = 1

    async def build(self, ctx: Context) -> str | None:
        raise asyncio.CancelledError()


async def _next(ctx: Context) -> Context:
    return ctx


def make_ctx(user_input: str = "hi", messages: list[Message] | None = None) -> Context:
    llm = FakeLLM(["unused"])
    return Context(user_input, messages if messages is not None else [], llm, {})


# --- ordering / joining ----------------------------------------------------------


async def test_sections_ordered_by_priority():
    node = ContextBuilderNode([StaticSection("second", 20), StaticSection("first", 10)])
    ctx = make_ctx()

    result = await node.handle(ctx, _next)

    assert result.messages[0] == Message("system", "first\n\nsecond")


async def test_all_sections_none_produce_no_system_message():
    node = ContextBuilderNode([StaticSection(None, 10)])
    ctx = make_ctx()

    result = await node.handle(ctx, _next)

    assert [m.role for m in result.messages] == ["user"]


# --- failure isolation -------------------------------------------------------------


async def test_failing_section_is_skipped_and_request_still_runs(caplog):
    node = ContextBuilderNode([FailingSection(), StaticSection("ok", 10)])
    ctx = make_ctx()

    with caplog.at_level("WARNING"):
        result = await node.handle(ctx, _next)

    assert result.messages[0] == Message("system", "ok")
    assert any("FailingSection" in r.getMessage() for r in caplog.records)


async def test_cancelled_error_from_section_propagates_and_is_never_logged_as_a_warning(caplog):
    node = ContextBuilderNode([CancellingSection()])
    ctx = make_ctx()

    with caplog.at_level("WARNING"), pytest.raises(asyncio.CancelledError):
        await node.handle(ctx, _next)

    assert caplog.records == []


# --- history rebuild ---------------------------------------------------------------


async def test_caller_history_list_not_mutated():
    history = [Message("user", "old q"), Message("assistant", "old a")]
    original = list(history)
    node = ContextBuilderNode([StaticSection("profile text", 10)])
    ctx = make_ctx(user_input="new q", messages=history)

    result = await node.handle(ctx, _next)

    assert history == original
    assert result.messages is not history
    assert [m.content for m in result.messages] == ["profile text", "old q", "old a", "new q"]


async def test_previous_system_messages_are_dropped_on_rebuild():
    history = [Message("system", "stale"), Message("user", "old q")]
    node = ContextBuilderNode([StaticSection("fresh", 10)])
    ctx = make_ctx(user_input="new q", messages=history)

    result = await node.handle(ctx, _next)

    assert [m.role for m in result.messages] == ["system", "user", "user"]
    assert result.messages[0].content == "fresh"


async def test_next_is_awaited_with_rebuilt_context():
    seen = []

    async def next_(ctx: Context) -> Context:
        seen.append(list(ctx.messages))
        return ctx

    node = ContextBuilderNode([StaticSection("sys", 10)])
    ctx = make_ctx(user_input="hi")

    await node.handle(ctx, next_)

    assert len(seen) == 1
    assert [m.role for m in seen[0]] == ["system", "user"]


# --- ProfileSection ----------------------------------------------------------------


async def test_profile_section_renders_preferences_and_instructions():
    store = FakeProfileStore(
        {"preferences": {"lang": "fr", "tone": "casual"}, "instructions": "Be terse."}
    )
    section = ProfileSection(store)

    text = await section.build(make_ctx())

    assert text == "User profile:\n- lang: fr\n- tone: casual\nBe terse."


async def test_profile_section_empty_profile_returns_none():
    assert await ProfileSection(FakeProfileStore({})).build(make_ctx()) is None
    empty = FakeProfileStore({"preferences": {}, "instructions": "  "})
    assert await ProfileSection(empty).build(make_ctx()) is None


async def test_profile_section_only_instructions():
    section = ProfileSection(FakeProfileStore({"instructions": "Answer in Spanish."}))

    text = await section.build(make_ctx())

    assert text == "User profile:\nAnswer in Spanish."


async def test_profile_section_only_preferences():
    section = ProfileSection(FakeProfileStore({"preferences": {"lang": "fr"}}))

    text = await section.build(make_ctx())

    assert text == "User profile:\n- lang: fr"


async def test_profile_reinjected_every_turn_through_agent():
    store = FakeProfileStore({"instructions": "Answer in English."})
    llm = FakeLLM(["ans1", "ans2"])
    agent = Agent(llm, [], [ContextBuilderNode([ProfileSection(store)]), ReActLoopNode()])

    await agent.run("hi", [])
    first_system = llm.calls[0]["messages"][0]
    assert first_system.role == "system"
    assert "Answer in English." in first_system.content

    store.profile = {"instructions": "Answer in French."}
    await agent.run("hi again", [])
    second_system = llm.calls[1]["messages"][0]
    assert "Answer in French." in second_system.content


# --- from_config -------------------------------------------------------------------


class _Stores:
    def __init__(self, profile) -> None:
        self.profile = profile


class _Deps:
    """`Deps`-shaped double: real `config/` dir by default, so `persona`/`fewshot`
    sections can be built without a full `app.Deps`."""

    def __init__(
        self,
        profile=None,
        config_dir: Path = Path("config"),
        persona: str = "default",
        prompt_version: str | None = None,
        embedder=None,
    ) -> None:
        self.stores = _Stores(profile if profile is not None else FakeProfileStore())
        self.config_dir = config_dir
        self.persona = persona
        self.prompt_version = prompt_version
        self.embedder = embedder


def test_from_config_default_sections_is_profile():
    store = FakeProfileStore({"instructions": "hi"})
    node = ContextBuilderNode.from_config({}, _Deps(store))

    assert len(node.sections) == 1
    assert isinstance(node.sections[0], ProfileSection)
    assert node.sections[0].profile_store is store


def test_from_config_ignores_profile_path_param():
    store = FakeProfileStore()
    node = ContextBuilderNode.from_config({"profile_path": "data/profile.yaml"}, _Deps(store))

    assert len(node.sections) == 1
    assert isinstance(node.sections[0], ProfileSection)


def test_from_config_unknown_param_raises():
    with pytest.raises(ValueError):
        ContextBuilderNode.from_config({"bogus": 1}, _Deps())


def test_from_config_unknown_section_name_raises():
    with pytest.raises(ValueError):
        ContextBuilderNode.from_config({"sections": ["bogus"]}, _Deps())


@pytest.mark.parametrize(
    ("name", "story"), [("memory", "Story 10"), ("artifact", "Story 15"), ("rag", "Story 11")]
)
def test_from_config_unimplemented_sections_raise_not_available_yet(name, story):
    with pytest.raises(ValueError, match=story):
        ContextBuilderNode.from_config({"sections": [name]}, _Deps())


def test_from_config_persona_section_is_available():
    node = ContextBuilderNode.from_config({"sections": ["persona"]}, _Deps())

    assert len(node.sections) == 1
    assert isinstance(node.sections[0], PersonaSection)


def test_from_config_fewshot_section_is_available():
    node = ContextBuilderNode.from_config({"sections": ["fewshot"]}, _Deps(embedder=FakeEmbedder()))

    assert len(node.sections) == 1
    assert isinstance(node.sections[0], FewShotSection)


def test_from_config_fewshot_passes_its_subdict_through():
    node = ContextBuilderNode.from_config(
        {"sections": ["fewshot"], "fewshot": {"k": 5}}, _Deps(embedder=FakeEmbedder())
    )

    assert node.sections[0].k == 5


def test_from_config_fewshot_without_embedder_raises():
    with pytest.raises(ValueError, match="embedder"):
        ContextBuilderNode.from_config({"sections": ["fewshot"]}, _Deps(embedder=None))


# --- from_config: budget -------------------------------------------------------------


def test_from_config_no_budget_means_no_compression():
    node = ContextBuilderNode.from_config({}, _Deps())
    assert node.budget is None


def test_from_config_budget_is_kept_as_given():
    node = ContextBuilderNode.from_config(
        {"budget": {"reserve_output": 512, "trigger_ratio": 0.5, "keep_last_turns": 2}}, _Deps()
    )
    assert node.budget == {"reserve_output": 512, "trigger_ratio": 0.5, "keep_last_turns": 2}


def test_from_config_budget_partial_is_kept_partial():
    node = ContextBuilderNode.from_config({"budget": {"keep_last_turns": 1}}, _Deps())
    assert node.budget == {"keep_last_turns": 1}


def test_from_config_budget_unknown_key_raises():
    with pytest.raises(ValueError, match="bogus"):
        ContextBuilderNode.from_config({"budget": {"bogus": 1}}, _Deps())


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, 2])
def test_from_config_budget_bad_ratio_raises(ratio):
    with pytest.raises(ValueError, match="trigger_ratio"):
        ContextBuilderNode.from_config({"budget": {"trigger_ratio": ratio}}, _Deps())


@pytest.mark.parametrize("value", [-1, -100])
def test_from_config_budget_negative_int_raises(value):
    with pytest.raises(ValueError):
        ContextBuilderNode.from_config({"budget": {"reserve_output": value}}, _Deps())


def test_from_config_budget_non_int_raises():
    with pytest.raises(ValueError):
        ContextBuilderNode.from_config({"budget": {"keep_last_turns": 1.5}}, _Deps())


def test_from_config_budget_not_a_mapping_raises():
    with pytest.raises(ValueError, match="budget"):
        ContextBuilderNode.from_config({"budget": ["nope"]}, _Deps())


# --- from_config: rewrite_query -------------------------------------------------------


def test_from_config_rewrite_query_defaults_to_false():
    node = ContextBuilderNode.from_config({}, _Deps())
    assert node.rewrite_query is False


def test_from_config_rewrite_query_true():
    node = ContextBuilderNode.from_config({"rewrite_query": True}, _Deps())
    assert node.rewrite_query is True


def test_from_config_rewrite_query_non_bool_raises():
    with pytest.raises(ValueError, match="rewrite_query"):
        ContextBuilderNode.from_config({"rewrite_query": "true"}, _Deps())


# --- handle: rewrite_query runs before sections gather --------------------------------


class RecordingQuerySection(ContextSection):
    priority = 50

    def __init__(self) -> None:
        self.seen_queries: list[str | None] = []

    async def build(self, ctx: Context) -> str | None:
        self.seen_queries.append(ctx.state.get("query"))
        return None


async def test_rewrite_query_sets_state_before_sections_gather():
    llm = FakeLLM(["rewritten query"])
    recorder = RecordingQuerySection()
    node = ContextBuilderNode([recorder], rewrite_query=True)
    history = [Message("user", "old q"), Message("assistant", "old a")]
    ctx = Context("new q", history, llm, {})

    result = await node.handle(ctx, _next)

    assert result.state["query"] == "rewritten query"
    assert recorder.seen_queries == ["rewritten query"]


async def test_rewrite_query_off_leaves_state_query_unset():
    node = ContextBuilderNode([StaticSection("sys", 10)], rewrite_query=False)
    ctx = make_ctx()

    result = await node.handle(ctx, _next)

    assert "query" not in result.state


# --- handle: budget compression pass ---------------------------------------------------


async def test_budget_applied_after_sections_assemble():
    summary_text = "Rolling summary of the earlier turns."
    llm = FakeLLM([summary_text], max_context_tokens=200)
    history = []
    for i in range(6):
        history.append(Message("user", f"question number {i} " * 5))
        history.append(Message("assistant", f"answer number {i} " * 5))
    node = ContextBuilderNode(
        [StaticSection("persona text", 10)],
        budget={"reserve_output": 0, "trigger_ratio": 0.01, "keep_last_turns": 1},
    )
    ctx = Context("current question", history, llm, {})

    result = await node.handle(ctx, _next)

    stats = result.state["context_stats"]
    assert stats["compressed"] is True
    assert result.messages[0].role == "system"
    assert summary_text in result.messages[0].content
    assert "persona text" in result.messages[0].content


async def test_no_budget_means_no_context_stats():
    node = ContextBuilderNode([StaticSection("sys", 10)])
    ctx = make_ctx()

    result = await node.handle(ctx, _next)

    assert "context_stats" not in result.state


# --- integration: persona switch through build_agent + chat_turn ----------------------


async def test_persona_switch_through_build_agent_and_chat_turn(tmp_path):
    from app import build_agent, chat_turn
    from storage.stores import InMemoryStores

    shutil.copytree(Path("config/personas"), tmp_path / "personas")
    shutil.copytree(Path("config/prompts"), tmp_path / "prompts")
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        """
llm:
  provider: ollama
  model: fake
persona: default
prompt_version: v2
pipeline:
  - persist
  - context: { sections: [persona] }
  - react
tools:
  - calculator
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class _NoopUI:
        def on_token(self, text: str) -> None:
            pass

        def on_event(self, kind: str, data: dict) -> None:
            pass

        async def ask(self, question: str, kind: str = "confirm") -> str:
            return "no"

    llm = FakeLLM(["first reply", "second reply"])
    stores = InMemoryStores()
    agent = await build_agent(str(cfg_path), _NoopUI(), llm=llm, stores=stores)

    await chat_turn(agent, stores, "s1", "hello")
    first_system = llm.calls[0]["messages"][0]
    assert first_system.role == "system"
    assert "Sensai" in first_system.content

    await chat_turn(agent, stores, "s1", "hello again", persona="tutor")
    second_system = llm.calls[1]["messages"][0]
    assert "Sensai" not in second_system.content
    assert "Léa" in second_system.content  # the tutor persona's name

    history_contents = [m.content for m in llm.calls[1]["messages"] if m.role != "system"]
    assert "hello" in history_contents
    assert "first reply" in history_contents


# --- integration with the real ProfileStore, if it exists yet -----------------------

try:
    from storage.profile import ProfileStore

    HAS_PROFILE_STORE = True
except ImportError:
    HAS_PROFILE_STORE = False


@pytest.mark.skipif(not HAS_PROFILE_STORE, reason="ProfileStore not implemented yet")
async def test_profile_section_with_real_in_memory_profile_store():
    store = ProfileStore(None)
    store.set_instructions("Be nice.")
    section = ProfileSection(store)

    text = await section.build(make_ctx())

    assert text == "User profile:\nBe nice."
