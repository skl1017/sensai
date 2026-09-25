"""Unit tests for the minimal `ContextBuilderNode`/`ProfileSection` (#25).

`FakeProfileStore` mirrors `ProfileStore.load()`'s shape so these tests don't
depend on the storage layer (built in parallel). One integration test at the
bottom exercises the real `ProfileStore` and is skipped if it doesn't exist
yet.
"""

from __future__ import annotations

import asyncio

import pytest

from core.agent import Agent
from core.types import Context, Message
from nodes.context_builder import ContextBuilderNode, ContextSection, ProfileSection
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeLLM


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
    def __init__(self, profile) -> None:
        self.stores = _Stores(profile)


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
        ContextBuilderNode.from_config({"bogus": 1}, _Deps(FakeProfileStore()))


@pytest.mark.parametrize("name", ["persona", "memory", "artifact", "rag", "fewshot"])
def test_from_config_story6_sections_raise_not_available_yet(name):
    with pytest.raises(ValueError, match="Story 6"):
        ContextBuilderNode.from_config({"sections": [name]}, _Deps(FakeProfileStore()))


def test_from_config_unknown_section_name_raises():
    with pytest.raises(ValueError):
        ContextBuilderNode.from_config({"sections": ["bogus"]}, _Deps(FakeProfileStore()))


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
