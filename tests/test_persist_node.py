"""Unit tests for `PersistNode` (#24): saves the exchange after `next` returns.

`FakeConversationStore` mirrors `ConversationStore.append`'s signature so
these tests don't depend on the storage layer (built in parallel). One
integration test at the bottom exercises the real `ConversationStore` and is
skipped if it doesn't exist yet.
"""

from __future__ import annotations

import asyncio

import pytest

from core.types import Context, Message
from nodes.persist_node import PersistNode
from tests.fakes import FakeLLM


class FakeConversationStore:
    """Records `append` calls; mirrors `ConversationStore.append`'s signature."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def append(
        self,
        session_id,
        parent_id,
        messages,
        *,
        artifact=None,
        summary=None,
        interrupted=False,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "parent_id": parent_id,
                "messages": list(messages),
                "artifact": artifact,
                "summary": summary,
                "interrupted": interrupted,
            }
        )
        return "new-id"


def make_ctx(**state) -> Context:
    llm = FakeLLM(["unused"])
    return Context("hello", [], llm, {}, state=state)


# --- persisting ----------------------------------------------------------------


async def test_short_circuit_downstream_is_persisted():
    """A cache-hit/block-style node that sets `ctx.response` without calling `next`
    further down still produces a `ctx` this node persists like any other."""

    async def cache_hit(ctx: Context) -> Context:
        ctx.response = "cached answer"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    result = await node.handle(ctx, cache_hit)

    assert result.response == "cached answer"
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["session_id"] == "s1"
    assert call["parent_id"] is None
    assert call["messages"] == [Message("user", "hello"), Message("assistant", "cached answer")]


async def test_blocked_not_persisted_by_default():
    async def blocked_next(ctx: Context) -> Context:
        ctx.response = "blocked message"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)  # persist_blocked=False
    ctx = make_ctx(session_id="s1", blocked="prompt injection")

    await node.handle(ctx, blocked_next)

    assert store.calls == []


async def test_blocked_persisted_when_persist_blocked_true():
    async def blocked_next(ctx: Context) -> Context:
        ctx.response = "blocked message"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store, persist_blocked=True)
    ctx = make_ctx(session_id="s1", blocked="prompt injection")

    await node.handle(ctx, blocked_next)

    assert len(store.calls) == 1
    assert store.calls[0]["messages"][1] == Message("assistant", "blocked message")


async def test_no_session_id_skips_persist():
    async def next_(ctx: Context) -> Context:
        ctx.response = "answer"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx()  # no session_id

    await node.handle(ctx, next_)

    assert store.calls == []


async def test_exception_downstream_saves_nothing_and_propagates():
    async def boom(ctx: Context) -> Context:
        raise ValueError("downstream broke")

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    with pytest.raises(ValueError, match="downstream broke"):
        await node.handle(ctx, boom)

    assert store.calls == []


async def test_cancellation_downstream_saves_nothing_and_propagates():
    started = asyncio.Event()

    async def hangs(ctx: Context) -> Context:
        started.set()
        await asyncio.sleep(5)
        ctx.response = "never reached"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    task = asyncio.create_task(node.handle(ctx, hangs))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.calls == []


async def test_tool_and_system_messages_are_not_persisted():
    async def react_like(ctx: Context) -> Context:
        ctx.messages.append(Message("assistant", "", tool_calls=[]))
        ctx.messages.append(Message("tool", "3", tool_call_id="1", name="calculator"))
        ctx.messages.append(Message("system", "<instruction>"))
        ctx.response = "the answer is 3"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    await node.handle(ctx, react_like)

    assert len(store.calls) == 1
    messages = store.calls[0]["messages"]
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages == [Message("user", "hello"), Message("assistant", "the answer is 3")]


async def test_artifact_and_summary_are_passed_through():
    async def next_(ctx: Context) -> Context:
        ctx.response = "answer"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1", artifact="# doc", summary={"text": "...", "up_to": "n3"})

    await node.handle(ctx, next_)

    call = store.calls[0]
    assert call["artifact"] == "# doc"
    assert call["summary"] == {"text": "...", "up_to": "n3"}


async def test_missing_response_persists_empty_assistant_message():
    async def next_(ctx: Context) -> Context:
        return ctx  # ctx.response stays None

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    await node.handle(ctx, next_)

    assert store.calls[0]["messages"][1] == Message("assistant", "")


async def test_parent_id_forwarded_to_append():
    async def next_(ctx: Context) -> Context:
        ctx.response = "answer"
        return ctx

    store = FakeConversationStore()
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1", parent_id="n7")

    await node.handle(ctx, next_)

    assert store.calls[0]["parent_id"] == "n7"


# --- from_config -----------------------------------------------------------------


class _Stores:
    def __init__(self, conversation) -> None:
        self.conversation = conversation


class _Deps:
    def __init__(self, conversation) -> None:
        self.stores = _Stores(conversation)


def test_from_config_wires_store_and_defaults_persist_blocked_false():
    store = FakeConversationStore()
    node = PersistNode.from_config({}, _Deps(store))

    assert node.store is store
    assert node.persist_blocked is False


def test_from_config_reads_persist_blocked():
    store = FakeConversationStore()
    node = PersistNode.from_config({"persist_blocked": True}, _Deps(store))

    assert node.persist_blocked is True


def test_from_config_ignores_dir_param():
    store = FakeConversationStore()
    node = PersistNode.from_config({"dir": "sessions/"}, _Deps(store))

    assert node.store is store
    assert node.persist_blocked is False


def test_from_config_unknown_param_raises():
    store = FakeConversationStore()
    with pytest.raises(ValueError):
        PersistNode.from_config({"bogus": 1}, _Deps(store))


# --- integration with the real ConversationStore, if it exists yet --------------

try:
    from storage.conversation import ConversationStore

    HAS_CONVERSATION_STORE = True
except ImportError:
    HAS_CONVERSATION_STORE = False


@pytest.mark.skipif(not HAS_CONVERSATION_STORE, reason="ConversationStore not implemented yet")
async def test_persist_node_with_real_in_memory_conversation_store():
    store = ConversationStore()  # None root -> in-memory
    node = PersistNode(store)
    ctx = make_ctx(session_id="s1")

    async def next_(ctx: Context) -> Context:
        ctx.response = "hi there"
        return ctx

    await node.handle(ctx, next_)

    head = store.head("s1")
    assert head is not None
    branch = store.branch("s1", head)
    assert [m.content for m in branch] == ["hello", "hi there"]
