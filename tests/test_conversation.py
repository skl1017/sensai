"""Unit tests for the in-memory `Conversation` tree."""

from __future__ import annotations

import pytest

from core.types import Message
from storage.conversation import Conversation


def test_empty_session():
    conv = Conversation()
    assert conv.head("s") is None
    assert conv.branch("s", None) == []
    assert conv.load_artifact("s") is None
    assert conv.load_summary("s") is None


def test_append_chains_messages_and_moves_head():
    conv = Conversation()
    head = conv.append("s", None, [Message("user", "a"), Message("assistant", "b")])
    assert conv.head("s") == head
    assert conv.branch("s", head) == [Message("user", "a"), Message("assistant", "b")]


def test_append_nothing_returns_parent():
    assert Conversation().append("s", None, []) is None


def test_sibling_branch_leaves_original_intact():
    conv = Conversation()
    first = conv.append("s", None, [Message("user", "q1"), Message("assistant", "a1")])
    conv.append("s", first, [Message("user", "q2"), Message("assistant", "a2")])
    alt = conv.append("s", first, [Message("user", "q2b"), Message("assistant", "a2b")])
    assert conv.head("s") == alt
    assert [m.content for m in conv.branch("s", alt)] == ["q1", "a1", "q2b", "a2b"]
    assert [m.content for m in conv.branch("s", first)] == ["q1", "a1"]


def test_sessions_are_isolated():
    conv = Conversation()
    conv.append("a", None, [Message("user", "x")])
    assert conv.head("b") is None


def test_unknown_parent_raises():
    conv = Conversation()
    with pytest.raises(KeyError):
        conv.branch("s", "nope")
    with pytest.raises(KeyError):
        conv.append("s", "nope", [Message("user", "x")])
