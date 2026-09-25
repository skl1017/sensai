"""Unit tests for `ConversationStore` (wiki §7.2/§9.3, X2)."""

from __future__ import annotations

import json
import os

import pytest

from core.types import Message
from storage.conversation import ConversationStore

# -- in-memory (root=None) ---------------------------------------------------


def test_empty_session():
    store = ConversationStore()
    assert store.head("s") is None
    assert store.branch("s", None) == []
    assert store.load_artifact("s") is None
    assert store.load_summary("s") is None
    assert store.children("s", None) == []


def test_append_chains_messages_and_moves_head():
    store = ConversationStore()
    head = store.append("s", None, [Message("user", "a"), Message("assistant", "b")])
    assert store.head("s") == head
    assert store.branch("s", head) == [Message("user", "a"), Message("assistant", "b")]


def test_append_nothing_returns_parent():
    store = ConversationStore()
    assert store.append("s", None, []) is None
    head = store.append("s", None, [Message("user", "a")])
    assert store.append("s", head, []) == head
    assert store.head("s") == head  # unaffected by the no-op append


def test_node_ids_are_short_hex_and_unique():
    store = ConversationStore()
    store.append("s", None, [Message("user", "a"), Message("assistant", "b")])
    nodes = store._load("s")["nodes"]
    assert len(nodes) == 2
    for node_id in nodes:
        assert len(node_id) == 12
        int(node_id, 16)  # valid hex


def test_node_has_created_at_iso_utc():
    store = ConversationStore()
    head = store.append("s", None, [Message("user", "a")])
    node = store.node("s", head)
    assert node["created_at"]
    assert "+00:00" in node["created_at"] or node["created_at"].endswith("Z")


def test_root_to_node_reconstruction():
    store = ConversationStore()
    n1 = store.append("s", None, [Message("user", "q1")])
    n2 = store.append("s", n1, [Message("assistant", "a1")])
    n3 = store.append("s", n2, [Message("user", "q2")])
    path = store.path("s", n3)
    assert [n["id"] for n in path] == [n1, n2, n3]
    assert store.branch("s", n3) == [
        Message("user", "q1"),
        Message("assistant", "a1"),
        Message("user", "q2"),
    ]


def test_sibling_branch_leaves_original_intact():
    store = ConversationStore()
    first = store.append("s", None, [Message("user", "q1"), Message("assistant", "a1")])
    store.append("s", first, [Message("user", "q2"), Message("assistant", "a2")])
    alt = store.append("s", first, [Message("user", "q2b"), Message("assistant", "a2b")])
    assert store.head("s") == alt
    assert [m.content for m in store.branch("s", alt)] == ["q1", "a1", "q2b", "a2b"]
    assert [m.content for m in store.branch("s", first)] == ["q1", "a1"]


def test_children_of_a_branch_point():
    store = ConversationStore()
    first = store.append("s", None, [Message("user", "q1")])
    a = store.append("s", first, [Message("assistant", "a1")])
    b = store.append("s", first, [Message("assistant", "a1b")])
    child_ids = {n["id"] for n in store.children("s", first)}
    assert child_ids == {a, b}


def test_sessions_are_isolated():
    store = ConversationStore()
    store.append("a", None, [Message("user", "x")])
    assert store.head("b") is None


def test_unknown_parent_raises():
    store = ConversationStore()
    with pytest.raises(KeyError):
        store.branch("s", "nope")
    with pytest.raises(KeyError):
        store.append("s", "nope", [Message("user", "x")])
    with pytest.raises(KeyError):
        store.node("s", "nope")
    with pytest.raises(KeyError):
        store.set_head("s", "nope")


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a/b", "a b", "a.b", ""])
def test_invalid_session_id_rejected(bad_id):
    store = ConversationStore()
    with pytest.raises(ValueError):
        store.head(bad_id)


def test_valid_session_id_accepted():
    store = ConversationStore()
    assert store.head("Session-1_ok") is None  # doesn't raise


def test_interrupted_flag_on_last_node():
    store = ConversationStore()
    head = store.append(
        "s", None, [Message("user", "hi"), Message("assistant", "partial")], interrupted=True
    )
    node = store.node("s", head)
    assert node.get("interrupted") is True
    parent = store.path("s", head)[0]
    assert "interrupted" not in parent


def test_artifact_and_summary_round_trip():
    store = ConversationStore()
    assert store.load_artifact("s") is None
    assert store.load_summary("s") is None
    store.append("s", None, [Message("user", "hi")], artifact="# doc", summary={"last": "n1"})
    assert store.load_artifact("s") == "# doc"
    assert store.load_summary("s") == {"last": "n1"}
    # A later append without artifact/summary keeps the previous values.
    store.append("s", store.head("s"), [Message("assistant", "ok")])
    assert store.load_artifact("s") == "# doc"
    assert store.load_summary("s") == {"last": "n1"}


def test_set_head_moves_head_for_navigation():
    store = ConversationStore()
    first = store.append("s", None, [Message("user", "q1")])
    store.append("s", first, [Message("assistant", "a1")])
    store.set_head("s", first)
    assert store.head("s") == first


def test_list_sessions_empty_by_default():
    assert ConversationStore().list_sessions() == []


def test_list_sessions_newest_first_with_title():
    store = ConversationStore()
    store.append("old", None, [Message("user", "first question")])
    store.append("new", None, [Message("user", "second question"), Message("assistant", "ok")])
    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == ["new", "old"]
    assert sessions[0]["title"] == "second question"
    assert sessions[0]["messages"] == 2


def test_list_sessions_title_truncated():
    store = ConversationStore()
    long_text = "x" * 100
    store.append("s", None, [Message("user", long_text)])
    title = store.list_sessions()[0]["title"]
    assert len(title) == 60
    assert title.endswith("…")


# -- on disk (root=tmp_path) --------------------------------------------------


def test_persists_to_one_json_file_per_session(tmp_path):
    store = ConversationStore(root=tmp_path)
    store.append("s1", None, [Message("user", "hi")])
    path = tmp_path / "s1.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["session_id"] == "s1"
    assert data["head"] is not None
    assert len(data["nodes"]) == 1


def test_reload_from_disk_with_a_new_instance(tmp_path):
    store = ConversationStore(root=tmp_path)
    first = store.append("s", None, [Message("user", "q1"), Message("assistant", "a1")])
    store.append("s", first, [Message("user", "q2")], artifact="art", summary={"x": 1})

    reopened = ConversationStore(root=tmp_path)
    assert reopened.head("s") == store.head("s")
    assert reopened.branch("s", reopened.head("s")) == [
        Message("user", "q1"),
        Message("assistant", "a1"),
        Message("user", "q2"),
    ]
    assert reopened.load_artifact("s") == "art"
    assert reopened.load_summary("s") == {"x": 1}


def test_list_sessions_scans_root(tmp_path):
    writer = ConversationStore(root=tmp_path)
    writer.append("a", None, [Message("user", "hello")])

    reader = ConversationStore(root=tmp_path)  # never touched "a" before listing
    sessions = reader.list_sessions()
    assert [s["id"] for s in sessions] == ["a"]


def test_failed_write_leaves_file_and_memory_intact(tmp_path, monkeypatch):
    store = ConversationStore(root=tmp_path)
    first = store.append("s", None, [Message("user", "q1")])
    path = tmp_path / "s.json"
    before_bytes = path.read_bytes()
    before_head = store.head("s")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.append("s", first, [Message("assistant", "a1")])

    # File and in-memory state are untouched.
    assert path.read_bytes() == before_bytes
    assert store.head("s") == before_head

    # No leftover temp file in the session directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name != "s.json"]
    assert leftovers == []


def test_failed_write_on_first_append_leaves_no_temp_file(tmp_path, monkeypatch):
    store = ConversationStore(root=tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.append("s", None, [Message("user", "q1")])

    assert list(tmp_path.iterdir()) == []
    assert store.head("s") is None
