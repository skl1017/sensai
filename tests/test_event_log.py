"""Tests for `storage/event_log.py`: both backends, patches, ordering."""

from __future__ import annotations

import json
import sqlite3

import pytest

from storage.event_log import EventLog, read_events


@pytest.fixture(params=["events.jsonl", "events.db", "events.sqlite"])
def log_path(request, tmp_path):
    return tmp_path / "nested" / "dir" / request.param


async def test_write_and_read_in_order(log_path):
    log = EventLog.open(log_path)
    for i in range(3):
        await log.write({"trace_id": f"t{i}", "ts": f"2026-01-0{i + 1}", "n": i})
    events = read_events(log_path)
    assert [e["trace_id"] for e in events] == ["t0", "t1", "t2"]
    assert events[1]["n"] == 1


async def test_update_merges_patch(log_path):
    log = EventLog.open(log_path)
    await log.write({"trace_id": "a", "ts": "x", "eval_scores": None, "status": "ok"})
    await log.write({"trace_id": "b", "ts": "y", "eval_scores": None})
    await log.update("a", {"eval_scores": {"relevance": 4}})
    events = read_events(log_path)
    assert len(events) == 2
    assert events[0]["eval_scores"] == {"relevance": 4}
    assert events[0]["status"] == "ok"
    assert "_patch" not in events[0]
    assert events[1]["eval_scores"] is None


async def test_update_before_write_is_kept(log_path):
    log = EventLog.open(log_path)
    await log.update("a", {"eval_scores": {"relevance": 1}})
    await log.write({"trace_id": "a", "ts": "x", "eval_scores": None})
    events = read_events(log_path)
    assert len(events) == 1
    assert events[0]["eval_scores"] == {"relevance": 1}


def test_read_missing_file(tmp_path):
    assert read_events(tmp_path / "nope.jsonl") == []
    assert read_events(tmp_path / "nope.db") == []


def test_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        EventLog.open(tmp_path / "events.txt")


async def test_jsonl_format_and_unicode(tmp_path):
    path = tmp_path / "e.jsonl"
    log = EventLog.open(path)
    await log.write({"trace_id": "a", "response": "café"})
    await log.update("a", {"x": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1]) == {"_patch": "a", "x": 1}
    assert "café" in lines[0]


async def test_sqlite_schema(tmp_path):
    path = tmp_path / "e.db"
    log = EventLog.open(path)
    await log.write({"trace_id": "a", "ts": "t1"})
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT trace_id, ts, data FROM events").fetchone()
    finally:
        conn.close()
    assert row[0] == "a" and row[1] == "t1" and json.loads(row[2])["trace_id"] == "a"


def test_jsonl_ignores_torn_line(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"trace_id": "a"}\n{"trace_id": "b"\n', encoding="utf-8")
    assert [e["trace_id"] for e in read_events(path)] == ["a"]
