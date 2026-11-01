"""Event log: one JSON event per request, JSONL or SQLite backend (EV3).

`EventLog.open(path)` picks the backend from the extension (`.jsonl` appends
lines, `.db`/`.sqlite` uses stdlib `sqlite3`). `write` stores a full event,
`update` merges a patch into an already written one (used by background
evaluation, which finishes after the request). Blocking I/O goes through
`asyncio.to_thread`. `read_events(path)` is the synchronous reader (dashboard,
tests): it returns events in write order with the patches merged.

JSONL: an update is a `{"_patch": trace_id, ...fields}` line, applied by
`read_events`. SQLite: table `events(trace_id TEXT PRIMARY KEY, ts TEXT,
data TEXT)`, `data` being the event as JSON, patched in place.

Depends only on the standard library.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

_JSONL_SUFFIXES = {".jsonl"}
_SQLITE_SUFFIXES = {".db", ".sqlite"}

_SCHEMA = "CREATE TABLE IF NOT EXISTS events (trace_id TEXT PRIMARY KEY, ts TEXT, data TEXT)"


def _dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class EventLog(ABC):
    @classmethod
    def open(cls, path: str | Path) -> EventLog:
        """Open (creating parent dirs) the log at `path`; the extension picks the backend."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in _JSONL_SUFFIXES:
            return JsonlEventLog(path)
        if suffix in _SQLITE_SUFFIXES:
            return SqliteEventLog(path)
        raise ValueError(
            f"Unsupported event log extension {path.suffix!r}: use .jsonl, .db or .sqlite"
        )

    @abstractmethod
    async def write(self, event: dict) -> None: ...

    @abstractmethod
    async def update(self, trace_id: str, patch: dict) -> None: ...


class JsonlEventLog(EventLog):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict) -> None:
        line = _dumps(record) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def write(self, event: dict) -> None:
        await asyncio.to_thread(self._append, event)

    async def update(self, trace_id: str, patch: dict) -> None:
        await asyncio.to_thread(self._append, {**patch, "_patch": trace_id})


class SqliteEventLog(EventLog):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _upsert(self, trace_id: str, patch: dict, replace: bool) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    row = conn.execute(
                        "SELECT data FROM events WHERE trace_id = ?", (trace_id,)
                    ).fetchone()
                    existing = json.loads(row[0]) if row else {}
                    # write(): an early patch (update before write) survives under the event.
                    data = {**patch, **existing} if replace else {**existing, **patch}
                    data["trace_id"] = trace_id
                    if row:  # UPDATE keeps the rowid, hence the write order
                        conn.execute(
                            "UPDATE events SET ts = ?, data = ? WHERE trace_id = ?",
                            (data.get("ts"), _dumps(data), trace_id),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO events (trace_id, ts, data) VALUES (?, ?, ?)",
                            (trace_id, data.get("ts"), _dumps(data)),
                        )
            finally:
                conn.close()

    async def write(self, event: dict) -> None:
        await asyncio.to_thread(self._upsert, str(event["trace_id"]), event, True)

    async def update(self, trace_id: str, patch: dict) -> None:
        await asyncio.to_thread(self._upsert, trace_id, patch, False)


def read_events(path: str | Path) -> list[dict]:
    """Read every event of the log at `path` (synchronous), patches merged, in write order."""
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() in _SQLITE_SUFFIXES:
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute("SELECT data FROM events ORDER BY rowid").fetchall()
        finally:
            conn.close()
        return [json.loads(r[0]) for r in rows]

    events: list[dict] = []
    patches: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line must not make the whole log unreadable
            (patches if "_patch" in record else events).append(record)
    by_id = {e.get("trace_id"): e for e in events}
    for patch in patches:
        target = by_id.get(patch["_patch"])
        if target is not None:
            target.update({k: v for k, v in patch.items() if k != "_patch"})
    return events
