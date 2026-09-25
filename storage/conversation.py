"""Durable conversation tree (wiki §7.2/§9.3, X2).

Each exchange message is a node `{id, parent_id, role, content, created_at}`;
a session is a tree, and editing an old message is appending under an older
parent, which creates a sibling branch while the original stays intact.

`ConversationStore(root=None)` keeps everything in memory (the default for
tests and `InMemoryStores`). Given a `root` directory, each session is a
single JSON file `root/<session_id>.json` holding
`{session_id, created_at, updated_at, head, nodes: {id: node}, artifact,
summary}`. Sessions are loaded from disk lazily, on first access, and every
write (`append`/`set_head`) replaces the whole file atomically: the new
session dict is built on a copy, written to a temp file in `root`, `fsync`ed
and `os.replace`d over the old file, and the in-memory copy is only swapped
in once that write has actually succeeded. A failed write therefore leaves
both the on-disk file and the in-memory state exactly as they were.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from core.types import Message
from storage._atomic import atomic_write

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"Invalid session id {session_id!r}: must match {_SESSION_ID_RE.pattern!r}"
        )


def _empty_session(session_id: str) -> dict:
    now = _now()
    return {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "head": None,
        "nodes": {},
        "artifact": None,
        "summary": None,
    }


def _title(nodes: dict[str, dict], limit: int = 60) -> str:
    """First user message, truncated; `""` if the session has none (yet)."""
    for node in nodes.values():
        if node["role"] == "user":
            content = node["content"]
            return content if len(content) <= limit else content[: limit - 1] + "…"
    return ""


class ConversationStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict] = {}

    # -- internal helpers ----------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        assert self._root is not None
        return self._root / f"{session_id}.json"

    def _load(self, session_id: str) -> dict:
        """Session dict for `session_id`, loading it from disk lazily on first access."""
        _validate_session_id(session_id)
        if session_id in self._sessions:
            return self._sessions[session_id]
        if self._root is not None:
            path = self._session_path(session_id)
            if path.exists():
                session = json.loads(path.read_text(encoding="utf-8"))
                self._sessions[session_id] = session
                return session
        session = _empty_session(session_id)
        self._sessions[session_id] = session
        return session

    def _save(self, session_id: str, updated: dict) -> None:
        """Write `updated` atomically (if on disk) and only then swap it into memory."""
        if self._root is not None:
            atomic_write(self._session_path(session_id), json.dumps(updated, indent=2))
        self._sessions[session_id] = updated

    # -- read API ---------------------------------------------------------

    def head(self, session_id: str) -> str | None:
        """Id of the current head node, or `None` for an empty session."""
        return self._load(session_id)["head"]

    def node(self, session_id: str, node_id: str) -> dict:
        nodes = self._load(session_id)["nodes"]
        if node_id not in nodes:
            raise KeyError(f"Unknown node {node_id!r} in session {session_id!r}")
        return nodes[node_id]

    def path(self, session_id: str, node_id: str) -> list[dict]:
        """Nodes from the root down to `node_id`, inclusive."""
        nodes = self._load(session_id)["nodes"]
        chain: list[dict] = []
        current: str | None = node_id
        while current is not None:
            if current not in nodes:
                raise KeyError(f"Unknown node {current!r} in session {session_id!r}")
            node = nodes[current]
            chain.append(node)
            current = node["parent_id"]
        chain.reverse()
        return chain

    def children(self, session_id: str, node_id: str | None = None) -> list[dict]:
        nodes = self._load(session_id)["nodes"]
        if node_id is not None and node_id not in nodes:
            raise KeyError(f"Unknown node {node_id!r} in session {session_id!r}")
        return [node for node in nodes.values() if node["parent_id"] == node_id]

    def branch(self, session_id: str, parent_id: str | None) -> list[Message]:
        """Messages from the root down to `parent_id` (inclusive); `[]` for `None`."""
        self._load(session_id)  # validates the session id even for the empty case
        if parent_id is None:
            return []
        nodes = self.path(session_id, parent_id)
        return [Message(node["role"], node["content"]) for node in nodes]

    def load_artifact(self, session_id: str) -> str | None:
        return self._load(session_id).get("artifact")

    def load_summary(self, session_id: str) -> dict | None:
        return self._load(session_id).get("summary")

    def list_sessions(self) -> list[dict]:
        """`{id, created_at, updated_at, title, messages}` for every non-empty session.

        Newest first (by `updated_at`). Scans `root` on disk when persisted, so
        it reflects sessions written by other instances/processes, not just
        this one's cache.
        """
        sessions: list[dict]
        if self._root is not None:
            sessions = []
            for path in self._root.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("nodes"):
                    sessions.append(data)
        else:
            sessions = [session for session in self._sessions.values() if session["nodes"]]

        sessions.sort(key=lambda session: session["updated_at"], reverse=True)
        return [
            {
                "id": session["session_id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "title": _title(session["nodes"]),
                "messages": len(session["nodes"]),
            }
            for session in sessions
        ]

    # -- write API ----------------------------------------------------------

    def append(
        self,
        session_id: str,
        parent_id: str | None,
        messages: list[Message],
        *,
        artifact: str | None = None,
        summary: dict | None = None,
        interrupted: bool = False,
    ) -> str | None:
        """Chain `messages` under `parent_id`; return the new head id (`parent_id` if empty).

        Also stores `artifact`/`summary` when given, and marks the last appended
        node `interrupted` when asked. Everything lands in one atomic write.
        """
        current = self._load(session_id)
        if parent_id is not None and parent_id not in current["nodes"]:
            raise KeyError(f"Unknown node {parent_id!r} in session {session_id!r}")

        nodes = dict(current["nodes"])
        new_parent = parent_id
        last_id: str | None = None
        for message in messages:
            node_id = uuid.uuid4().hex[:12]
            nodes[node_id] = {
                "id": node_id,
                "parent_id": new_parent,
                "role": message.role,
                "content": message.content,
                "created_at": _now(),
            }
            new_parent = node_id
            last_id = node_id

        if last_id is not None and interrupted:
            nodes[last_id] = {**nodes[last_id], "interrupted": True}

        updated = {
            **current,
            "nodes": nodes,
            "head": new_parent if last_id is not None else current["head"],
            "updated_at": _now(),
        }
        if artifact is not None:
            updated["artifact"] = artifact
        if summary is not None:
            updated["summary"] = summary

        self._save(session_id, updated)
        return new_parent

    def set_head(self, session_id: str, node_id: str) -> None:
        """Move the session head to `node_id` (branch navigation)."""
        current = self._load(session_id)
        if node_id not in current["nodes"]:
            raise KeyError(f"Unknown node {node_id!r} in session {session_id!r}")
        updated = {**current, "head": node_id, "updated_at": _now()}
        self._save(session_id, updated)
