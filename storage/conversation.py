"""In-memory conversation tree (wiki §9.3, X2).

Each exchange message is a node `{id, parent_id, role, content}`; a session is
a tree, and editing an old message is appending under an older parent, which
creates a sibling branch while the original stays intact. Only what
`app.chat_turn` needs lives here; persistence (SQLite, atomic writes) and
artifact/summary storage come with their own stories.
"""

from __future__ import annotations

import itertools

from core.types import Message


class Conversation:
    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, dict]] = {}
        self._heads: dict[str, str] = {}
        self._ids = itertools.count(1)

    def head(self, session_id: str) -> str | None:
        """Id of the most recently appended node, or `None` for an empty session."""
        return self._heads.get(session_id)

    def branch(self, session_id: str, parent_id: str | None) -> list[Message]:
        """Messages from the root down to `parent_id` (inclusive); `[]` for `None`."""
        nodes = self._nodes.get(session_id, {})
        chain: list[Message] = []
        current = parent_id
        while current is not None:
            if current not in nodes:
                raise KeyError(f"Unknown node {current!r} in session {session_id!r}")
            node = nodes[current]
            chain.append(Message(node["role"], node["content"]))
            current = node["parent_id"]
        chain.reverse()
        return chain

    def append(self, session_id: str, parent_id: str | None, messages: list[Message]) -> str | None:
        """Chain `messages` under `parent_id`; return the new head id (`parent_id` if empty)."""
        nodes = self._nodes.setdefault(session_id, {})
        if parent_id is not None and parent_id not in nodes:
            raise KeyError(f"Unknown node {parent_id!r} in session {session_id!r}")
        for message in messages:
            node_id = f"n{next(self._ids)}"
            nodes[node_id] = {
                "id": node_id,
                "parent_id": parent_id,
                "role": message.role,
                "content": message.content,
            }
            parent_id = node_id
            self._heads[session_id] = node_id
        return parent_id

    def load_artifact(self, session_id: str) -> str | None:
        return None

    def load_summary(self, session_id: str) -> dict | None:
        return None
