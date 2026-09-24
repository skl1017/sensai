"""`Stores`: the storage handles shared through `Deps` (wiki §9.1).

A Protocol so nodes/tools depend on the shape, not on concrete storage classes.
Attributes are typed `Any` until each storage backend lands with its own story.
"""

from __future__ import annotations

from typing import Any, Protocol

from storage.conversation import Conversation


class Stores(Protocol):
    conversation: Any
    memory_db: Any
    vector_store: Any
    journal: Any


class InMemoryStores:
    """Concrete `Stores`: only the conversation exists so far, the rest is `None`."""

    def __init__(self, conversation: Any = None) -> None:
        self.conversation = conversation if conversation is not None else Conversation()
        self.memory_db: Any = None
        self.vector_store: Any = None
        self.journal: Any = None
