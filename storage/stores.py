"""`Stores`: the storage handles shared through `Deps` (wiki §9.1).

A Protocol so nodes/tools depend on the shape, not on concrete storage classes.
Attributes are typed `Any` until each storage backend lands with its own story.
"""

from __future__ import annotations

from typing import Any, Protocol

from storage.conversation import ConversationStore
from storage.profile import ProfileStore


class Stores(Protocol):
    conversation: Any
    profile: Any
    memory_db: Any
    vector_store: Any
    journal: Any


class InMemoryStores:
    """Concrete `Stores`: conversation and profile default to in-memory-backed
    instances, the rest is `None` until their stories land."""

    def __init__(self, conversation: Any = None, profile: Any = None) -> None:
        self.conversation = conversation if conversation is not None else ConversationStore()
        self.profile = profile if profile is not None else ProfileStore(None)
        self.memory_db: Any = None
        self.vector_store: Any = None
        self.journal: Any = None
