"""`Stores`: the storage handles shared through `Deps` (wiki §9.1).

A Protocol so nodes/tools depend on the shape, not on concrete storage classes.
Attributes are typed `Any` until each storage backend lands with its own story.
"""

from __future__ import annotations

from typing import Any, Protocol


class Stores(Protocol):
    conversation: Any
    memory_db: Any
    vector_store: Any
    journal: Any
