"""`ContextSection`: one pluggable slice of the system prompt (wiki §7.5).

Kept in its own module so section implementations (`persona_section.py`,
`fewshot_section.py`, ...) can subclass it without importing
`context_builder.py`, which in turn imports them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Context


class ContextSection(ABC):
    """One pluggable slice of the system prompt, ordered by `priority` (lower first)."""

    priority: int

    @abstractmethod
    async def build(self, ctx: Context) -> str | None:
        """Render this section's text for `ctx`, or `None` to contribute nothing."""
