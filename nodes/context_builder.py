"""Minimal `ContextBuilderNode`: injects the user profile before the ReAct loop (#25).

This is deliberately the smallest version of the node described in the wiki
(§9.1, pipeline position 5): a single `ContextSection` (`ProfileSection`)
exists so far, and there is no token-budget/summarization pass yet. Story 6
extends this same node with the other sections envisioned there (persona,
memory, artifact, rag, fewshot) and the compression pass — adding one is
meant to be a new `ContextSection` subclass wired in `from_config`, not a
change to `ContextBuilderNode.handle` itself.

Sections run concurrently (`asyncio.gather`), are sorted by `priority`, and
are joined into a single system `Message`. `ctx.messages` is then rebuilt as
a brand-new list — the caller's own history list (owned by `app.chat_turn`)
is never mutated, matching the `Context` invariant that nodes only ever
append to `ctx.messages`, never mutate it in place, except here where the
whole point of this node is to rebuild it fresh every turn.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from core.node import Next, PipelineNode
from core.types import Context, Message

logger = logging.getLogger(__name__)

_STORY_6_SECTIONS = {"persona", "memory", "artifact", "rag", "fewshot"}
_KNOWN_PARAMS = {"sections", "profile_path"}


class ContextSection(ABC):
    """One pluggable slice of the system prompt, ordered by `priority` (lower first)."""

    priority: int

    @abstractmethod
    async def build(self, ctx: Context) -> str | None:
        """Render this section's text for `ctx`, or `None` to contribute nothing."""


class ProfileSection(ContextSection):
    """Renders the static user profile (preferences + custom instructions) — #25.

    The profile is read fresh via `asyncio.to_thread(profile_store.load)` on
    *every* turn rather than cached at construction time, so a profile edited
    between turns (the CLI's `/profile set ...`) is reflected on the very
    next request. Returns `None` when the profile has neither preferences
    nor instructions, so an empty profile never adds an empty system message.
    """

    def __init__(self, profile_store, priority: int = 20) -> None:
        self.profile_store = profile_store
        self.priority = priority

    async def build(self, ctx: Context) -> str | None:
        profile = await asyncio.to_thread(self.profile_store.load)
        preferences = profile.get("preferences") or {}
        instructions = (profile.get("instructions") or "").strip()

        if not preferences and not instructions:
            return None

        lines = ["User profile:"]
        lines.extend(f"- {key}: {value}" for key, value in preferences.items())
        if instructions:
            lines.append(instructions)
        return "\n".join(lines)


class ContextBuilderNode(PipelineNode):
    def __init__(self, sections: list[ContextSection]) -> None:
        self.sections = sections

    @classmethod
    def from_config(cls, params: dict, deps) -> ContextBuilderNode:
        """Build from `agent.yaml` params: `{sections: [profile]}`.

        `profile_path` is accepted but ignored here: it is consumed by
        `app.open_stores`, which builds the concrete `ProfileStore`. Unknown
        param keys raise `ValueError`, and an unknown section name raises
        `ValueError` too — including the Story 6 names (`persona`, `memory`,
        `artifact`, `rag`, `fewshot`), which aren't available yet.

        `deps` is intentionally untyped here (not `Deps`) so this module
        never has to import `app.py`, which would break the `nodes/` ->
        `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown ContextBuilderNode param(s): {', '.join(sorted(unknown))}")

        names = params.get("sections", ["profile"])
        sections = [cls._build_section(name, deps) for name in names]
        return cls(sections)

    @staticmethod
    def _build_section(name: str, deps) -> ContextSection:
        if name == "profile":
            return ProfileSection(deps.stores.profile)
        if name in _STORY_6_SECTIONS:
            raise ValueError(f"Context section {name!r} is not available yet (Story 6)")
        raise ValueError(f"Unknown context section {name!r}")

    async def handle(self, ctx: Context, next: Next) -> Context:
        results = await asyncio.gather(
            *(section.build(ctx) for section in self.sections),
            return_exceptions=True,
        )

        built: list[tuple[int, str]] = []
        for section, result in zip(self.sections, results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                logger.warning(
                    "Context section %s failed, skipping it: %s",
                    type(section).__name__,
                    result,
                )
                continue
            if result is not None:
                built.append((section.priority, result))

        built.sort(key=lambda item: item[0])

        history = [m for m in ctx.messages if m.role != "system"]
        system = [Message("system", "\n\n".join(text for _, text in built))] if built else []
        ctx.messages = system + history + [Message("user", ctx.user_input)]

        return await next(ctx)
