"""`ContextBuilderNode`: assembles the system prompt and fits it to budget (#25, Story 6).

Pipeline position 5 (wiki §9.1). Each turn:

1. Query rewrite (optional, `rewrite_query: true`): `ctx.state["query"]` is
   set to a standalone rewrite of `ctx.user_input` (`nodes/query_rewriter.py`),
   computed *before* the sections below run, so a section that reads
   `ctx.state.get("query")` (`FewShotSection` today, `RagSection` later) sees
   the rewritten form.
2. Sections (`ContextSection` instances, one per `sections:` entry) build
   concurrently (`asyncio.gather`), sorted by `priority` (lower first) and
   joined with a blank line into a single system `Message`. A section that
   raises is logged and skipped, so one bad section never fails the whole
   turn; `asyncio.CancelledError` always propagates. Available sections:
   `profile` (`ProfileSection`, static user preferences/instructions),
   `persona` (`PersonaSection`, the active persona's rendered prompt
   template) and `fewshot` (`FewShotSection`, the closest bank examples).
   `memory`, `artifact` and `rag` aren't implemented yet (Story 10, 15, 11).
3. `ctx.messages` is rebuilt from scratch: `[system?] + history +
   [current user message]` — the caller's own history list is never
   mutated, matching the `Context` invariant that nodes only ever append to
   `ctx.messages`, except here where the whole point of this node is to
   rebuild it fresh every turn.
4. Token budget (optional, `budget: {...}`): `nodes/context_budget.py`'s
   `apply_budget` fits the rebuilt `ctx.messages` under the LLM's context
   window, folding old turns into a rolling summary (or dropping them, on
   LLM failure). Absent `budget`, no compression pass runs at all.

`ctx.state` keys written here: `query` (step 1), `prompt` (by
`PersonaSection`, when the `persona` section is enabled), `summary` and
`context_stats` (by `apply_budget`, when `budget` is configured).
"""

from __future__ import annotations

import asyncio
import logging

from core.node import Next, PipelineNode
from core.types import Context, Message
from nodes.context_budget import apply_budget
from nodes.context_section import ContextSection
from nodes.fewshot_section import FewShotSection
from nodes.persona_section import PersonaSection
from nodes.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)

# Section names described in the wiki that aren't implemented yet, and the
# story that will add them.
_UNAVAILABLE_SECTIONS = {
    "memory": "Story 10",
    "artifact": "Story 15",
    "rag": "Story 11",
}

_KNOWN_PARAMS = {"sections", "profile_path", "budget", "rewrite_query", "fewshot"}
_BUDGET_KEYS = {"reserve_output", "trigger_ratio", "keep_last_turns"}


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


def _validate_budget(raw: dict) -> dict:
    """Validate the `budget:` sub-mapping and return it, unfilled (`apply_budget` has defaults)."""
    if not isinstance(raw, dict):
        raise ValueError(f"budget: expected a mapping, got {type(raw).__name__}")

    unknown = set(raw) - _BUDGET_KEYS
    if unknown:
        raise ValueError(f"Unknown budget param(s): {', '.join(sorted(unknown))}")

    validated: dict[str, int | float] = {}
    for key in ("reserve_output", "keep_last_turns"):
        if key in raw:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"budget.{key}: expected an int >= 0, got {value!r}")
            validated[key] = value

    if "trigger_ratio" in raw:
        value = raw["trigger_ratio"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < value <= 1):
            raise ValueError(f"budget.trigger_ratio: expected a number in (0, 1], got {value!r}")
        validated["trigger_ratio"] = value

    return validated


class ContextBuilderNode(PipelineNode):
    def __init__(
        self,
        sections: list[ContextSection],
        budget: dict | None = None,
        rewrite_query: bool = False,
    ) -> None:
        self.sections = sections
        self.budget = budget
        self.rewrite_query = rewrite_query

    @classmethod
    def from_config(cls, params: dict, deps) -> ContextBuilderNode:
        """Build from `agent.yaml` params.

        `{sections, profile_path, budget, rewrite_query, fewshot}` are the
        only known keys; anything else raises `ValueError`. `profile_path`
        is accepted but ignored here: it is consumed by `app.open_stores`,
        which builds the concrete `ProfileStore`. `fewshot` is the sub-dict
        forwarded to `FewShotSection.from_config` when `fewshot` is among
        `sections` (default `{}`). `budget`, when present, is validated by
        `_validate_budget`; absent, no compression pass runs. `rewrite_query`
        defaults to `False`.

        An unknown section name raises `ValueError`, and so do the
        not-yet-implemented ones (`memory`, `artifact`, `rag`), each naming
        the story that will add it.

        `deps` is intentionally untyped here (not `Deps`) so this module
        never has to import `app.py`, which would break the `nodes/` ->
        `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown ContextBuilderNode param(s): {', '.join(sorted(unknown))}")

        names = params.get("sections", ["profile"])
        fewshot_params = params.get("fewshot", {})
        sections = [cls._build_section(name, fewshot_params, deps) for name in names]

        budget = params.get("budget")
        if budget is not None:
            budget = _validate_budget(budget)

        rewrite = params.get("rewrite_query", False)
        if not isinstance(rewrite, bool):
            raise ValueError(f"rewrite_query: expected a bool, got {type(rewrite).__name__}")

        return cls(sections, budget=budget, rewrite_query=rewrite)

    @staticmethod
    def _build_section(name: str, fewshot_params: dict, deps) -> ContextSection:
        if name == "profile":
            return ProfileSection(deps.stores.profile)
        if name == "persona":
            return PersonaSection.from_deps(deps)
        if name == "fewshot":
            return FewShotSection.from_config(fewshot_params, deps)
        if name in _UNAVAILABLE_SECTIONS:
            story = _UNAVAILABLE_SECTIONS[name]
            raise ValueError(f"Context section {name!r} is not available yet ({story})")
        raise ValueError(f"Unknown context section {name!r}")

    async def handle(self, ctx: Context, next: Next) -> Context:
        history = [m for m in ctx.messages if m.role != "system"]

        if self.rewrite_query:
            ctx.state["query"] = await rewrite_query(ctx.llm, history, ctx.user_input)

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

        system = [Message("system", "\n\n".join(text for _, text in built))] if built else []
        ctx.messages = system + history + [Message("user", ctx.user_input)]

        if self.budget is not None:
            ctx.messages = await apply_budget(ctx.messages, ctx.llm, ctx.state, **self.budget)

        return await next(ctx)
