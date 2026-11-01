"""`CacheNode`: semantic cache in front of the model (wiki §7.4, #59/#60).

Sits right after `PersistNode` and before `ContextBuilderNode` (pipeline
position 4): a hit short-circuits the model, and since it is after persist the
cached exchange is still saved to the conversation. When `OutputGuardrailNode`
lands it goes *after* this one in the list, so the answer reaching the write
below is already masked and no PII is ever cached.

Hit: the question is embedded, the closest entry of the same scope is looked
up, and it is a hit when its similarity is >= `threshold` AND the lexical guard
(`_same_discriminators`) agrees. `ctx.response` is set to the cached answer,
`ctx.state["cache_hit"]` is True and `next` is not called. A hit streams no
token: entry points print `ctx.response` (cf `ui/cli.py`). Miss: the rest of
the chain runs, then the answer is stored (after purging expired entries).

`ctx.state["cache_hit"]` is owned by this node and is False on every other path.

Scope key: sha256 over (persona, `template@version`, profile). An entry is only
visible within its scope, so changing persona, prompt version (a pinned one or
a new latest file) or profile invalidates it. Resolved like `PersonaSection`.

Never read nor written (just `next`):
- `mode: first_turn_only` (default) when the history already holds a user or
  assistant message: a follow-up depends on that history. `always` lifts this.
- a turn with `state["rag_filters"]`: the answer depends on those filters, and
  an entry stored without them could cite documents they exclude.
- embedding or lookup failure: treated as a miss (warning logged).

Never written:
- empty answer, blocked request (`state["blocked"]`), truncated ReAct loop
  (`state["react_truncated"]`);
- a turn that used tools (`state["react_trace"]` non-empty): the answer may
  depend on changing data or side effects;
- a search filtered from the question (`state["rag_applied_filters"]`): a close
  question could carry other dates or another file.
A write error is logged, never raised.

Guard: embeddings do not separate a sentence from its negation (still 0.99 at
the chosen 0.96 threshold, see `bench/cache_threshold.md`), nor "2 + 2" from
"2 + 3". The cached and the new question must carry the same numbers and the
same negation words.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable

from core.llm import IEmbedder
from core.node import Next, PipelineNode
from core.types import Context
from nodes.persona_section import _load_persona
from storage.prompt_registry import PromptRegistry
from storage.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

_KNOWN_PARAMS = {"threshold", "ttl_hours", "mode", "embedding_dim"}
_MODES = ("first_turn_only", "always")

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_NEGATION_RE = re.compile(
    r"\b(?:ne|n'|pas|jamais|aucune?|rien|sans"
    r"|not|no|never|nothing|without|\w+n't)\b"
)


def _same_discriminators(a: str, b: str) -> bool:
    """True if `a` and `b` carry the same numbers and the same negation words."""
    a, b = a.lower().replace("’", "'"), b.lower().replace("’", "'")
    return Counter(_NUMBER_RE.findall(a)) == Counter(_NUMBER_RE.findall(b)) and Counter(
        _NEGATION_RE.findall(a)
    ) == Counter(_NEGATION_RE.findall(b))


def _scope_key(persona: str, template: str, version: str, profile: dict | None) -> str:
    raw = json.dumps([persona, f"{template}@{version}", profile or {}], sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _default_scope_fn(deps) -> Callable[[Context], Awaitable[str]]:
    """Scope from (persona, prompt template@version, profile), read fresh each turn."""
    personas_dir = deps.config_dir / "personas"
    registry = PromptRegistry(deps.config_dir / "prompts")
    profile_store = deps.stores.profile

    def compute(persona: str) -> str:
        template = _load_persona(personas_dir, persona)["prompt_template"]
        _, version = registry.get(template, deps.prompt_version)
        profile = profile_store.load() if profile_store is not None else None
        return _scope_key(persona, template, version, profile)

    async def scope_fn(ctx: Context) -> str:
        persona = ctx.state.get("persona") or deps.persona
        return await asyncio.to_thread(compute, persona)

    return scope_fn


class CacheNode(PipelineNode):
    def __init__(
        self,
        embedder: IEmbedder,
        cache: SemanticCache,
        scope_fn: Callable[[Context], Awaitable[str]],
        threshold: float = 0.96,
        ttl_hours: float = 24,
        mode: str = "first_turn_only",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"Invalid CacheNode mode {mode!r}: expected one of {', '.join(_MODES)}"
            )
        self.embedder = embedder
        self.cache = cache
        self.scope_fn = scope_fn
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self.mode = mode

    @classmethod
    def from_config(cls, params: dict, deps) -> CacheNode:
        """Build from `agent.yaml` params; wires `deps.embedder` and `deps.stores.semantic_cache`.

        `embedding_dim` is accepted but ignored here: it is consumed by
        `app._stores_from_config`, which sizes the cache's vector table.
        Any other unknown key, a missing embedder/cache or a bad `mode` raises `ValueError`.

        `deps` is intentionally untyped here (not `Deps`) so this module never has to
        import `app.py`, which would break the `nodes/` -> `core/` dependency rule.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown CacheNode param(s): {', '.join(sorted(unknown))}")
        if deps.embedder is None:
            raise ValueError("CacheNode requires an embedder (deps.embedder)")
        if deps.stores is None or deps.stores.semantic_cache is None:
            raise ValueError("CacheNode requires a semantic cache (deps.stores.semantic_cache)")
        kwargs = {key: params[key] for key in ("threshold", "ttl_hours", "mode") if key in params}
        return cls(deps.embedder, deps.stores.semantic_cache, _default_scope_fn(deps), **kwargs)

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.state["cache_hit"] = False

        if ctx.state.get("rag_filters"):
            return await next(ctx)

        if self.mode == "first_turn_only" and any(
            m.role in ("user", "assistant") for m in ctx.messages
        ):
            return await next(ctx)

        try:
            scope = await self.scope_fn(ctx)
            embedding = (await self.embedder.embed([ctx.user_input]))[0]
        except Exception:
            logger.warning("Cache: scope/embedding failed, skipping cache", exc_info=True)
            return await next(ctx)

        try:
            hit = await self.cache.lookup(scope, embedding, self.ttl_seconds)
        except Exception:
            logger.warning("Cache: lookup failed, treating as a miss", exc_info=True)
            hit = None

        if (
            hit is not None
            and hit.similarity >= self.threshold
            and _same_discriminators(hit.question, ctx.user_input)
        ):
            ctx.response = hit.response
            ctx.state["cache_hit"] = True
            logger.info("Cache hit (similarity %.3f)", hit.similarity)
            return ctx

        ctx = await next(ctx)

        state = ctx.state
        if (
            not (ctx.response or "").strip()
            or state.get("blocked")
            or state.get("react_truncated")
            or state.get("react_trace")
            or state.get("rag_applied_filters")
        ):
            return ctx

        try:
            await self.cache.purge_expired(self.ttl_seconds)
            await self.cache.store(scope, ctx.user_input, embedding, ctx.response)
        except Exception:
            logger.warning("Cache: write failed", exc_info=True)
        return ctx
