"""Long-term memory tools over `MemoryDB` (wiki §8.7): search, save, update, forget.

Four small `ITool`s share one `MemoryDB` and an `IEmbedder`: queries and
"key: value" fact texts are embedded for similarity search. `memory_forget` is destructive but
is NOT wrapped here: put it under `approval` in `agent.yaml`.
"""

from __future__ import annotations

from core import runtime
from core.llm import IEmbedder
from core.tool import ITool
from storage.memory_db import Fact, MemoryDB

MAX_VALUE_CHARS = 500
MAX_FACTS_PER_SAVE = 20
MAX_RELATIONS_PER_SAVE = 20
SEARCH_LIMIT = 10
NO_MATCH = "No matching memory."


def _cut(text: str) -> str:
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return _cut(value.strip())


def _fact_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("id must be an integer (the id shown by memory_search)")
    return value


def _line(fact: Fact) -> str:
    return f"[{fact.id}] {fact.entity} / {fact.key}: {_cut(fact.value)}"


class _MemoryTool(ITool):
    def __init__(self, db: MemoryDB, embedder: IEmbedder) -> None:
        self.db = db
        self.embedder = embedder


class MemorySearchTool(_MemoryTool):
    name = "memory_search"
    description = (
        "Search long-term memory for facts about the user or the world. Use it before "
        "answering anything that may depend on something remembered from earlier "
        "(preferences, names, projects, decisions). Returns up to 10 facts with their id."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for."}},
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(self, query: str) -> str:
        query = _text(query, "query")
        (vector,) = await self.embedder.embed([query])
        facts = await self.db.asearch(vector, SEARCH_LIMIT)
        return "\n".join(_line(f) for f in facts) if facts else NO_MATCH


class MemorySaveTool(_MemoryTool):
    name = "memory_save"
    description = (
        "Remember durable information. Use it when the user states a lasting fact "
        "(preference, name, project, relationship) worth recalling in later sessions. "
        "Saving an existing entity/key overwrites its value. Group the facts of one "
        "entity in a single call; do not store trivia or secrets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Who or what the facts are about."},
            "entity_type": {"type": "string", "description": "Optional kind, e.g. 'person'."},
            "facts": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Fact key -> value, e.g. {'favorite_color': 'green'}.",
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rel": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["rel", "target"],
                    "additionalProperties": False,
                },
                "description": "Optional links to other entities, e.g. works_at -> Acme.",
            },
        },
        "required": ["entity", "facts"],
        "additionalProperties": False,
    }

    async def run(
        self,
        entity: str,
        facts: dict[str, str],
        entity_type: str | None = None,
        relations: list[dict[str, str]] | None = None,
    ) -> str:
        entity = _text(entity, "entity")
        if not isinstance(facts, dict) or not facts:
            raise ValueError("facts must be a non-empty object of key -> value")
        if len(facts) > MAX_FACTS_PER_SAVE:
            raise ValueError(f"Too many facts ({len(facts)}, max {MAX_FACTS_PER_SAVE})")
        clean = {_text(k, "fact key"): _text(v, "fact value") for k, v in facts.items()}
        rels = relations or []
        if len(rels) > MAX_RELATIONS_PER_SAVE:
            raise ValueError(f"Too many relations ({len(rels)}, max {MAX_RELATIONS_PER_SAVE})")
        pairs = []
        for rel in rels:
            if not isinstance(rel, dict):
                raise ValueError("each relation must be an object with 'rel' and 'target'")
            pairs.append(
                (_text(rel.get("rel"), "relation rel"), _text(rel.get("target"), "target"))
            )
        vectors = await self.embedder.embed([f"{k}: {v}" for k, v in clean.items()])
        embeddings = dict(zip(clean, vectors, strict=True))
        if entity_type:
            await self.db.aupsert_entity(entity, _text(entity_type, "entity_type"))
        ids = await self.db.asave(
            entity, clean, pairs, session=runtime.session_id.get(), embeddings=embeddings
        )
        return f"Saved {len(ids)} fact(s) about {entity} (ids: {', '.join(map(str, ids))})."


class MemoryUpdateTool(_MemoryTool):
    name = "memory_update"
    description = (
        "Change the value of an existing remembered fact. Use it when the user corrects "
        "or updates something already in memory: find its id with memory_search first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Fact id from memory_search."},
            "value": {"type": "string", "description": "The new value."},
        },
        "required": ["id", "value"],
        "additionalProperties": False,
    }

    async def run(self, id: int, value: str) -> str:
        fact_id, value = _fact_id(id), _text(value, "value")
        fact = await self.db.aget_fact(fact_id)  # unknown id -> ValueError
        (vector,) = await self.embedder.embed([f"{fact.key}: {value}"])
        await self.db.aupdate_fact(fact_id, value, vector)
        return f"Updated fact {fact_id} ({fact.entity} / {fact.key})."


class MemoryForgetTool(_MemoryTool):
    name = "memory_forget"
    description = (
        "Permanently delete one remembered fact. Use it only when the user asks to forget "
        "something or a fact is wrong and has no replacement: find its id with "
        "memory_search first."
    )
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "Fact id from memory_search."}},
        "required": ["id"],
        "additionalProperties": False,
    }

    async def run(self, id: int) -> str:
        fact_id = _fact_id(id)
        if not await self.db.aforget(fact_id):
            raise ValueError(f"Unknown fact id {fact_id}")
        return f"Forgot fact {fact_id}."


def from_config(params: dict, deps: object) -> list[ITool]:
    """Build the four memory tools; `db` is consumed by `app._stores_from_config`."""
    unknown = set(params) - {"db"}
    if unknown:
        raise ValueError(f"Unknown memory tool param(s): {', '.join(sorted(unknown))}")
    stores = deps.stores
    db = getattr(stores, "memory_db", None) if stores is not None else None
    if db is None:
        raise ValueError("memory tool requires deps.stores.memory_db (none configured)")
    embedder = deps.embedder
    if embedder is None:
        raise ValueError("memory tool requires an embedder (llm.embed_model)")
    return [
        cls(db, embedder)
        for cls in (MemorySearchTool, MemorySaveTool, MemoryUpdateTool, MemoryForgetTool)
    ]
