"""`PersonaSection`: renders the active persona's system prompt (wiki §9.2, A5 #29).

A persona is a small YAML document (`config/personas/<name>.yaml`) with
`name`, `role`, `tone`, `scope` and `prompt_template` (the base name of a
Markdown template resolved through `PromptRegistry`, e.g. `tutor` ->
`config/prompts/tutor_v2.md`). `PersonaSection.build` reads both the persona
file and the prompt text fresh on every turn (`asyncio.to_thread`, same
convention as `ProfileSection`/`ProfileStore.load`), so switching
`ctx.state["persona"]` between turns — or editing a persona/prompt file on
disk — takes effect on the very next request without restarting the agent.

The rendered prompt is recorded in `ctx.state["prompt"] = {"persona",
"template", "version"}` and logged, so `LoggingNode` (EV3) can trace which
prompt version answered a given turn, and `CacheNode` (A6) can fold it into
its cache key to invalidate across persona/prompt changes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import string
from pathlib import Path

import yaml

from core.types import Context
from nodes.context_section import ContextSection
from storage.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PERSONA_FIELDS = ("name", "role", "tone", "scope")


def _validate_persona_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid persona name {name!r}: must match {_NAME_RE.pattern}")


def _load_persona(personas_dir: Path, name: str) -> dict:
    _validate_persona_name(name)
    path = personas_dir / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown persona {name!r}: no such file {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "prompt_template" not in data:
        raise ValueError(f"Persona {name!r} ({path}) is missing a 'prompt_template' key")
    return data


class PersonaSection(ContextSection):
    """Renders `persona_fields` into the persona's prompt template.

    `personas_dir`/`registry.prompts_dir` are resolved by the caller
    (`from_config`/`from_deps`), never guessed here. `default_persona` is
    used when `ctx.state` carries no `persona` key (or an empty one);
    `prompt_version` pins a version for every persona, `None` meaning
    "latest" (resolved independently per turn, so a newly published version
    is picked up without a restart). The default `priority` (100) puts the
    persona *last* in the system prompt, right before the conversation:
    small models follow the most recent instructions best, so the persona's
    scope rules must not be buried under the profile or few-shot examples.
    """

    def __init__(
        self,
        personas_dir: str | Path,
        registry: PromptRegistry,
        default_persona: str = "default",
        prompt_version: str | None = None,
        priority: int = 100,
    ) -> None:
        self.personas_dir = Path(personas_dir)
        self.registry = registry
        self.default_persona = default_persona
        self.prompt_version = prompt_version
        self.priority = priority

    async def build(self, ctx: Context) -> str | None:
        name = ctx.state.get("persona") or self.default_persona

        persona = await asyncio.to_thread(_load_persona, self.personas_dir, name)
        template = persona["prompt_template"]
        text, version = await asyncio.to_thread(self.registry.get, template, self.prompt_version)

        fields = {field: persona.get(field, "") for field in _PERSONA_FIELDS}
        rendered = string.Template(text).safe_substitute(fields)

        ctx.state["prompt"] = {"persona": name, "template": template, "version": version}
        logger.info("Persona %r rendered with prompt %s@%s", name, template, version)

        return rendered.strip()

    @classmethod
    def from_deps(cls, deps) -> PersonaSection:
        """Build from `Deps`: `deps.config_dir / "personas"` / `"prompts"`.

        Fails fast (`ValueError`) if `deps.persona` (the default persona)
        or its prompt template/version don't exist, so a misconfigured
        `agent.yaml` is caught at agent-build time rather than on the first
        chat turn. `deps` is untyped here so this module never imports
        `app.py` (would break the `nodes/` -> `core/` dependency rule).
        """
        personas_dir = deps.config_dir / "personas"
        prompts_dir = deps.config_dir / "prompts"
        registry = PromptRegistry(prompts_dir)

        persona = _load_persona(personas_dir, deps.persona)
        registry.get(persona["prompt_template"], deps.prompt_version)

        return cls(
            personas_dir,
            registry,
            default_persona=deps.persona,
            prompt_version=deps.prompt_version,
        )
