"""Unit tests for `PersonaSection` (wiki §9.2, A5 #29)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.types import Context, Message
from nodes.persona_section import PersonaSection
from storage.prompt_registry import PromptRegistry
from tests.fakes import FakeLLM


def make_ctx(state: dict | None = None) -> Context:
    llm = FakeLLM(["unused"])
    return Context("hi", [Message("user", "hi")], llm, {}, state=state or {})


def _write_persona(personas_dir: Path, filename: str, **fields) -> None:
    personas_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{key}: {value}" for key, value in fields.items())
    (personas_dir / f"{filename}.yaml").write_text(body + "\n", encoding="utf-8")


def _write_prompt(prompts_dir: Path, template: str, version: str, text: str) -> None:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / f"{template}_{version}.md").write_text(text, encoding="utf-8")


@pytest.fixture
def personas_and_prompts(tmp_path):
    personas_dir = tmp_path / "personas"
    prompts_dir = tmp_path / "prompts"

    _write_persona(
        personas_dir,
        "default",
        name="Sensai",
        role="assistant",
        tone="neutral",
        scope="none",
        prompt_template="default",
    )
    _write_prompt(prompts_dir, "default", "v1", "You are $name, $role. Tone: $tone.")
    _write_prompt(prompts_dir, "default", "v2", "v2: You are $name, $role. Scope: $scope.")

    _write_persona(
        personas_dir,
        "tutor",
        name="Lea",
        role="math tutor",
        tone="kind",
        scope="math only",
        prompt_template="tutor",
    )
    _write_prompt(prompts_dir, "tutor", "v1", "Tutor $name: $role, $tone, $scope.")

    return personas_dir, prompts_dir


# -- rendering ----------------------------------------------------------------


async def test_build_renders_latest_version_by_default(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    ctx = make_ctx()

    result = await section.build(ctx)

    assert result == "v2: You are Sensai, assistant. Scope: none."


async def test_build_respects_explicit_prompt_version(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir), prompt_version="v1")
    ctx = make_ctx()

    result = await section.build(ctx)

    assert result == "You are Sensai, assistant. Tone: neutral."


async def test_build_records_prompt_state(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    ctx = make_ctx()

    await section.build(ctx)

    assert ctx.state["prompt"] == {"persona": "default", "template": "default", "version": "v2"}


async def test_default_persona_used_when_state_has_none(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir), default_persona="default")
    ctx = make_ctx(state={})

    result = await section.build(ctx)

    assert "Sensai" in result


async def test_ctx_state_persona_overrides_default(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir), default_persona="default")
    ctx = make_ctx(state={"persona": "tutor"})

    result = await section.build(ctx)

    assert result == "Tutor Lea: math tutor, kind, math only."
    assert ctx.state["prompt"]["persona"] == "tutor"


async def test_switching_persona_between_turns_changes_prompt_without_touching_history(
    personas_and_prompts,
):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    history = [Message("user", "old"), Message("assistant", "old reply")]

    ctx1 = Context("hi", list(history), FakeLLM(["unused"]), {}, state={"persona": "default"})
    result1 = await section.build(ctx1)

    ctx2 = Context("hi", list(history), FakeLLM(["unused"]), {}, state={"persona": "tutor"})
    result2 = await section.build(ctx2)

    assert result1 != result2
    assert "Sensai" in result1
    assert "Lea" in result2
    assert history == [Message("user", "old"), Message("assistant", "old reply")]


# -- errors --------------------------------------------------------------------


async def test_unknown_persona_raises_value_error(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    ctx = make_ctx(state={"persona": "ghost"})

    with pytest.raises(ValueError, match="ghost"):
        await section.build(ctx)


async def test_persona_missing_prompt_template_key_raises(tmp_path):
    personas_dir = tmp_path / "personas"
    prompts_dir = tmp_path / "prompts"
    _write_persona(personas_dir, "broken", name="X", role="Y", tone="Z", scope="W")

    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    ctx = make_ctx(state={"persona": "broken"})

    with pytest.raises(ValueError, match="prompt_template"):
        await section.build(ctx)


@pytest.mark.parametrize("bad_name", ["../etc/passwd", "a/b"])
async def test_persona_name_path_traversal_rejected(personas_and_prompts, bad_name):
    personas_dir, prompts_dir = personas_and_prompts
    section = PersonaSection(personas_dir, PromptRegistry(prompts_dir))
    ctx = make_ctx(state={"persona": bad_name})

    with pytest.raises(ValueError):
        await section.build(ctx)


# -- from_deps ------------------------------------------------------------------


@dataclass
class FakeDeps:
    config_dir: Path
    persona: str = "default"
    prompt_version: str | None = None


async def test_from_deps_builds_working_section(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    config_dir = personas_dir.parent
    deps = FakeDeps(config_dir=config_dir, persona="default")

    section = PersonaSection.from_deps(deps)
    ctx = make_ctx()

    result = await section.build(ctx)

    assert "Sensai" in result


def test_from_deps_fails_fast_on_missing_default_persona(tmp_path):
    (tmp_path / "personas").mkdir()
    (tmp_path / "prompts").mkdir()
    deps = FakeDeps(config_dir=tmp_path, persona="ghost")

    with pytest.raises(ValueError):
        PersonaSection.from_deps(deps)


def test_from_deps_fails_fast_on_missing_prompt_version(personas_and_prompts):
    personas_dir, prompts_dir = personas_and_prompts
    config_dir = personas_dir.parent
    deps = FakeDeps(config_dir=config_dir, persona="default", prompt_version="v99")

    with pytest.raises(ValueError):
        PersonaSection.from_deps(deps)


# -- against the real config/ dir -----------------------------------------------


async def test_real_config_default_persona():
    section = PersonaSection(Path("config/personas"), PromptRegistry(Path("config/prompts")))
    ctx = make_ctx()

    result = await section.build(ctx)

    assert "Sensai" in result
    assert ctx.state["prompt"]["persona"] == "default"


def test_real_config_from_deps():
    deps = FakeDeps(config_dir=Path("config"), persona="default")
    section = PersonaSection.from_deps(deps)
    assert section.default_persona == "default"
