"""Unit tests for `PromptRegistry` (wiki §9.2, A4 #30)."""

from __future__ import annotations

import pytest

from storage.prompt_registry import PromptRegistry

# -- version discovery / selection -------------------------------------------


def _write(tmp_path, name: str, text: str = "content") -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_get_with_no_version_returns_latest(tmp_path):
    _write(tmp_path, "tutor_v1.md", "one")
    _write(tmp_path, "tutor_v2.md", "two")
    registry = PromptRegistry(tmp_path)

    text, version = registry.get("tutor")

    assert (text, version) == ("two", "v2")


def test_latest_version_sorts_numerically_not_lexically(tmp_path):
    _write(tmp_path, "tutor_v2.md", "two")
    _write(tmp_path, "tutor_v10.md", "ten")
    _write(tmp_path, "tutor_v9.md", "nine")
    registry = PromptRegistry(tmp_path)

    text, version = registry.get("tutor")

    assert (text, version) == ("ten", "v10")


def test_get_with_explicit_version(tmp_path):
    _write(tmp_path, "tutor_v1.md", "one")
    _write(tmp_path, "tutor_v2.md", "two")
    registry = PromptRegistry(tmp_path)

    text, version = registry.get("tutor", "v1")

    assert (text, version) == ("one", "v1")


def test_versions_returns_sorted_list(tmp_path):
    _write(tmp_path, "tutor_v2.md", "two")
    _write(tmp_path, "tutor_v10.md", "ten")
    _write(tmp_path, "tutor_v1.md", "one")
    registry = PromptRegistry(tmp_path)

    assert registry.versions("tutor") == ["v1", "v2", "v10"]


def test_versions_of_unknown_template_is_empty(tmp_path):
    registry = PromptRegistry(tmp_path)
    assert registry.versions("nope") == []


# -- missing template / version -> ValueError --------------------------------


def test_get_unknown_template_raises_value_error(tmp_path):
    registry = PromptRegistry(tmp_path)

    with pytest.raises(ValueError, match="nope"):
        registry.get("nope")


def test_get_unknown_version_raises_value_error_listing_available(tmp_path):
    _write(tmp_path, "tutor_v1.md", "one")
    _write(tmp_path, "tutor_v2.md", "two")
    registry = PromptRegistry(tmp_path)

    with pytest.raises(ValueError, match="v1, v2"):
        registry.get("tutor", "v99")


def test_get_missing_dir_raises_value_error(tmp_path):
    registry = PromptRegistry(tmp_path / "does_not_exist")

    with pytest.raises(ValueError):
        registry.get("tutor")


# -- path traversal guard -----------------------------------------------------


@pytest.mark.parametrize("bad_name", ["../etc/passwd", "a/b", "a\\b", ""])
def test_get_rejects_invalid_template_name(tmp_path, bad_name):
    registry = PromptRegistry(tmp_path)

    with pytest.raises(ValueError):
        registry.get(bad_name)


def test_get_rejects_invalid_version(tmp_path):
    _write(tmp_path, "tutor_v1.md", "one")
    registry = PromptRegistry(tmp_path)

    with pytest.raises(ValueError):
        registry.get("tutor", "../v1")


# -- against the real config/ dir --------------------------------------------


def test_real_config_prompts_default_and_tutor():
    registry = PromptRegistry("config/prompts")

    default_text, default_version = registry.get("default")
    tutor_text, tutor_version = registry.get("tutor")

    assert default_version == "v2"
    assert tutor_version == "v2"
    assert "$name" in default_text
    assert "$name" in tutor_text
