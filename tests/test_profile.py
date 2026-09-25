"""Unit tests for `ProfileStore` (wiki §9.2, X2 #25)."""

from __future__ import annotations

import os

import pytest
import yaml

from storage.profile import ProfileStore

# -- in-memory (path=None) ---------------------------------------------------


def test_load_missing_gives_empty_profile():
    store = ProfileStore(None)
    assert store.load() == {"preferences": {}, "instructions": ""}


def test_set_and_unset_preference_in_memory():
    store = ProfileStore(None)
    store.set_preference("lang", "fr")
    assert store.load()["preferences"] == {"lang": "fr"}
    store.set_preference("tone", "formal")
    assert store.load()["preferences"] == {"lang": "fr", "tone": "formal"}
    store.unset_preference("lang")
    assert store.load()["preferences"] == {"tone": "formal"}


def test_unset_missing_preference_is_a_noop():
    store = ProfileStore(None)
    store.unset_preference("nope")  # doesn't raise
    assert store.load()["preferences"] == {}


def test_set_instructions_in_memory():
    store = ProfileStore(None)
    store.set_instructions("Always answer in French.")
    assert store.load()["instructions"] == "Always answer in French."


def test_clear_in_memory():
    store = ProfileStore(None)
    store.set_preference("lang", "fr")
    store.set_instructions("hi")
    store.clear()
    assert store.load() == {"preferences": {}, "instructions": ""}


# -- on disk (path=tmp_path/profile.yaml) ------------------------------------


def test_load_missing_file_gives_empty_profile(tmp_path):
    store = ProfileStore(tmp_path / "profile.yaml")
    assert store.load() == {"preferences": {}, "instructions": ""}
    assert not (tmp_path / "profile.yaml").exists()


def test_round_trip_through_yaml(tmp_path):
    path = tmp_path / "profile.yaml"
    store = ProfileStore(path)
    store.set_preference("lang", "fr")
    store.set_instructions("Be concise.")

    assert path.exists()
    on_disk = yaml.safe_load(path.read_text())
    assert on_disk == {"preferences": {"lang": "fr"}, "instructions": "Be concise."}


def test_reload_from_disk_with_a_new_instance(tmp_path):
    path = tmp_path / "profile.yaml"
    ProfileStore(path).set_preference("lang", "fr")

    reopened = ProfileStore(path)
    assert reopened.load()["preferences"] == {"lang": "fr"}


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "profile.yaml"
    ProfileStore(path).set_instructions("hi")
    assert path.exists()


def test_failed_write_leaves_file_and_state_intact(tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    store = ProfileStore(path)
    store.set_preference("lang", "fr")
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.set_preference("tone", "formal")

    assert path.read_bytes() == before
    assert store.load()["preferences"] == {"lang": "fr"}

    leftovers = [p for p in tmp_path.iterdir() if p.name != "profile.yaml"]
    assert leftovers == []
