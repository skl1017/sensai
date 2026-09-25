"""Static user profile, reinjected into context every turn (wiki §9.2, X2 #25).

The profile is a small YAML document, `{preferences: {str: str},
instructions: str}`, edited only through explicit `/profile` CLI commands:
no node or tool writes to it on its own. `ProfileSection` (in
`nodes/context_builder.py`) reads it once per turn via `load()`.

`ProfileStore(path=None)` keeps the profile in memory only. Given a `path`,
`load()` always reads the file fresh (so concurrent readers see the latest
saved state) and writes go through the same atomic temp-file-then-replace
pattern as `ConversationStore`, so a failed write never corrupts the file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from storage._atomic import atomic_write


def _empty_profile() -> dict:
    return {"preferences": {}, "instructions": ""}


class ProfileStore:
    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path is not None else None
        self._memory: dict | None = None  # backing store when `path` is None

    def load(self) -> dict:
        """Current profile; an empty one if nothing was ever saved."""
        if self._path is None:
            return self._memory if self._memory is not None else _empty_profile()
        if not self._path.exists():
            return _empty_profile()
        data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        return {
            "preferences": dict(data.get("preferences") or {}),
            "instructions": data.get("instructions") or "",
        }

    def _save(self, profile: dict) -> None:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self._path, yaml.safe_dump(profile, sort_keys=False))
        else:
            self._memory = profile

    def set_preference(self, key: str, value: str) -> None:
        profile = self.load()
        profile["preferences"] = {**profile["preferences"], key: value}
        self._save(profile)

    def unset_preference(self, key: str) -> None:
        profile = self.load()
        preferences = dict(profile["preferences"])
        preferences.pop(key, None)
        profile["preferences"] = preferences
        self._save(profile)

    def set_instructions(self, text: str) -> None:
        profile = self.load()
        profile["instructions"] = text
        self._save(profile)

    def clear(self) -> None:
        self._save(_empty_profile())
