"""`PromptRegistry`: versioned Markdown prompt templates on disk (wiki §9.2, A4 #30).

Prompt templates live as flat files, `<prompts_dir>/<template>_<version>.md`
(e.g. `config/prompts/tutor_v2.md`), so a new version is a new file and old
ones stay around for rollback/comparison. `PersonaSection` (#29) resolves a
persona's `prompt_template` name through `get()` and renders it with
`string.Template.safe_substitute`.

Versions are the literal `vN` suffix (`v1`, `v2`, ..., `v10`), compared
numerically so `v10` sorts after `v2`. `get(template, version=None)` always
reads the file fresh off disk (synchronous I/O: callers wrap it in
`asyncio.to_thread`, same convention as `ProfileStore.load`), so a prompt
edited between turns is picked up on the very next request.

`template` and `version` are validated against a strict pattern before
touching the filesystem: no path separators, no `..`, so a value that
ultimately comes from config or persona YAML can never escape `prompts_dir`.
"""

from __future__ import annotations

import re
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VERSION_RE = re.compile(r"^v(\d+)$")


def _validate_name(value: str, kind: str) -> None:
    if not _NAME_RE.match(value):
        raise ValueError(f"Invalid {kind} {value!r}: must match {_NAME_RE.pattern}")


class PromptRegistry:
    def __init__(self, prompts_dir: str | Path) -> None:
        self.prompts_dir = Path(prompts_dir)

    def versions(self, template: str) -> list[str]:
        """Available versions of `template`, sorted oldest to newest (numeric on `vN`)."""
        _validate_name(template, "template name")

        found: list[tuple[int, str]] = []
        if self.prompts_dir.is_dir():
            prefix = f"{template}_"
            for path in self.prompts_dir.glob(f"{prefix}v*.md"):
                stem = path.stem  # "<template>_v<N>"
                version = stem[len(prefix) :]
                match = _VERSION_RE.match(version)
                if match:
                    found.append((int(match.group(1)), version))

        found.sort(key=lambda item: item[0])
        return [version for _, version in found]

    def get(self, template: str, version: str | None = None) -> tuple[str, str]:
        """`(text, version)` for `template`; `version=None` means the latest `vN`.

        Raises `ValueError` if the template has no versions at all, or if the
        requested `version` doesn't exist — both list the versions that are
        actually available.
        """
        _validate_name(template, "template name")
        available = self.versions(template)

        if version is None:
            if not available:
                raise ValueError(
                    f"No versions found for prompt template {template!r} in {self.prompts_dir}"
                )
            resolved = available[-1]
        else:
            _validate_name(version, "prompt version")
            if version not in available:
                raise ValueError(
                    f"Unknown version {version!r} for prompt template {template!r}. "
                    f"Available: {', '.join(available) if available else '(none)'}"
                )
            resolved = version

        path = self.prompts_dir / f"{template}_{resolved}.md"
        text = path.read_text(encoding="utf-8")
        return text, resolved
