"""`PermissionPolicy`: filesystem access control checked before any disk
operation (T4, wiki §8.2).

The file tools built on top of this module (#40: `read_file`, `list_dir`,
`write_file`) must resolve the path they were given and check it against a
`PermissionPolicy` *before* touching the disk — never trust an as-typed path.
`PermissionPolicy.check` does the resolution itself (`Path.resolve()`
follows symlinks and collapses `..`), so a rule is always evaluated against
the real, final filesystem location a request would actually reach, not the
string the model wrote. That's what defeats the classic
`../../etc/passwd`-style traversal and a symlink planted inside an allowed
root pointing outside it.

Deny-by-default: a path with no matching allow `Rule` is refused, and any
matching `deny=True` rule wins over any matching allow rule, regardless of
which was declared first — there is no rule ordering to reason about.

`Rule(root, modes, deny)` grants (`deny=False`, the default) or forbids
(`deny=True`) a set of `modes` (a subset of `{"read", "write"}`) under
`root`. Every rule's `root` is itself resolved once, at `PermissionPolicy`
construction — so a root that is itself a symlink still resolves to the
same real location a request against it would.

`config/permissions.yaml` format
---------------------------------
```yaml
rules:
  - root: data/
    modes: [read, write]
  - root: data/secrets/
    modes: [read, write]
    deny: true
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_MODES = frozenset({"read", "write"})


class PermissionDenied(Exception):
    """Raised by `PermissionPolicy.check`; the message names the allowed root(s)."""


@dataclass
class Rule:
    root: Path
    modes: frozenset[str]
    deny: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.modes = frozenset(self.modes)
        if not self.modes:
            raise ValueError("Rule.modes must not be empty")
        if not self.modes <= _MODES:
            raise ValueError(f"Unsupported mode(s) in {self.modes}; expected a subset of {_MODES}")


class PermissionPolicy:
    """Deny-by-default filesystem access control (wiki §8.2).

    Built from a list of `Rule`s. `check` is the only thing a tool should
    call before touching the disk.
    """

    def __init__(self, rules: list[Rule]) -> None:
        # Re-wrapped (not mutated in place) so a rule's `root` is resolved
        # exactly like `check` resolves the paths it's asked about, without
        # surprising the caller by mutating the `Rule` objects it passed in.
        self.rules = [Rule(root=r.root.resolve(), modes=r.modes, deny=r.deny) for r in rules]

    def check(self, path: str | Path, mode: str) -> Path:
        """Resolve `path` (symlinks/`..` followed) and check `mode` access.

        Returns the resolved `Path` if some rule allows it. Raises
        `PermissionDenied` — naming the allowed root(s) for `mode` — if no
        rule allows it (deny-by-default), or if any matching rule denies it
        (a deny always wins over a matching allow).
        """
        if mode not in _MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(_MODES)}")

        resolved = Path(path).resolve()
        matching = [r for r in self.rules if mode in r.modes and _is_within(resolved, r.root)]

        if not matching or any(r.deny for r in matching):
            allowed = sorted({str(r.root) for r in self.rules if mode in r.modes and not r.deny})
            roots = ", ".join(allowed) if allowed else "(none configured)"
            raise PermissionDenied(
                f"Access denied for {str(path)!r} (mode={mode!r}). "
                f"Allowed root(s) for {mode!r}: {roots}"
            )

        return resolved

    @classmethod
    def from_config(cls, path: str | Path) -> PermissionPolicy:
        """Load `rules` from a YAML file (see module docstring for the format)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rules = [_parse_rule(entry, i) for i, entry in enumerate(data.get("rules") or [])]
        return cls(rules)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _parse_rule(entry: dict, index: int) -> Rule:
    where = f"rules[{index}]"
    if not isinstance(entry, dict) or "root" not in entry:
        raise ValueError(f"{where}.root: required")
    modes = entry.get("modes")
    if not modes or not isinstance(modes, list):
        raise ValueError(f"{where}.modes: required, expected a non-empty list")
    return Rule(
        root=Path(entry["root"]), modes=frozenset(modes), deny=bool(entry.get("deny", False))
    )
