"""`PermissionPolicy` and the file tools built on top of it (T4, wiki §8.2).

`ReadFileTool`, `ListDirTool` and `WriteFileTool` (#40) all resolve the path
they were given and check it against a `PermissionPolicy` *before* touching
the disk — never trust an as-typed path. `PermissionPolicy.check` does the
resolution itself (`Path.resolve()` follows symlinks and collapses `..`), so
a rule is always evaluated against the real, final filesystem location a
request would actually reach, not the string the model wrote. That's what
defeats the classic `../../etc/passwd`-style traversal and a symlink planted
inside an allowed root pointing outside it.

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

import asyncio
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.tool import ITool

_MODES = frozenset({"read", "write"})
DEFAULT_PERMISSIONS_PATH = "config/permissions.yaml"


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


def _shared_policy(deps: object, permissions_path: str) -> PermissionPolicy:
    """Lazily load and cache one `PermissionPolicy` on `deps.permissions`.

    All three file tools call this from `from_config`, so within a single
    `build_agent` call they end up sharing the exact same instance (loaded
    once, from whichever tool's config is assembled first) instead of each
    re-parsing `permissions.yaml` on its own.
    """
    policy = getattr(deps, "permissions", None)
    if policy is None:
        policy = PermissionPolicy.from_config(permissions_path)
        deps.permissions = policy
    return policy


MAX_READ_BYTES = 200 * 1024  # 200 KB (#40)


def _read_file_sync(path: Path) -> str:
    """Read at most `MAX_READ_BYTES` (+1, to detect truncation) without loading
    an arbitrarily large file fully into memory first."""
    with path.open("rb") as f:
        data = f.read(MAX_READ_BYTES + 1)
    truncated = len(data) > MAX_READ_BYTES
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n…[truncated, file larger than {MAX_READ_BYTES // 1024} KB]"
    return text


class ReadFileTool(ITool):
    """Read a text file's content, truncated past `MAX_READ_BYTES` (#40)."""

    name = "read_file"
    description = (
        "Read a text file and return its content. Files larger than 200 KB are "
        "truncated, with a note saying so. Fails if the path is outside the "
        "allowed roots or the file doesn't exist."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file to read."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy

    async def run(self, path: str) -> str:
        # Permission check + disk I/O are both synchronous, so the whole
        # thing runs in a worker thread and never blocks the event loop.
        return await asyncio.to_thread(self._read_sync, path)

    def _read_sync(self, path: str) -> str:
        resolved = self.policy.check(path, "read")
        return _read_file_sync(resolved)

    @classmethod
    def from_config(cls, params: dict, deps: object) -> ReadFileTool:
        path = params.get("permissions_path", DEFAULT_PERMISSIONS_PATH)
        return cls(policy=_shared_policy(deps, path))


class ListDirTool(ITool):
    """List a directory's entries, filtering out any the policy denies (#40)."""

    name = "list_dir"
    description = (
        "List the entries of a directory (names only, not recursive). Entries "
        "the permission policy denies are silently left out rather than causing "
        "an error."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the directory to list."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy

    async def run(self, path: str) -> str:
        return await asyncio.to_thread(self._list_sync, path)

    def _list_sync(self, path: str) -> str:
        resolved = self.policy.check(path, "read")
        if not resolved.is_dir():
            raise NotADirectoryError(f"{path!r} is not a directory.")

        entries = []
        for entry in sorted(resolved.iterdir()):
            try:
                self.policy.check(entry, "read")
            except PermissionDenied:
                continue
            entries.append(entry.name + ("/" if entry.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty)"

    @classmethod
    def from_config(cls, params: dict, deps: object) -> ListDirTool:
        path = params.get("permissions_path", DEFAULT_PERMISSIONS_PATH)
        return cls(policy=_shared_policy(deps, path))


_WRITE_MODES = frozenset({"create", "overwrite"})


class WriteFileTool(ITool):
    """Create or overwrite a text file; no delete in this version (#40).

    Side-effecting, so `agent.yaml`'s `approval` section wraps it in
    `ApprovedTool` (`app.py::needs_approval`) — this class itself has no
    notion of approval, it only enforces the create/overwrite contract.
    """

    name = "write_file"
    description = (
        "Write text to a file. mode='create' (default) fails if the file already "
        "exists; mode='overwrite' replaces its content. There is no way to delete "
        "a file with this tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "The text to write."},
            "mode": {
                "type": "string",
                "enum": sorted(_WRITE_MODES),
                "description": (
                    "'create' (default) fails if the file already exists; "
                    "'overwrite' replaces its content."
                ),
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy

    async def run(self, path: str, content: str, mode: str = "create") -> str:
        return await asyncio.to_thread(self._write_sync, path, content, mode)

    def _write_sync(self, path: str, content: str, mode: str) -> str:
        if mode not in _WRITE_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(_WRITE_MODES)}")

        resolved = self.policy.check(path, "write")
        if mode == "create" and resolved.exists():
            raise FileExistsError(f"{path!r} already exists; use mode='overwrite' to replace it.")

        resolved.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path!r} (mode={mode!r})."

    @classmethod
    def from_config(cls, params: dict, deps: object) -> WriteFileTool:
        path = params.get("permissions_path", DEFAULT_PERMISSIONS_PATH)
        return cls(policy=_shared_policy(deps, path))
