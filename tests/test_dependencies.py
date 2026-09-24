"""core/ depends on nothing but the standard library (and itself).

This encodes the dependency rule from CLAUDE.md: "core/ depends on nothing;
everything depends on core/, never the reverse." It walks every module under
core/ with ast (no imports are actually executed) and checks that each
absolute import targets either a stdlib module or the core package itself.
Relative imports (e.g. `from . import types`) are always allowed since they
can only ever reach inside core/.
"""

import ast
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "core"
STDLIB_MODULES = set(sys.stdlib_module_names)


def _core_py_files():
    return sorted(CORE_DIR.rglob("*.py"))


def _top_level(module_name: str) -> str:
    return module_name.split(".")[0]


def _iter_absolute_imports(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import, e.g. `from . import types` -> always allowed
                continue
            if node.module:
                yield node.module


def test_core_files_exist():
    assert CORE_DIR.is_dir(), "core/ package is missing"
    assert (CORE_DIR / "__init__.py").exists(), "core/__init__.py is missing"


def test_core_only_imports_stdlib_or_itself():
    violations = []

    for path in _core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _iter_absolute_imports(tree):
            top = _top_level(module_name)
            if top == "core":
                continue
            if top in STDLIB_MODULES:
                continue
            violations.append(f"{path.relative_to(CORE_DIR.parent)}: imports '{module_name}'")

    assert not violations, "core/ must only depend on the stdlib or itself:\n" + "\n".join(
        violations
    )
