"""Tests for `PermissionPolicy`/`Rule` (T4, wiki §8.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.file_tool import PermissionDenied, PermissionPolicy, Rule


def _policy(tmp_path, *, deny_subdir=False):
    root = tmp_path / "sandbox"
    root.mkdir()
    rules = [Rule(root=root, modes={"read", "write"})]
    if deny_subdir:
        secret = root / "secret"
        secret.mkdir()
        rules.append(Rule(root=secret, modes={"read", "write"}, deny=True))
    return PermissionPolicy(rules), root


class TestRuleValidation:
    def test_empty_modes_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Rule(root="x", modes=[])

    def test_unsupported_mode_rejected(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            Rule(root="x", modes=["execute"])

    def test_modes_normalized_to_frozenset(self):
        rule = Rule(root="x", modes=["read", "read", "write"])
        assert rule.modes == frozenset({"read", "write"})


class TestAllowedAccess:
    def test_file_inside_root_is_allowed(self, tmp_path):
        policy, root = _policy(tmp_path)
        target = root / "notes.txt"
        target.write_text("hi")
        assert policy.check(target, "read") == target.resolve()

    def test_nested_path_inside_root_is_allowed(self, tmp_path):
        policy, root = _policy(tmp_path)
        nested = root / "a" / "b" / "c.txt"
        assert policy.check(nested, "write") == nested.resolve()

    def test_root_itself_is_allowed(self, tmp_path):
        policy, root = _policy(tmp_path)
        assert policy.check(root, "read") == root.resolve()


class TestModeIsolation:
    def test_write_denied_when_rule_only_allows_read(self, tmp_path):
        root = tmp_path / "sandbox"
        root.mkdir()
        policy = PermissionPolicy([Rule(root=root, modes={"read"})])
        target = root / "f.txt"
        policy.check(target, "read")
        with pytest.raises(PermissionDenied):
            policy.check(target, "write")

    def test_unsupported_check_mode_raises_value_error(self, tmp_path):
        policy, root = _policy(tmp_path)
        with pytest.raises(ValueError):
            policy.check(root, "execute")


class TestDenyByDefault:
    def test_path_outside_every_root_is_denied(self, tmp_path):
        policy, root = _policy(tmp_path)
        outside = tmp_path / "outside.txt"
        with pytest.raises(PermissionDenied):
            policy.check(outside, "read")

    def test_denial_message_names_allowed_root(self, tmp_path):
        policy, root = _policy(tmp_path)
        outside = tmp_path / "outside.txt"
        with pytest.raises(PermissionDenied, match=str(root.resolve())):
            policy.check(outside, "read")

    def test_no_rules_at_all_denies_everything(self, tmp_path):
        policy = PermissionPolicy([])
        with pytest.raises(PermissionDenied, match="none configured"):
            policy.check(tmp_path / "x", "read")


class TestPathTraversal:
    @pytest.mark.parametrize(
        "traversal",
        [
            "../../etc/passwd",
            "../../../etc/passwd",
            "a/../../etc/passwd",
            "..",
        ],
    )
    def test_traversal_out_of_root_is_denied(self, tmp_path, traversal):
        policy, root = _policy(tmp_path)
        with pytest.raises(PermissionDenied):
            policy.check(root / traversal, "read")

    def test_traversal_that_stays_inside_root_is_allowed(self, tmp_path):
        policy, root = _policy(tmp_path)
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        assert policy.check(nested / ".." / "b", "read") == nested.resolve()


class TestSymlinkEscape:
    def test_symlink_pointing_outside_root_is_denied(self, tmp_path):
        policy, root = _policy(tmp_path)
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        link = root / "link.txt"
        link.symlink_to(secret)
        with pytest.raises(PermissionDenied):
            policy.check(link, "read")

    def test_symlinked_directory_pointing_outside_root_is_denied(self, tmp_path):
        policy, root = _policy(tmp_path)
        secret_dir = tmp_path / "secret_dir"
        secret_dir.mkdir()
        (secret_dir / "f.txt").write_text("nope")
        link_dir = root / "link_dir"
        link_dir.symlink_to(secret_dir)
        with pytest.raises(PermissionDenied):
            policy.check(link_dir / "f.txt", "read")

    def test_symlink_staying_inside_root_is_allowed(self, tmp_path):
        policy, root = _policy(tmp_path)
        real = root / "real.txt"
        real.write_text("ok")
        link = root / "link.txt"
        link.symlink_to(real)
        assert policy.check(link, "read") == real.resolve()


class TestDenyWins:
    def test_deny_rule_overrides_broader_allow(self, tmp_path):
        policy, root = _policy(tmp_path, deny_subdir=True)
        target = root / "secret" / "f.txt"
        with pytest.raises(PermissionDenied):
            policy.check(target, "read")

    def test_allow_still_works_outside_the_denied_subdir(self, tmp_path):
        policy, root = _policy(tmp_path, deny_subdir=True)
        target = root / "public.txt"
        assert policy.check(target, "read") == target.resolve()

    def test_deny_wins_regardless_of_declaration_order(self, tmp_path):
        root = tmp_path / "sandbox"
        secret = root / "secret"
        secret.mkdir(parents=True)
        # deny rule declared *before* the broader allow this time
        policy = PermissionPolicy(
            [
                Rule(root=secret, modes={"read"}, deny=True),
                Rule(root=root, modes={"read"}),
            ]
        )
        with pytest.raises(PermissionDenied):
            policy.check(secret / "f.txt", "read")


class TestFromConfig:
    def test_loads_rules_from_yaml(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        cfg = tmp_path / "permissions.yaml"
        cfg.write_text(f"rules:\n  - root: {root}\n    modes: [read, write]\n", encoding="utf-8")
        policy = PermissionPolicy.from_config(cfg)
        target = root / "f.txt"
        assert policy.check(target, "write") == target.resolve()

    def test_deny_rule_from_yaml(self, tmp_path):
        root = tmp_path / "workspace"
        secret = root / "secret"
        secret.mkdir(parents=True)
        cfg = tmp_path / "permissions.yaml"
        cfg.write_text(
            f"rules:\n"
            f"  - root: {root}\n    modes: [read]\n"
            f"  - root: {secret}\n    modes: [read]\n    deny: true\n",
            encoding="utf-8",
        )
        policy = PermissionPolicy.from_config(cfg)
        with pytest.raises(PermissionDenied):
            policy.check(secret / "f.txt", "read")

    def test_empty_rules_denies_everything(self, tmp_path):
        cfg = tmp_path / "permissions.yaml"
        cfg.write_text("rules: []\n", encoding="utf-8")
        policy = PermissionPolicy.from_config(cfg)
        with pytest.raises(PermissionDenied):
            policy.check(tmp_path / "x", "read")

    def test_missing_root_key_raises(self, tmp_path):
        cfg = tmp_path / "permissions.yaml"
        cfg.write_text("rules:\n  - modes: [read]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="root"):
            PermissionPolicy.from_config(cfg)

    def test_missing_modes_key_raises(self, tmp_path):
        cfg = tmp_path / "permissions.yaml"
        cfg.write_text(f"rules:\n  - root: {tmp_path}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="modes"):
            PermissionPolicy.from_config(cfg)

    def test_loads_the_shipped_default_config(self):
        policy = PermissionPolicy.from_config("config/permissions.yaml")
        allowed = policy.check("data/whatever.yaml", "read")
        assert allowed == Path("data/whatever.yaml").resolve()
        with pytest.raises(PermissionDenied):
            policy.check("/etc/passwd", "read")
