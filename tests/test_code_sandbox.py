"""Unit tests for `CodeSandboxTool` (T2, wiki §8.4): no Docker required.

Uses `SubprocessBackend` (real, local, fast) to exercise the shared
`_run_and_wait` plumbing and `CodeSandboxTool`'s own formatting/truncation —
everything that doesn't depend on Docker's actual isolation guarantees (that
part lives in `tests/test_code_sandbox_docker.py`, marked `slow`).
"""

from __future__ import annotations

import pytest

from core.tool import ITool
from tools.code_sandbox import (
    MAX_OUTPUT_CHARS,
    CodeSandboxTool,
    DockerBackend,
    RunResult,
    SandboxBackend,
    SubprocessBackend,
    _truncate,
)


class TestTruncate:
    def test_short_text_is_untouched(self):
        assert _truncate("hello") == "hello"

    def test_long_text_is_truncated_with_a_note(self):
        text = "x" * (MAX_OUTPUT_CHARS + 500)
        result = _truncate(text)
        assert result.startswith("x" * 10)
        assert "truncated" in result
        assert len(result) < len(text)


class TestSandboxBackendContract:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            SandboxBackend()


class TestSubprocessBackend:
    async def test_successful_code_returns_stdout_and_exit_code_zero(self):
        result = await SubprocessBackend().run("print(1 + 1)", timeout=5)
        assert result.exit_code == 0
        assert result.stdout.strip() == "2"
        assert result.stderr == ""
        assert result.timed_out is False

    async def test_raising_code_returns_nonzero_exit_and_traceback_in_stderr(self):
        result = await SubprocessBackend().run("1 / 0", timeout=5)
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    async def test_slow_code_is_killed_at_the_timeout(self):
        result = await SubprocessBackend().run("import time; time.sleep(5)", timeout=0.2)
        assert result.timed_out is True
        assert result.exit_code == -1


class _FakeBackend(SandboxBackend):
    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.calls: list[tuple[str, float]] = []

    async def run(self, code: str, timeout: float) -> RunResult:
        self.calls.append((code, timeout))
        return self._result


class TestCodeSandboxToolRun:
    async def test_formats_exit_code_stdout_and_stderr(self):
        backend = _FakeBackend(RunResult(exit_code=0, stdout="hi", stderr=""))
        tool = CodeSandboxTool(backend, timeout=5)
        result = await tool.run(code="print('hi')")

        assert "exit_code: 0" in result
        assert "stdout:\nhi" in result
        assert "stderr:\n" in result
        assert backend.calls == [("print('hi')", 5)]

    async def test_timed_out_result_is_reported_without_a_traceback(self):
        backend = _FakeBackend(RunResult(exit_code=-1, stdout="", stderr="", timed_out=True))
        tool = CodeSandboxTool(backend, timeout=3)
        result = await tool.run(code="while True: pass")
        assert result == "Error: execution timed out after 3s."

    async def test_long_stdout_and_stderr_are_truncated(self):
        big = "y" * (MAX_OUTPUT_CHARS + 500)
        backend = _FakeBackend(RunResult(exit_code=0, stdout=big, stderr=big))
        tool = CodeSandboxTool(backend)
        result = await tool.run(code="...")
        assert result.count("truncated") == 2

    async def test_traceback_from_broken_code_comes_back_as_an_observation(self):
        """The acceptance criterion: a broken snippet's traceback is returned as
        a normal string (an observation), never raised — the model can read it
        and fix its own code."""
        tool = CodeSandboxTool(SubprocessBackend(), timeout=5)
        result = await tool.run(code="raise ValueError('bad input')")
        assert "exit_code: 1" in result
        assert "ValueError: bad input" in result

    def test_schema_and_contract(self):
        tool = CodeSandboxTool(SubprocessBackend())
        assert isinstance(tool, ITool)
        assert tool.name == "run_python"
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False
        assert tool.parameters["required"] == ["code"]


class TestFromConfig:
    def test_defaults_to_docker_backend(self):
        tool = CodeSandboxTool.from_config({}, None)
        assert isinstance(tool.backend, DockerBackend)
        assert tool.backend.image == "python:3.12-slim"
        assert tool.backend.memory == "256m"
        assert tool.backend.cpus == "1"
        assert tool.backend.pids_limit == 64
        assert tool.timeout == 10.0

    def test_docker_backend_params_are_forwarded(self):
        params = {
            "backend": "docker",
            "image": "python:3.13-slim",
            "memory": "128m",
            "cpus": "0.5",
            "pids_limit": 32,
            "timeout": 5,
        }
        tool = CodeSandboxTool.from_config(params, None)
        assert tool.backend.image == "python:3.13-slim"
        assert tool.backend.memory == "128m"
        assert tool.backend.cpus == "0.5"
        assert tool.backend.pids_limit == 32
        assert tool.timeout == 5

    def test_subprocess_backend_opt_in(self):
        tool = CodeSandboxTool.from_config({"backend": "subprocess"}, None)
        assert isinstance(tool.backend, SubprocessBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            CodeSandboxTool.from_config({"backend": "qemu"}, None)
