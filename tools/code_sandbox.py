"""`CodeSandboxTool`: run model-generated Python in an isolated sandbox (T2, wiki §8.4).

`SandboxBackend` is the interchangeable execution backend (same pattern as
`SearchEngine` in `tools/web_search.py`): `DockerBackend` is the real one —
a throwaway, network-less, resource-capped container — and `SubprocessBackend`
is a dev-only fallback with no real isolation at all, used when Docker isn't
available. Both are shelled out to via `asyncio.create_subprocess_exec`
(never `create_subprocess_shell`), so the model-supplied `code` is passed as
a single argv entry, never interpolated into a shell string — no shell
injection surface regardless of what the code contains.

A tool error here is never a crash: a broken snippet's traceback comes back
as `stderr` in the observation, exactly like any other tool failure, so the
model can read it and fix its own code (the point of this tool existing).
`run_python` is side-effecting (arbitrary code execution) so it is listed in
`agent.yaml`'s `approval` section and wrapped by `ApprovedTool` (A3).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.tool import ITool

MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n…[truncated, {omitted} chars omitted]"


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxBackend(ABC):
    """Interchangeable code execution backend (Docker today, subprocess fallback)."""

    @abstractmethod
    async def run(self, code: str, timeout: float) -> RunResult:
        """Run `code` as a Python script, bounded by `timeout` seconds."""


async def _run_and_wait(cmd: list[str], timeout: float, on_timeout=None) -> RunResult:
    """Shared `communicate()`-with-timeout plumbing for both backends.

    `on_timeout`, when given, is awaited *before* killing the local process —
    `DockerBackend` uses it to `docker kill` the container itself, since
    killing the local `docker run` client alone does not guarantee the
    container stops.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        if on_timeout is not None:
            await on_timeout()
        proc.kill()
        await proc.wait()
        return RunResult(exit_code=-1, stdout="", stderr="", timed_out=True)

    return RunResult(
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


class DockerBackend(SandboxBackend):
    """Run code in a throwaway, network-less, resource-capped container.

    - `--rm`: the container is deleted on exit, nothing persists.
    - `--network none`: no network interface at all (the acceptance
      criterion: the sandbox cannot reach the network).
    - `--memory`/`--cpus`: hard resource caps.
    - `--pids-limit`: caps the number of processes/threads the container can
      create — this is what actually stops a fork bomb; `--memory` alone
      would not (a fork bomb exhausts PIDs long before it exhausts RAM).
    - `--user 65534:65534` (`nobody`): never runs as root inside the
      container, by numeric uid/gid so it doesn't depend on the image having
      a `nobody` entry in `/etc/passwd`.
    - `--read-only` + `--tmpfs /tmp`: the root filesystem can't be written
      to; `/tmp` is the only writable path, and it's wiped with the
      container.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        memory: str = "256m",
        cpus: str = "1",
        pids_limit: int = 64,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    async def run(self, code: str, timeout: float) -> RunResult:
        container_name = f"sensai-sandbox-{uuid.uuid4().hex}"
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--user",
            "65534:65534",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "-i",
            self.image,
            "python3",
            "-c",
            code,
        ]

        async def kill_container() -> None:
            killer = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()

        return await _run_and_wait(cmd, timeout, on_timeout=kill_container)


class SubprocessBackend(SandboxBackend):
    """Run code as a plain local subprocess. NOT a real sandbox.

    No network block, no filesystem restriction, no resource cap beyond the
    timeout itself — the code runs with this process's own privileges. Exists
    only so the tool still works in dev without Docker installed; never use
    this where isolation actually matters.
    """

    async def run(self, code: str, timeout: float) -> RunResult:
        cmd = [sys.executable, "-c", code]
        return await _run_and_wait(cmd, timeout)


_BACKENDS = {"docker": DockerBackend, "subprocess": SubprocessBackend}


class CodeSandboxTool(ITool):
    name = "run_python"
    description = (
        "Run a Python snippet in an isolated sandbox (no network access) and "
        "get back its exit code, stdout and stderr. Use this to test code, "
        "compute something precisely, or verify a fix actually works. If it "
        "fails, the traceback comes back in stderr — read it and correct "
        "your code, don't just guess what went wrong."
    )
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The Python code to run."}},
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, backend: SandboxBackend, timeout: float = 10.0) -> None:
        self.backend = backend
        self.timeout = timeout

    async def run(self, code: str) -> str:
        result = await self.backend.run(code, self.timeout)
        if result.timed_out:
            return f"Error: execution timed out after {self.timeout}s."
        return (
            f"exit_code: {result.exit_code}\n"
            f"stdout:\n{_truncate(result.stdout)}\n"
            f"stderr:\n{_truncate(result.stderr)}"
        )

    @classmethod
    def from_config(cls, params: dict, deps: object) -> CodeSandboxTool:
        backend_name = params.get("backend", "docker")
        backend_cls = _BACKENDS.get(backend_name)
        if backend_cls is None:
            raise ValueError(
                f"Unknown sandbox backend {backend_name!r}. Available: {sorted(_BACKENDS)}"
            )

        if backend_cls is DockerBackend:
            backend = DockerBackend(
                image=params.get("image", "python:3.12-slim"),
                memory=params.get("memory", "256m"),
                cpus=params.get("cpus", "1"),
                pids_limit=params.get("pids_limit", 64),
            )
        else:
            backend = SubprocessBackend()

        return cls(backend=backend, timeout=params.get("timeout", 10.0))
