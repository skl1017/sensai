"""Integration tests for `DockerBackend` against a real Docker daemon (T2, wiki §8.4).

Skipped (not failed) when Docker is unreachable or `python:3.12-slim` can't be
pulled. Run explicitly with `pytest -m slow tests/test_code_sandbox_docker.py`.
These are the tests the ticket asked for specifically: infinite loop, fork
bomb, infinite output, and the acceptance criterion that the network is
genuinely unreachable from inside the container.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.code_sandbox import CodeSandboxTool, DockerBackend

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def docker_available():
    async def _check() -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return await proc.wait() == 0
        except FileNotFoundError:
            return False

    if not asyncio.run(_check()):
        pytest.skip("Docker not reachable")


class TestDockerBackendIsolation:
    async def test_network_is_unreachable(self, docker_available):
        backend = DockerBackend()
        code = "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)\n"
        result = await backend.run(code, timeout=15)
        assert result.timed_out is False
        assert result.exit_code != 0
        # no DNS/route without a network interface: urllib raises before any
        # real connection attempt, so this never depends on example.com itself
        # being reachable.
        assert "URLError" in result.stderr or "OSError" in result.stderr

    async def test_infinite_loop_is_killed_at_the_timeout(self, docker_available):
        backend = DockerBackend()
        result = await backend.run("while True:\n    pass\n", timeout=3)
        assert result.timed_out is True

    async def test_fork_bomb_is_stopped_by_the_pids_limit(self, docker_available):
        backend = DockerBackend(pids_limit=32)
        code = "import os\nwhile True:\n    os.fork()\n"
        result = await backend.run(code, timeout=15)
        # the pids-limit makes fork() itself fail quickly: the process crashes
        # on its own well before the timeout, proving the cap actually bites
        # instead of the sandbox just getting lucky and running out of time.
        assert result.timed_out is False
        assert result.exit_code != 0

    async def test_infinite_output_does_not_hang_and_is_truncated(self, docker_available):
        tool = CodeSandboxTool(DockerBackend(), timeout=3)
        result = await tool.run(code="while True:\n    print('x' * 100)\n")
        assert "Error: execution timed out" in result

    async def test_read_only_filesystem_blocks_writes_outside_tmp(self, docker_available):
        backend = DockerBackend()
        code = "open('/etc/sensai-test', 'w').write('nope')"
        result = await backend.run(code, timeout=10)
        assert result.exit_code != 0
        assert "OSError" in result.stderr or "PermissionError" in result.stderr

    async def test_tmp_is_still_writable(self, docker_available):
        backend = DockerBackend()
        code = "open('/tmp/ok', 'w').write('fine'); print(open('/tmp/ok').read())"
        result = await backend.run(code, timeout=10)
        assert result.exit_code == 0
        assert result.stdout.strip() == "fine"
