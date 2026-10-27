import sys

import pytest


def pytest_collection_finish(session):
    print("\nMODULES models:", sorted(m for m in sys.modules if m.endswith("models")))


@pytest.fixture
def no_mcp(monkeypatch):
    """The real config declares an `npx` MCP server: never spawn it in unit tests."""

    async def _no_servers(configs, stack, **_):
        return []

    monkeypatch.setattr("app.connect_servers", _no_servers)
