"""Integration tests for `web_search` against a real SearXNG instance (#126).

Skipped (not failed) when no instance answers at `SEARXNG_URL` (default
`http://localhost:8080`, the `searxng` service of docker-compose.yml). Run
with `docker compose up -d searxng`, then `pytest -m slow
tests/test_web_search_integration.py`.

An instance that answers but refuses the JSON format (HTTP 403, SearXNG's
default) is a failure, not a skip: that's exactly what
`config/searxng/settings.yml` must prevent.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tools.web_search import SearXNGProvider, WebSearchTool, build_web_tools

pytestmark = pytest.mark.slow

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")
QUERY = "python asyncio"


@pytest.fixture(scope="module")
def searxng_url() -> str:
    try:
        httpx.get(f"{SEARXNG_URL}/healthz", timeout=5)
    except httpx.HTTPError:
        pytest.skip(f"SearXNG not reachable at {SEARXNG_URL}")
    return SEARXNG_URL


def test_json_format_is_enabled(searxng_url):
    response = httpx.get(f"{searxng_url}/search", params={"q": QUERY, "format": "json"}, timeout=20)

    assert response.status_code == 200, "JSON format disabled? see config/searxng/settings.yml"
    assert "results" in response.json()


async def test_provider_returns_real_results(searxng_url):
    results = await SearXNGProvider(searxng_url, timeout=20).search(QUERY, 3)

    assert 0 < len(results) <= 3
    assert all(r.url.startswith("http") and r.title for r in results)


async def test_web_search_tool_end_to_end(searxng_url, monkeypatch):
    """Built like `app.py` does from `agent.yaml` (no SerpApi key: SearXNG)."""
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    tool = next(
        t
        for t in build_web_tools({"url": searxng_url, "timeout": 20}, None)
        if isinstance(t, WebSearchTool)
    )

    output = await tool.run(query=QUERY, max_results=2)

    assert output.count("<web_result url=") == 2
