"""Tests for `WebSearchTool`/`FetchPageTool` and their guardrails (T3, wiki §8.3)."""

from __future__ import annotations

import socket

import httpx
import pytest

import tools.web_search as web_search
from core.tool import ITool
from tools.web_search import (
    MAX_PAGE_CHARS,
    FetchPageTool,
    SearchEngine,
    SearchResult,
    SearXNGProvider,
    SerpApiProvider,
    SSRFBlocked,
    WebSearchTool,
    _check_public_url,
    _TextExtractor,
    _truncate_text,
    _wrap_result,
    build_web_tools,
)


def _fake_getaddrinfo(ip: str):
    def fn(host, port):
        return [(socket.AF_INET, 1, 6, "", (ip, 0))]

    return fn


def _patch_client(monkeypatch, handler):
    """Make every `httpx.AsyncClient` built during the test use a `MockTransport`."""
    transport = httpx.MockTransport(handler)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


class TestCheckPublicUrl:
    def test_disallowed_scheme_rejected(self):
        with pytest.raises(SSRFBlocked, match="scheme"):
            _check_public_url("ftp://example.com/file")

    def test_missing_hostname_rejected(self):
        with pytest.raises(SSRFBlocked, match="hostname"):
            _check_public_url("http:///path")

    def test_unresolvable_host_rejected(self, monkeypatch):
        def fail(host, port):
            raise socket.gaierror("not found")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        with pytest.raises(SSRFBlocked, match="resolve"):
            _check_public_url("http://doesnotexist.invalid")

    @pytest.mark.parametrize(
        "ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254"]
    )
    def test_private_or_local_address_rejected(self, monkeypatch, ip):
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(ip))
        with pytest.raises(SSRFBlocked, match="private/local"):
            _check_public_url("http://evil.test")

    def test_public_address_allowed(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
        _check_public_url("http://example.test")  # must not raise

    def test_any_private_address_among_several_blocks(self, monkeypatch):
        def fn(host, port):
            return [
                (socket.AF_INET, 1, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, 1, 6, "", ("127.0.0.1", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fn)
        with pytest.raises(SSRFBlocked):
            _check_public_url("http://mixed.test")

    def test_loopback_literal_is_rejected_without_dns(self):
        # 127.0.0.1 is already an IP literal: getaddrinfo resolves it locally,
        # no real DNS/network call happens.
        with pytest.raises(SSRFBlocked, match="private/local"):
            _check_public_url("http://127.0.0.1/secret")


class TestSearchEngineContract:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            SearchEngine()


class TestSearXNGProvider:
    def test_strips_trailing_slash_from_url(self):
        assert SearXNGProvider("http://localhost:8080/").url == "http://localhost:8080"

    async def test_search_builds_results_from_json(self, monkeypatch):
        def handler(request):
            assert request.url.params["q"] == "python"
            assert request.url.params["format"] == "json"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "A", "url": "https://a.test", "content": "snippet a"},
                        {"title": "B", "url": "https://b.test", "content": "snippet b"},
                    ]
                },
            )

        _patch_client(monkeypatch, handler)
        provider = SearXNGProvider("http://localhost:8080")
        results = await provider.search("python", max_results=5)

        assert results == [
            SearchResult(title="A", url="https://a.test", snippet="snippet a"),
            SearchResult(title="B", url="https://b.test", snippet="snippet b"),
        ]

    async def test_search_caps_at_max_results(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200,
                json={"results": [{"title": str(i), "url": "u", "content": ""} for i in range(5)]},
            )

        _patch_client(monkeypatch, handler)
        provider = SearXNGProvider("http://localhost:8080")
        results = await provider.search("q", max_results=2)
        assert len(results) == 2

    async def test_search_raises_on_http_error(self, monkeypatch):
        _patch_client(monkeypatch, lambda request: httpx.Response(500))
        provider = SearXNGProvider("http://localhost:8080")
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("q", 5)

    async def test_missing_fields_default_to_empty_string(self, monkeypatch):
        _patch_client(monkeypatch, lambda r: httpx.Response(200, json={"results": [{}]}))
        provider = SearXNGProvider("http://localhost:8080")
        results = await provider.search("q", 5)
        assert results == [SearchResult(title="", url="", snippet="")]

    def test_from_config_builds_provider(self):
        provider = SearXNGProvider.from_config({"url": "http://localhost:8080", "timeout": 3.0})
        assert provider.url == "http://localhost:8080"
        assert provider.timeout == 3.0

    def test_from_config_missing_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            SearXNGProvider.from_config({})


class TestSerpApiProvider:
    def test_is_a_search_engine(self):
        assert isinstance(SerpApiProvider(api_key="k"), SearchEngine)

    async def test_search_builds_results_from_organic_results(self, monkeypatch):
        def handler(request):
            assert request.url.params["engine"] == "google"
            assert request.url.params["q"] == "python"
            assert request.url.params["api_key"] == "secret-key"
            return httpx.Response(
                200,
                json={
                    "organic_results": [
                        {"title": "A", "link": "https://a.test", "snippet": "snippet a"},
                        {"title": "B", "link": "https://b.test", "snippet": "snippet b"},
                    ]
                },
            )

        _patch_client(monkeypatch, handler)
        provider = SerpApiProvider(api_key="secret-key")
        results = await provider.search("python", max_results=5)

        assert results == [
            SearchResult(title="A", url="https://a.test", snippet="snippet a"),
            SearchResult(title="B", url="https://b.test", snippet="snippet b"),
        ]

    async def test_search_caps_at_max_results(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "organic_results": [
                        {"title": str(i), "link": "u", "snippet": ""} for i in range(5)
                    ]
                },
            )

        _patch_client(monkeypatch, handler)
        provider = SerpApiProvider(api_key="k")
        results = await provider.search("q", max_results=2)
        assert len(results) == 2

    async def test_search_raises_on_http_error(self, monkeypatch):
        _patch_client(monkeypatch, lambda request: httpx.Response(401))
        provider = SerpApiProvider(api_key="bad-key")
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("q", 5)

    async def test_missing_fields_default_to_empty_string(self, monkeypatch):
        _patch_client(monkeypatch, lambda r: httpx.Response(200, json={"organic_results": [{}]}))
        provider = SerpApiProvider(api_key="k")
        results = await provider.search("q", 5)
        assert results == [SearchResult(title="", url="", snippet="")]

    def test_from_config_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "env-key")
        provider = SerpApiProvider.from_config({"timeout": 3.0})
        assert provider.api_key == "env-key"
        assert provider.timeout == 3.0

    def test_from_config_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
            SerpApiProvider.from_config({})


class TestTextExtractor:
    def test_strips_script_and_style(self):
        extractor = _TextExtractor()
        extractor.feed("<p>Hello</p><script>evil()</script><style>.x{}</style><div>World</div>")
        text = extractor.text()
        assert "Hello" in text
        assert "World" in text
        assert "evil" not in text
        assert ".x" not in text

    def test_collapses_whitespace_and_drops_empty_lines(self):
        extractor = _TextExtractor()
        extractor.feed("<p>\n   Hello   \n</p>\n\n<div>World</div>")
        assert extractor.text() == "Hello\nWorld"


class TestWrapResult:
    def test_wraps_with_url_attribute(self):
        assert (
            _wrap_result("https://x.test", "body")
            == '<web_result url="https://x.test">\nbody\n</web_result>'
        )


class TestTruncateText:
    def test_short_text_is_untouched(self):
        assert _truncate_text("hello") == "hello"

    def test_long_text_is_truncated_with_a_note(self):
        text = "x" * (MAX_PAGE_CHARS + 100)
        result = _truncate_text(text)
        assert result.startswith("x" * 10)
        assert "truncated" in result
        assert len(result) < len(text)


class _FakeEngine(SearchEngine):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append((query, max_results))
        return self._results[:max_results]


class TestWebSearchTool:
    async def test_run_wraps_each_result(self):
        engine = _FakeEngine([SearchResult(title="T", url="https://x.test", snippet="S")])
        tool = WebSearchTool(engine)
        result = await tool.run(query="q")
        assert '<web_result url="https://x.test">' in result
        assert "T" in result
        assert "S" in result

    async def test_no_results_message(self):
        tool = WebSearchTool(_FakeEngine([]))
        assert await tool.run(query="q") == "No results found."

    @pytest.mark.parametrize("max_results", [0, 11, -1])
    async def test_max_results_out_of_range_raises(self, max_results):
        tool = WebSearchTool(_FakeEngine([]))
        with pytest.raises(ValueError):
            await tool.run(query="q", max_results=max_results)

    async def test_default_max_results_is_5(self):
        engine = _FakeEngine([SearchResult("t", "u", "s") for _ in range(10)])
        tool = WebSearchTool(engine)
        await tool.run(query="q")
        assert engine.calls == [("q", 5)]

    def test_schema_and_contract(self):
        tool = WebSearchTool(_FakeEngine([]))
        assert isinstance(tool, ITool)
        assert tool.name == "web_search"
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False
        assert tool.parameters["required"] == ["query"]


class TestFetchPageTool:
    async def test_extracts_main_text_and_wraps(self, monkeypatch):
        monkeypatch.setattr(web_search, "_check_public_url", lambda url: None)
        html = b"<html><script>evil()</script><p>Hello world</p></html>"
        _patch_client(monkeypatch, lambda request: httpx.Response(200, content=html))

        tool = FetchPageTool()
        result = await tool.run("https://example.test/page")

        assert "Hello world" in result
        assert "evil" not in result
        assert result.startswith('<web_result url="https://example.test/page">')

    async def test_stops_streaming_past_max_page_bytes(self, monkeypatch):
        monkeypatch.setattr(web_search, "_check_public_url", lambda url: None)
        big_html = b"<p>" + b"a" * 5000 + b"</p>"
        _patch_client(monkeypatch, lambda request: httpx.Response(200, content=big_html))

        tool = FetchPageTool(max_page_bytes=100)
        result = await tool.run("https://example.test/big")
        # The raw page is far bigger than max_page_bytes; the result staying
        # small proves the stream was cut short, not downloaded in full then
        # truncated after the fact.
        assert len(result) < 500

    async def test_http_error_propagates(self, monkeypatch):
        monkeypatch.setattr(web_search, "_check_public_url", lambda url: None)
        _patch_client(monkeypatch, lambda request: httpx.Response(404))

        tool = FetchPageTool()
        with pytest.raises(httpx.HTTPStatusError):
            await tool.run("https://example.test/missing")

    async def test_ssrf_guard_runs_before_any_request(self, monkeypatch):
        def boom(request):
            raise AssertionError("should never make a real request")

        _patch_client(monkeypatch, boom)

        tool = FetchPageTool()
        with pytest.raises(SSRFBlocked):
            await tool.run("http://127.0.0.1/secret")

    def test_schema_and_contract(self):
        tool = FetchPageTool()
        assert isinstance(tool, ITool)
        assert tool.name == "fetch_page"
        assert tool.parameters["additionalProperties"] is False
        assert tool.parameters["required"] == ["url"]


class TestBuildWebTools:
    def test_returns_web_search_and_fetch_page(self):
        tools = build_web_tools({"provider": "searxng", "url": "http://localhost:8080"}, None)
        assert [t.name for t in tools] == ["web_search", "fetch_page"]
        assert isinstance(tools[0].engine, SearXNGProvider)

    def test_provider_defaults_to_searxng_without_a_serpapi_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        tools = build_web_tools({"url": "http://localhost:8080"}, None)
        assert isinstance(tools[0].engine, SearXNGProvider)

    def test_provider_defaults_to_serpapi_when_a_key_is_set(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "env-key")
        tools = build_web_tools({}, None)
        assert isinstance(tools[0].engine, SerpApiProvider)

    def test_explicit_provider_is_never_overridden_by_the_env(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "env-key")
        tools = build_web_tools({"provider": "searxng", "url": "http://localhost:8080"}, None)
        assert isinstance(tools[0].engine, SearXNGProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown search provider"):
            build_web_tools({"provider": "bing", "url": "http://x"}, None)

    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            build_web_tools({"provider": "searxng"}, None)

    def test_timeout_is_shared_across_both_tools(self):
        web, fetch = build_web_tools({"url": "http://localhost:8080", "timeout": 2.5}, None)
        assert web.engine.timeout == 2.5
        assert fetch.timeout == 2.5

    def test_serpapi_provider_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "env-key")
        web, _fetch = build_web_tools({"provider": "serpapi"}, None)
        assert isinstance(web.engine, SerpApiProvider)
        assert web.engine.api_key == "env-key"

    def test_serpapi_without_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
            build_web_tools({"provider": "serpapi"}, None)
