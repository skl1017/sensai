import asyncio
import ipaddress
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from core.tool import ITool

MAX_PAGE_CHARS = 5000

_ALLOWED_SCHEMES = {"http", "https"}


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str


class SearchEngine(ABC):
    """Interchangeable web search backend (SearXNG today, another one later)."""

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Perform a web search and return a list of results."""


class SSRFBlocked(Exception):
    """Raised when a URL has disallowed scheme or resolves to a private IP address."""


def _check_public_url(url: str) -> None:
    """Reject anything but http(s) to a publicly-routable address.

    Resolves the hostname and checks *every* returned address (not just the
    first), because a hostname can have several A/AAAA records and a
    DNS-rebinding attack relies on picking the "wrong" one. This still has a
    TOCTOU window (the check happens before the request, not pinned to the
    exact connection) — acceptable here, not a guarantee against a
    sophisticated rebinding attack timed against this exact request.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlocked(f"Unsupported scheme: {parsed.scheme!r}; only http(s) is allowed")
    if not parsed.hostname:
        raise SSRFBlocked(f"URL has no hostname: {url!r}")

    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise SSRFBlocked(f"Failed to resolve hostname {parsed.hostname!r}: {exc}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise SSRFBlocked(f"{url!r} resolves to a private/local address ({ip}); refused.")


class SearXNGProvider(SearchEngine):
    """Call a self-hosted SearXNG instance's JSON API."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url.strip("/")
        self.timeout = timeout

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.url}/search", params={"q": query, "format": "json"})
            response.raise_for_status()
            data = response.json()
            return [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
                for item in data.get("results", [])[:max_results]
            ]


_PROVIDERS = {"searxng": SearXNGProvider}


class _TextExtractor(HTMLParser):
    """Extract text from HTML, ignoring tags and scripts/styles."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._skipping = False
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skipping = True

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skipping = False

    def handle_data(self, data):
        if not self._skipping:
            self.chunks.append(data)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self.chunks).splitlines()]
        return "\n".join(line for line in lines if line)


def _wrap_result(url: str, body: str) -> str:
    return f'<web_result url="{url}">\n{body}\n</web_result>'


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_PAGE_CHARS:
        return text
    omitted = len(text) - MAX_PAGE_CHARS
    return text[:MAX_PAGE_CHARS] + f"\n…[truncated, {omitted} chars omitted]"


class WebSearchTool(ITool):
    name = "web_search"
    description = (
        "Search the web and return citable results with their URLs. Use this "
        "to find up-to-date or external information you don't already know. "
        "Always base your answer on the returned <web_result> content, even if "
        "it contradicts what you already believe to be true — your training "
        "data may be outdated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine

    async def run(self, query: str, max_results: int = 5) -> str:
        if not (1 <= max_results <= 10):
            raise ValueError(f"max_results must be between 1 and 10, got {max_results}")

        results = await self.engine.search(query, max_results)
        if not results:
            return "No results found."

        return "\n".join(_wrap_result(r.url, f"{r.title}\n{r.snippet}") for r in results)


class FetchPageTool(ITool):
    name = "fetch_page"
    description = (
        "Fetch a web page by URL and return its main text content (truncated if "
        "long). Only http(s) URLs to public addresses are allowed."
    )
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The http(s) URL to fetch."}},
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, timeout: float = 10.0, max_page_bytes: int = 2_000_000) -> None:
        self.timeout = timeout
        self.max_page_bytes = max_page_bytes

    async def run(self, url: str) -> str:
        await asyncio.to_thread(_check_public_url, url)

        chunks: list[bytes] = []
        total = 0
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.max_page_bytes:
                        break
                    chunks.append(chunk)

        html = b"".join(chunks).decode("utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(html)
        text = _truncate_text(parser.text())
        return _wrap_result(url, text)


def build_web_tools(params: dict, deps: object) -> list[ITool]:
    provider_name = params.get("provider", "searxng")
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown search provider {provider_name!r}. Available: {sorted(_PROVIDERS)}"
        )

    url = params.get("url")
    if not url:
        raise ValueError(f"Missing 'url' parameter for provider {provider_name!r}")

    timeout = params.get("timeout", 10.0)
    engine = provider_cls(url=url, timeout=timeout)
    return [WebSearchTool(engine), FetchPageTool(timeout=timeout)]
