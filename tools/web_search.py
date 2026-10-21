import ipaddress
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

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
