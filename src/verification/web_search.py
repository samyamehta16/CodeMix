"""
Module 3: Web search / retrieval interface.

Defines a provider-agnostic interface (`SearchProvider`) so the concrete
search backend can be swapped without touching any other module. Wired to
Serper.dev (https://serper.dev - a lightweight wrapper around Google
Search results) as the first provider, chosen for its simple REST API and
free tier.

To add another provider later (Tavily, Google Programmable Search, Bing,
Brave, ...): implement a new SearchProvider subclass with a `search()`
method returning `List[SearchResult]`, and add a branch for it in
`get_search_provider()`.

IMPORTANT: this module's ONLY job is query -> retrieved results. It must
NEVER decide truth/falsity - that's llm_verifier.py's job.

Standalone test:
    python src/verification/web_search.py
    python src/verification/web_search.py "RBI new 5000 rupee note"
"""
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEARCH_PROVIDER, SERPER_API_KEY

sys.path.append(str(Path(__file__).resolve().parent))
from schemas import SearchResult

import cache


class SearchProviderError(RuntimeError):
    """Raised on missing configuration, API failure, rate limit, or
    timeout. The caller should treat this as 'evidence unavailable' and
    say so explicitly - never fabricate results to fill the gap."""


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        ...


class NotConfiguredProvider(SearchProvider):
    """Used when no usable search API key/provider is set. Fails loudly
    and clearly instead of silently returning an empty/fake result."""
    name = "not_configured"

    def __init__(self, reason: str):
        self.reason = reason

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        raise SearchProviderError(
            f"No web search provider is configured ({self.reason}). "
            f"Set SEARCH_PROVIDER and the matching API key in .env - see .env.example."
        )


class SerperSearchProvider(SearchProvider):
    """https://serper.dev - Google Search results via a simple REST API.
    Free tier: 2,500 one-time queries, no credit card required."""
    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5, use_cache: bool = True) -> List[SearchResult]:
        cache_key = f"{query}::{num_results}"
        if use_cache:
            cached = cache.get("search_serper", cache_key)
            if cached is not None:
                return [SearchResult(**item) for item in cached]

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}

        try:
            resp = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=15)
        except requests.exceptions.Timeout:
            raise SearchProviderError(f"Serper request timed out for query: {query!r}")
        except requests.exceptions.RequestException as e:
            raise SearchProviderError(f"Serper request failed for query {query!r}: {e}")

        if resp.status_code == 429:
            raise SearchProviderError("Serper rate limit / quota exceeded (HTTP 429).")
        if resp.status_code == 401:
            raise SearchProviderError("Serper rejected the API key (HTTP 401) - check SERPER_API_KEY.")
        if resp.status_code != 200:
            raise SearchProviderError(f"Serper returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError:
            raise SearchProviderError("Serper returned a non-JSON response.")

        organic = data.get("organic", [])
        results = []
        for i, item in enumerate(organic[:num_results]):
            url = item.get("link", "")
            results.append(SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
                source_domain=urlparse(url).netloc.replace("www.", ""),
                published_date=item.get("date"),
                rank=i + 1,
            ))

        if use_cache:
            cache.set("search_serper", cache_key, [asdict(r) for r in results])
        return results


def get_search_provider() -> SearchProvider:
    """Factory - reads SEARCH_PROVIDER/keys from config (.env) and returns
    the matching provider, or a NotConfiguredProvider if nothing usable is
    set. This is the single place that needs updating to add/switch
    providers."""
    provider_name = (SEARCH_PROVIDER or "").lower().strip()

    if provider_name == "serper":
        if not SERPER_API_KEY:
            return NotConfiguredProvider("SEARCH_PROVIDER=serper but SERPER_API_KEY is empty")
        return SerperSearchProvider(SERPER_API_KEY)

    if not provider_name:
        return NotConfiguredProvider("SEARCH_PROVIDER is not set in .env")

    return NotConfiguredProvider(
        f"Unknown SEARCH_PROVIDER '{provider_name}' - only 'serper' is implemented so far"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default="RBI new 5000 rupee note")
    args = p.parse_args()

    provider = get_search_provider()
    print(f"Using provider: {provider.name}")
    try:
        results = provider.search(args.query, num_results=5)
    except SearchProviderError as e:
        print(f"[UNAVAILABLE] {e}")
    else:
        for r in results:
            print(f"- {r.title} ({r.source_domain}) -> {r.url}")
