"""
Module 4: Webpage text extraction.

Given a search result URL, fetches the page and extracts the main
readable text (not raw HTML, not just the search snippet) so the LLM
verifier has real article content to reason over. Uses trafilatura first
(purpose-built for stripping nav/ads/boilerplate from articles); falls
back to a basic BeautifulSoup extraction if trafilatura can't parse it,
or if it isn't installed.

Never raises for "normal" failure modes (timeout, 404, non-HTML,
unparseable page) - always returns an Evidence object with
extraction_success=False and a specific extraction_error so the caller
can report exactly what went wrong instead of silently dropping evidence.

Standalone test:
    python src/verification/webpage_extraction.py
    python src/verification/webpage_extraction.py https://example.com
"""
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent))
from schemas import Evidence, SearchResult

# Browser-like UA: many news sites block generic research-bot strings.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT_SECONDS = 12
MAX_CHARS = 6000  # cap so the LLM prompt doesn't blow up on very long articles
MIN_SNIPPET_CHARS = 40  # Serper snippets shorter than this are not useful as evidence


def _extract_with_trafilatura(html: str):
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        return trafilatura.extract(html, include_comments=False, include_tables=False)
    except Exception:
        return None


def _extract_with_bs4(html: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text if text.strip() else None
    except Exception:
        return None


def _snippet_fallback(result: SearchResult, reason: str) -> Evidence:
    """Use the Serper snippet when full-page fetch/parse fails (YouTube,
    Facebook, 403s, JS shells). Weaker than article text but still usable
    for the LLM verifier — better than UNVERIFIED with zero evidence."""
    snippet = (result.snippet or "").strip()
    if len(snippet) < MIN_SNIPPET_CHARS:
        return Evidence(result=result, extraction_success=False, extraction_error=reason)
    text = f"[Search snippet only — full page unavailable: {reason}]\n{result.title}\n{snippet}"
    return Evidence(result=result, extracted_text=text[:MAX_CHARS], extraction_success=True)


def extract_from_url(result: SearchResult) -> Evidence:
    if not result.url:
        return Evidence(result=result, extraction_success=False, extraction_error="No URL provided")

    try:
        resp = requests.get(result.url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return _snippet_fallback(result, f"Timed out after {TIMEOUT_SECONDS}s")
    except requests.exceptions.SSLError as e:
        return _snippet_fallback(result, f"SSL error: {e}")
    except requests.exceptions.RequestException as e:
        return _snippet_fallback(result, f"Request failed: {e}")

    if resp.status_code != 200:
        return _snippet_fallback(result, f"HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if content_type and "html" not in content_type.lower() and "text/" not in content_type.lower():
        return _snippet_fallback(result, f"Non-HTML content-type: {content_type}")

    text = _extract_with_trafilatura(resp.text) or _extract_with_bs4(resp.text)
    if not text:
        return _snippet_fallback(
            result,
            "Could not extract readable text from page "
            "(paywalled, JS-rendered, or unsupported layout)",
        )

    return Evidence(result=result, extracted_text=text[:MAX_CHARS], extraction_success=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("url", nargs="?", default="https://www.rbi.org.in/")
    args = p.parse_args()

    dummy_result = SearchResult(title="test", url=args.url, snippet="", source_domain="")
    ev = extract_from_url(dummy_result)
    if ev.extraction_success:
        print(f"Extracted {len(ev.extracted_text)} chars:\n")
        print(ev.extracted_text[:500])
    else:
        print(f"[FAILED] {ev.extraction_error}")
