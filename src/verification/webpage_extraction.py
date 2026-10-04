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

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TamilCodemixResearchBot/1.0; academic project)"}
TIMEOUT_SECONDS = 12
MAX_CHARS = 6000  # cap so the LLM prompt doesn't blow up on very long articles


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


def extract_from_url(result: SearchResult) -> Evidence:
    if not result.url:
        return Evidence(result=result, extraction_success=False, extraction_error="No URL provided")

    try:
        resp = requests.get(result.url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return Evidence(result=result, extraction_success=False,
                         extraction_error=f"Timed out after {TIMEOUT_SECONDS}s")
    except requests.exceptions.SSLError as e:
        return Evidence(result=result, extraction_success=False,
                         extraction_error=f"SSL error: {e}")
    except requests.exceptions.RequestException as e:
        return Evidence(result=result, extraction_success=False,
                         extraction_error=f"Request failed: {e}")

    if resp.status_code != 200:
        return Evidence(result=result, extraction_success=False,
                         extraction_error=f"HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        return Evidence(result=result, extraction_success=False,
                         extraction_error=f"Non-HTML content-type: {content_type or 'unknown'}")

    text = _extract_with_trafilatura(resp.text) or _extract_with_bs4(resp.text)
    if not text:
        return Evidence(result=result, extraction_success=False,
                         extraction_error="Could not extract readable text from page "
                                          "(paywalled, JS-rendered, or unsupported layout)")

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
