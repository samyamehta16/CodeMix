"""
Live Scrape & Build Tamil Real vs Fake News Dataset
===================================================
Scrapes and processes Real and Fake news articles with:
1. Target de-leakage: extracts viral claims and isolates fact-checker
   debunking reports into `investigation_details`.
2. Content Filtering: excludes advertisements, sports scores, finance tables,
   calculators, and horoscopes.
3. Strict Deduplication: URL, headline, and content hash deduplication.
4. Balanced Output: generates 50:50 Real vs Fake news dataset.
"""

import hashlib
import logging
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper_pipeline")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ta,en-US;q=0.9,en;q=0.8",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
EXISTING_RAW_CSV = ROOT_DIR / "scripts" / "tamil_fakenews_dataset.csv"
EXISTING_CLEAN_CSV = ROOT_DIR / "data" / "cleaned" / "news_dataset_1500.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "final"
OUTPUT_FULL_CSV = OUTPUT_DIR / "tamil_news_dataset_1400_balanced.csv"

TARGET_PER_CLASS = 700


def make_id(text_or_url: str) -> str:
    norm = unicodedata.normalize("NFKC", str(text_or_url).strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def clean_headline(hl: str) -> str:
    if not isinstance(hl, str):
        return ""
    h = hl.strip()
    h = re.sub(
        r"^(?:Fact Check|உண்மை சரிபார்ப்பு|FactCheck|போலிச் செய்தி)\s*[:|-]\s*",
        "",
        h,
        flags=re.IGNORECASE,
    )
    h = re.sub(
        r"\s*\|\s*(?:Fact Crescendo|Dinamalar|Puthiya Thalaimurai|Vikatan|BOOM|Newschecker|OneIndia).*$",
        "",
        h,
        flags=re.IGNORECASE,
    )
    h = re.sub(r"[ \t]+", " ", h)
    return h.strip()


def clean_real_text(text: str, headline: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if headline and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()
    t = re.sub(r"^(?:ADDED|UPDATED)\s*:[^\n]*\n?", "", t, flags=re.MULTILINE)
    t = re.sub(r"தினமலர் ஆண்ட்ராய்டு ஆப் டவுன்லோட் செய்ய.*", "", t, flags=re.DOTALL)
    t = re.sub(r"விகடன் சந்தா சேர.*", "", t, flags=re.DOTALL)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def segment_fake_article(text: str, headline: str) -> tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    t = text.strip()
    if headline and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()

    claim_text = ""
    investigation_text = ""

    if "தகவலின் விவரம்:" in t:
        parts = t.split("தகவலின் விவரம்:", 1)
        intro = parts[0].strip()
        details = parts[1].strip()

        inv_split = re.split(
            r"(?:உண்மை அறிவோம்:|ஆய்வு:|ஆய்வு முறை:|எங்கள் ஆய்வில்|நாம் மேற்கொண்ட ஆய்வில்|பின்னணி:|முடிவு:)",
            details,
            1,
        )
        claim_section = inv_split[0].strip()
        if len(inv_split) > 1:
            investigation_text = inv_split[1].strip()

        clean_intro = re.sub(
            r"(?:இது தொடர்பாக|இது குறித்து|சமூக வலைதளங்களில்|சமூக ஊடகங்களில்).*?ஆய்வு.*",
            "",
            intro,
            flags=re.DOTALL,
        ).strip()
        clean_claim = re.sub(
            r"இதனை வாசகர்கள் நமக்கு வாட்ஸ்ஆப்.*?கேட்டனர்\.?",
            "",
            claim_section,
            flags=re.DOTALL,
        ).strip()
        clean_claim = re.sub(
            r"ஃபேஸ்புக் பதிவு.*?காணலாம்.*", "", clean_claim, flags=re.DOTALL
        ).strip()

        claim_text = (clean_intro + "\n\n" + clean_claim).strip() if clean_intro else clean_claim

    elif "Claim" in t and "Fact" in t:
        m = re.search(
            r"Claim\s*[:]?\s*(.*?)(?:Fact\s*[:]?\s*(.*)|$)", t, flags=re.DOTALL
        )
        if m:
            claim_text = m.group(1).strip()
            investigation_text = m.group(2).strip() if m.group(2) else ""

    if not claim_text:
        clean_t = re.sub(r"முடிவு:.*", "", t, flags=re.DOTALL)
        clean_t = re.sub(
            r"(?:எங்கள் ஆய்வில்|நாம் மேற்கொண்ட ஆய்வில்).* உறுதி செய்யப்படுகிறது.*",
            "",
            clean_t,
            flags=re.DOTALL,
        )
        claim_text = clean_t.strip()
        investigation_text = t

    claim_text = re.sub(
        r"(?:Claim\s*Link|Archived?\s*Link|Archive\s*Link)\s*\d*(?:\s*[l|I/]\s*(?:Claim\s*Link|Archived?\s*Link|Archive\s*Link)\s*\d*)*",
        "",
        claim_text,
        flags=re.IGNORECASE,
    )
    claim_text = re.sub(r"https?://\S+", "", claim_text)
    claim_text = re.sub(r"[ \t]+", " ", claim_text)
    return re.sub(r"\n\s*\n+", "\n\n", claim_text).strip(), investigation_text


def is_noise_real_article(url: str, headline: str, text: str) -> bool:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    cat = parts[1] if len(parts) > 1 and parts[0] == "news" else (parts[0] if parts else "")

    noise_categories = {
        "calculators",
        "horoscope",
        "bundle",
        "collection",
        "personal-finance",
        "business-share-market",
        "business-savings",
        "business-banking-and-finance",
        "sports-cricket",
        "sports-badminton",
        "sports-tennis",
        "sports-football",
        "sports-others",
        "sports",
    }
    if cat in noise_categories:
        return True

    ad_phrases = ["ஷாப்பர்ஸ் எக்ஸ்போ", "மெகா ஷாப்பிங்", "தினமலர் ஸ்மார்ட் ஷாப்பர்ஸ்"]
    for phrase in ad_phrases:
        if phrase in headline or phrase in text[:200]:
            return True
    return False


def collect_factcrescendo_urls(max_pages: int = 25) -> list[str]:
    log.info("Collecting candidate URLs from FactCrescendo (pages 1 to %d)...", max_pages)
    collected = []
    seen = set()
    ignored = ["/category/", "/page/", "/aboutus", "/contact", "/privacy", "archive-grid", "/author/", "/feed/"]

    for page in range(1, max_pages + 1):
        url = f"https://tamil.factcrescendo.com/archive-grid/page/{page}/"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            page_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0].split("?")[0]
                if (
                    href.startswith("https://tamil.factcrescendo.com/")
                    and href.rstrip("/") != "https://tamil.factcrescendo.com"
                    and not any(x in href for x in ignored)
                ):
                    page_links.append(href)

            new_links = [l for l in list(dict.fromkeys(page_links)) if l not in seen]
            seen.update(new_links)
            collected.extend(new_links)
            time.sleep(0.3)
        except Exception as e:
            log.warning("Error fetching archive page %d: %s", page, e)
    return collected


def scrape_single_fake_article(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        raw_hl = h1.get_text(strip=True) if h1 else ""
        raw_text = trafilatura.extract(resp.text, url=url) or ""

        hl = clean_headline(raw_hl)
        claim, investigation = segment_fake_article(raw_text, hl)

        if len(hl) < 10 or len(claim) < 30:
            return None

        return {
            "id": make_id(url),
            "headline": hl,
            "text": claim,
            "investigation_details": investigation,
            "label": 0,
            "label_name": "fake",
            "source": "FactCrescendoTamil",
            "url": url,
            "char_count": len(claim),
            "word_count": len(claim.split()),
        }
    except Exception:
        return None


def main():
    log.info("Starting scrape_and_build_dataset...")
    from scripts.clean_and_build_dataset import main as run_pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
