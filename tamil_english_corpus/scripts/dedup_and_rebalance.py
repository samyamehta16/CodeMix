"""
Strict Deduplication and Balancing for Real & Fake News Dataset
---------------------------------------------------------------
1. Performs multi-level deduplication:
   - Exact URLs
   - Normalized headlines (alphanumeric only, lowercase)
   - Normalized text prefixes (first 120 characters)
   - Jaccard similarity on word tokens (> 0.85 threshold)
2. Re-balances to exactly 750 unique Real (1) and 750 unique Fake (0) rows.
3. Automatically fetches fresh fake articles from FactCrescendo to replace any removed duplicates.
4. Overwrites data/cleaned/news_dataset_1500.csv and scripts/news_dataset_1500.csv.
"""

import hashlib
import logging
import os
import re
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dedup_rebalance")

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_1 = ROOT_DIR / "data" / "cleaned" / "news_dataset_1500.csv"
CLEANED_CSV_2 = ROOT_DIR / "scripts" / "news_dataset_1500.csv"
ORIGINAL_CSV = ROOT_DIR / "scripts" / "tamil_fakenews_dataset.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ta,en-US;q=0.9,en;q=0.8",
}


def norm_str(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[\s\W_]+", "", s.lower())


def get_shingles(s: str) -> set[str]:
    return set(re.findall(r"\w+", str(s).lower()))


def clean_real_text(text: str, headline: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if isinstance(headline, str) and headline.strip() and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()
    t = re.sub(r'^(?:ADDED|UPDATED)\s*:[^\n]*\n?', '', t, flags=re.MULTILINE)
    t = re.sub(r'[ \t]+', ' ', t)
    return re.sub(r'\n\s*\n+', '\n\n', t).strip()


def clean_fake_text(text: str, headline: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if isinstance(headline, str) and headline.strip() and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()
    t = re.sub(r'உண்மைப் பதிவைக் காண:[^\n]*\n?', '', t)
    t = re.sub(r'(?:Facebook|Archive|Twitter|YouTube)\s*I\s*(?:Facebook|Archive|Twitter|YouTube)[^\n]*\n?', '', t)
    t = re.sub(r'(?:முடிவு:|வாசகர்கள் தங்களுக்கு சந்தேகம்).*', '', t, flags=re.DOTALL)
    t = re.sub(r'சமூக ஊடகங்களில் எங்களை பின் தொடர.*', '', t, flags=re.DOTALL)
    t = re.sub(r'எங்களது வாட்ஸ் ஆப் எண்ணிற்கு.*', '', t, flags=re.DOTALL)
    t = re.sub(r'[ \t]+', ' ', t)
    return re.sub(r'\n\s*\n+', '\n\n', t).strip()


def clean_headline(hl: str) -> str:
    if not isinstance(hl, str):
        return ""
    h = hl.strip()
    h = re.sub(r'\s*\|\s*(?:Fact Crescendo|Dinamalar|Puthiya Thalaimurai|Vikatan).*$', '', h, flags=re.IGNORECASE)
    h = re.sub(r'[ \t]+', ' ', h)
    return h.strip()


def make_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def scrape_fresh_fake_article(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        raw_hl = h1.get_text(strip=True) if h1 else ""
        raw_text = trafilatura.extract(resp.text, url=url) or ""

        hl = clean_headline(raw_hl)
        body = clean_fake_text(raw_text, hl)

        if len(hl) < 10 or len(body) < 100:
            return None

        return {
            "id": make_id(url),
            "headline": hl,
            "text": body,
            "label": 0,
            "label_name": "fake",
            "source": "FactCrescendoTamil",
            "url": url,
            "char_count": len(body),
            "word_count": len(body.split()),
        }
    except Exception:
        return None


def fetch_candidate_urls_page(page: int) -> list[str]:
    url = f"https://tamil.factcrescendo.com/archive-grid/page/{page}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        page_links = []
        ignored = ['/category/', '/page/', '/aboutus', '/contact', '/privacy', 'archive-grid', '/author/', '/feed/']
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]
            if (href.startswith("https://tamil.factcrescendo.com/") and
                    href.rstrip("/") != "https://tamil.factcrescendo.com" and
                    not any(x in href for x in ignored)):
                page_links.append(href)
        return list(dict.fromkeys(page_links))
    except Exception:
        return []


def deduplicate_records(records: list[dict]) -> tuple[list[dict], int]:
    """Strict multi-level deduplication."""
    clean_records = []
    seen_urls = set()
    seen_hl = set()
    seen_ts = set()
    seen_shingles = []
    removed_count = 0

    for r in records:
        url = r["url"]
        hl = r["headline"]
        body = r["text"]

        nh = norm_str(hl)
        ts = norm_str(body[:120])
        sh = get_shingles(body)

        if url in seen_urls or nh in seen_hl or ts in seen_ts:
            removed_count += 1
            continue

        # Check Jaccard similarity with existing records
        is_jaccard_dupe = False
        if len(sh) >= 10:
            # Check against the last 150 added records
            for existing_sh in seen_shingles[-150:]:
                inter = len(sh.intersection(existing_sh))
                union = len(sh.union(existing_sh))
                if union > 0 and (inter / union) > 0.85:
                    is_jaccard_dupe = True
                    break

        if is_jaccard_dupe:
            removed_count += 1
            continue

        seen_urls.add(url)
        seen_hl.add(nh)
        seen_ts.add(ts)
        seen_shingles.append(sh)
        clean_records.append(r)

    return clean_records, removed_count


def main():
    log.info("Starting strict deduplication and balancing process...")
    current_df = pd.read_csv(CLEANED_CSV_1)
    log.info("Current dataset size: %d rows", len(current_df))

    # 1. Deduplicate Fake News
    fake_records = current_df[current_df["label"] == 0].to_dict("records")
    clean_fake, fake_dupes_removed = deduplicate_records(fake_records)
    log.info("Fake news: %d -> %d clean unique rows (%d duplicates removed)",
             len(fake_records), len(clean_fake), fake_dupes_removed)

    # If we need more fake news to reach 750, scrape fresh ones from archive-grid
    needed_fake = 750 - len(clean_fake)
    if needed_fake > 0:
        log.info("Need %d fresh fake articles to reach exactly 750...", needed_fake)
        existing_fake_urls = {r["url"] for r in clean_fake}
        page = 23
        while needed_fake > 0 and page <= 35:
            cand_urls = fetch_candidate_urls_page(page)
            for u in cand_urls:
                if u in existing_fake_urls:
                    continue
                art = scrape_fresh_fake_article(u)
                if not art:
                    continue
                # Test against current clean_fake
                test_list = clean_fake + [art]
                deduped, removed = deduplicate_records(test_list)
                if len(deduped) > len(clean_fake):
                    clean_fake.append(art)
                    existing_fake_urls.add(u)
                    needed_fake = 750 - len(clean_fake)
                    log.info("  Scraped fresh unique fake article [%s]: %s (remaining: %d)",
                             art["id"], art["headline"][:50], needed_fake)
                    if needed_fake == 0:
                        break
            page += 1

    clean_fake = clean_fake[:750]
    log.info("Final unique Fake news count: %d", len(clean_fake))

    # 2. Re-pool and Deduplicate Real News
    orig_df = pd.read_csv(ORIGINAL_CSV, encoding_errors="replace")
    real_raw = orig_df[orig_df["label"] == "real"].copy()
    candidate_real = []
    for _, row in real_raw.iterrows():
        url = str(row.get("url", "")).strip()
        raw_hl = str(row.get("headline", "")).strip()
        raw_text = str(row.get("text", "")).strip()
        source = str(row.get("source_name", "RealNews")).strip()

        hl = clean_headline(raw_hl)
        body = clean_real_text(raw_text, hl)

        if len(hl) < 10 or len(body) < 100:
            continue
        candidate_real.append({
            "id": make_id(url),
            "headline": hl,
            "text": body,
            "label": 1,
            "label_name": "real",
            "source": source,
            "url": url,
            "char_count": len(body),
            "word_count": len(body.split()),
        })

    clean_real, real_dupes_removed = deduplicate_records(candidate_real)
    log.info("Real news: %d candidates -> %d unique non-duplicate rows (%d duplicates removed)",
             len(candidate_real), len(clean_real), real_dupes_removed)

    clean_real = clean_real[:750]
    log.info("Final unique Real news count: %d", len(clean_real))

    # 3. Combine and Global Cross-Deduplication
    combined = clean_real + clean_fake
    final_records, cross_dupes = deduplicate_records(combined)
    log.info("Combined total: %d (cross duplicates: %d)", len(final_records), cross_dupes)

    final_df = pd.DataFrame(final_records)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Save to files
    final_df.to_csv(CLEANED_CSV_1, index=False, encoding="utf-8-sig")
    final_df.to_csv(CLEANED_CSV_2, index=False, encoding="utf-8-sig")
    log.info("Successfully updated: %s", CLEANED_CSV_1)
    log.info("Successfully updated: %s", CLEANED_CSV_2)

    # 5. Verification checks
    print("\n" + "=" * 60)
    print("FINAL DATASET DUPLICATION AUDIT")
    print("=" * 60)
    print(f"Total Rows:                   {len(final_df)}")
    print(f"Real News (1):                {(final_df['label'] == 1).sum()}")
    print(f"Fake News (0):                {(final_df['label'] == 0).sum()}")
    print(f"Duplicate Exact Rows:         {final_df.duplicated().sum()}")
    print(f"Duplicate IDs:                {final_df.duplicated(subset=['id']).sum()}")
    print(f"Duplicate URLs:               {final_df.duplicated(subset=['url']).sum()}")
    print(f"Duplicate Exact Headlines:    {final_df.duplicated(subset=['headline']).sum()}")
    final_df['norm_hl'] = final_df['headline'].apply(norm_str)
    print(f"Duplicate Normalized HLs:     {final_df.duplicated(subset=['norm_hl']).sum()}")
    print(f"Duplicate Exact Texts:        {final_df.duplicated(subset=['text']).sum()}")
    final_df['norm_ts'] = final_df['text'].str.slice(0, 120).apply(norm_str)
    print(f"Duplicate Text Starts (120c): {final_df.duplicated(subset=['norm_ts']).sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
