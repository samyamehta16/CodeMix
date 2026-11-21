"""
Tamil-English Codemix Fake News Dataset Builder (v2 — scaled to 1000+ articles)
--------------------------------------------------------------------------------
Changes vs. the v1 script that only pulled ~30 articles:

1. PAGINATION — v1 only ever fetched page 1 of each listing URL, so it hit a hard
   ceiling around MAX_ARTICLES_PER_LISTING per category. v2 walks each listing
   through multiple pages using several common pagination URL shapes
   (`/page/2`, `?page=2`, `/page-2`, `-page-2`) and keeps whichever shape
   actually returns new, distinct links for that site. It stops paginating a
   listing when it hits `MAX_PAGES_PER_LISTING`, or when it sees
   `STALE_PAGE_LIMIT` consecutive pages with no new (unseen) article links.

2. ROBOTS.TXT CACHING — v1 re-downloaded and re-parsed robots.txt on *every
   single article request* (2x HTTP requests per article). v2 fetches and
   parses each domain's robots.txt exactly once and caches the RobotFileParser
   in memory for the rest of the run. Same policy is honored, just not
   re-fetched hundreds of times.

3. RESUMABLE STATE — v2 loads URLs already present in the output CSV (if it
   exists) into `seen_urls` at startup, so re-running the script to top up
   toward the target doesn't re-fetch or duplicate articles you already have.

4. TARGET-DRIVEN LOOP — instead of "one pass over one page per listing", the
   script now runs a target-driven loop per source: keep paginating and
   scraping until either (a) the source's own target is hit, (b) every listing
   for that source is exhausted, or (c) the global TARGET_TOTAL_ARTICLES is
   hit. Targets are split roughly evenly across "real" and "fake" sources so
   you don't end up with 1000 real articles and 20 fake ones.

5. LIGHT CONCURRENCY — sources are scraped in parallel with a small thread
   pool (default 4 workers, one source per worker at a time), while requests
   *within* a single source/domain stay strictly sequential with the same
   randomized polite delay as before. This parallelizes across domains
   without hammering any single domain harder than v1 did.

6. CLI FLAGS — `--target`, `--max-pages`, `--workers`, `--output` let you
   tune the run without editing constants.

IMPORTANT CAVEAT: I could not re-verify the pagination URL shapes for these
specific sites live (no network access to news domains from this environment).
The base listing URLs and article-URL regexes are unchanged from your
verified-live v1 script. The pagination shapes are the common patterns these
kinds of Tamil news/fact-check sites (several run on WordPress) typically use,
but you should spot-check that `page 2` for a couple of listings on each
domain actually shows different articles than page 1 before a full run —
run with `--target 40` first as a smoke test. If a site uses a pagination
shape not in PAGINATION_TEMPLATES, that source will just plateau at
whatever page 1 returns (same behavior as v1), not error out.

Output: tamil_fakenews_dataset.csv
Columns: id, source_name, source_type, url, published_date, headline, text,
         label, lang_mix_score, scraped_at
"""

import argparse
import csv
import hashlib
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from threading import Lock
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup
from langdetect import detect_langs  # noqa: F401 (available if you want stricter lang filtering)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ta,en-US;q=0.9,en;q=0.8",
}
REQUEST_DELAY_RANGE = (1.5, 3.0)  # seconds, randomized to be polite, per request
MAX_ARTICLES_PER_LISTING_PAGE = 60   # cap on links pulled from a single listing page
MAX_PAGES_PER_LISTING = 40           # hard ceiling so a runaway pattern can't loop forever
STALE_PAGE_LIMIT = 3                 # give up on a listing after this many pages with 0 new links

# Candidate pagination URL shapes, tried in this order for page N (N >= 2).
# {base} is the listing URL with any trailing slash stripped.
PAGINATION_TEMPLATES = [
    "{base}/page/{n}",     # WordPress-style (BOOM, FactCrescendo, many blogs)
    "{base}/page/{n}/",
    "{base}?page={n}",     # generic query-param pagination
    "{base}/page-{n}",
    "{base}-page-{n}",
    "{base}/{n}",          # bare numeric segment (some Dravidian-language CMS setups)
]

OUTPUT_CSV_DEFAULT = "tamil_fakenews_dataset.csv"
TARGET_TOTAL_DEFAULT = 1200  # aim above 1000 to allow for skipped/short articles


# --------------------------------------------------------------------------
# Sources (unchanged base listing URLs / regexes from the verified v1 script)
# --------------------------------------------------------------------------
SOURCES = [
    # ---------------- REAL NEWS (label = real) ----------------
    {
        "name": "PuthiyaThalaimurai",
        "label_type": "real",
        "listing_urls": [
            "https://www.puthiyathalaimurai.com/tamilnadu",
            "https://www.puthiyathalaimurai.com/india",
            "https://www.puthiyathalaimurai.com/world",
            "https://www.puthiyathalaimurai.com/sports",
            "https://www.puthiyathalaimurai.com/technology",
            "https://www.puthiyathalaimurai.com/health",
            "https://www.puthiyathalaimurai.com/lifestyle",
        ],
        "article_url_pattern": (
            r"https://www\.puthiyathalaimurai\.com/"
            r"(?!author/|collection/|search/|video-stories/|ampstories/|trending/)"
            r"[a-z-]+/[a-z0-9-]{10,}$"
        ),
    },
    {
        "name": "Dinamalar",
        "label_type": "real",
        "listing_urls": [
            "https://www.dinamalar.com/news/tamil-nadu-news",
            "https://www.dinamalar.com/news/india-tamil-news",
            "https://www.dinamalar.com/news/world-tamil-news",
            "https://www.dinamalar.com/news/sports",
            "https://www.dinamalar.com/news/business",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-chennai",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-coimbatore",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-madurai",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-trichirappalli",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-salem",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-tiruppur",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-erode",
            "https://www.dinamalar.com/news/tamil-nadu-district-news-thirunelveli",
        ],
        "article_url_pattern": r"https://www\.dinamalar\.com/news/[a-z-]+/[a-z0-9-]+/\d+$",
    },
    {
        "name": "Vikatan-TamilNadu",
        "label_type": "real",
        "listing_urls": ["https://www.vikatan.com/news/tamilnadu"],
        "article_url_pattern": r"https://www\.vikatan\.com/(?:[a-z-]+/){1,2}[a-z][a-z0-9-]{15,}$",
    },

    # ---------------- FACT-CHECK / FAKE (label = fake) ----------------
    {
        "name": "BOOMTamil-FactCheck",
        "label_type": "fake",
        "listing_urls": ["https://tamil.boomlive.in/fact-check"],
        "article_url_pattern": r"https://tamil\.boomlive\.in/[a-z-]+/[a-z0-9-]+$",
        "claim_boundary_regex": r"(?:உண்மை|Fact\s*Check|Claim|Conclusion|முடிவு)",
    },
    {
        "name": "FactCrescendoTamil",
        "label_type": "fake",
        "listing_urls": ["https://tamil.factcrescendo.com"],
        "article_url_pattern": r"https://tamil\.factcrescendo\.com/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/?$",
        "claim_boundary_regex": r"(?:உண்மை|Fact\s*Check|Claim|Conclusion|முடிவு|Verdict)",
    },
    {
        "name": "OneIndiaTamil-FactCheck",
        "label_type": "fake",
        "listing_urls": ["https://tamil.oneindia.com/fact-check/"],
        "article_url_pattern": r"https://tamil\.oneindia\.com/[a-z-]+/[a-z0-9-]+-\d+\.html$",
        "claim_boundary_regex": r"(?:உண்மை|Fact\s*Check|Claim|Conclusion|முடிவு|Verdict)",
    },
    {
        "name": "NewscheckerTamil",
        "label_type": "fake",
        "listing_urls": ["https://newschecker.in/ta"],
        "article_url_pattern": r"https://newschecker\.in/ta/fact-check-ta/(?:fact-check-ta/)?[a-z0-9-]+$",
        "claim_boundary_regex": r"(?:உண்மை|Fact\s*Check|Claim|Conclusion|முடிவு|Verdict)",
    },
]


@dataclass
class Article:
    id: str
    source_name: str
    source_type: str
    url: str
    published_date: str
    headline: str
    text: str
    label: str
    lang_mix_score: float
    scraped_at: str


# --------------------------------------------------------------------------
# Robots.txt — cached per domain instead of re-fetched per article
# --------------------------------------------------------------------------
_robots_cache: dict[str, RobotFileParser | None] = {}
_robots_lock = Lock()


def get_robot_parser(netloc: str, scheme: str) -> RobotFileParser | None:
    """Return a cached RobotFileParser for this domain, fetching it once."""
    with _robots_lock:
        if netloc in _robots_cache:
            return _robots_cache[netloc]

    robots_url = f"{scheme}://{netloc}/robots.txt"
    parser: RobotFileParser | None
    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            log.info("No robots.txt found for %s -- allowing", netloc)
            parser = None  # no robots.txt => nothing to disallow
        elif resp.status_code != 200:
            log.warning("Could not read robots.txt for %s (HTTP %s) -- treating as disallow-all",
                        netloc, resp.status_code)
            rp = RobotFileParser()
            rp.parse(["User-agent: *", "Disallow: /"])
            parser = rp
        else:
            rp = RobotFileParser()
            rp.parse(resp.text.splitlines())
            parser = rp
    except requests.RequestException as e:
        log.warning("Could not fetch robots.txt for %s: %s -- treating as disallow-all", netloc, e)
        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /"])
        parser = rp

    with _robots_lock:
        _robots_cache[netloc] = parser
    return parser


def is_allowed_by_robots(url: str) -> bool:
    parsed = urlparse(url)
    parser = get_robot_parser(parsed.netloc, parsed.scheme)
    if parser is None:
        return True
    return parser.can_fetch(HEADERS["User-Agent"], url)


def polite_get(url: str) -> requests.Response | None:
    if not is_allowed_by_robots(url):
        log.info("Blocked by robots.txt: %s", url)
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
        return resp
    except requests.RequestException as e:
        log.warning("Request failed for %s: %s", url, e)
        return None


# --------------------------------------------------------------------------
# Link discovery with pagination
# --------------------------------------------------------------------------
def extract_links_from_html(html: str, page_url: str, pattern: re.Pattern) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        full_url = urljoin(page_url, a["href"].split("#")[0].split("?")[0])
        if pattern.match(full_url):
            links.append(full_url)
    return list(dict.fromkeys(links))[:MAX_ARTICLES_PER_LISTING_PAGE]


def paginated_urls_for(base_listing_url: str, page: int) -> list[str]:
    base = base_listing_url.rstrip("/")
    return [tmpl.format(base=base, n=page) for tmpl in PAGINATION_TEMPLATES]


def get_article_links_paginated(listing_url: str, url_pattern: str, already_seen: set[str],
                                 target_remaining: int) -> list[str]:
    """
    Walk a listing across pages until MAX_PAGES_PER_LISTING, target_remaining
    is satisfied, or STALE_PAGE_LIMIT consecutive pages add nothing new.
    Returns the list of *new* (not in already_seen) article links found.
    """
    pattern = re.compile(url_pattern)
    collected: list[str] = []
    stale_streak = 0
    working_template = None  # once a pagination shape works for this listing, stick with it

    # Page 1 = the listing URL itself, no pagination suffix.
    resp = polite_get(listing_url)
    if resp:
        page1_links = extract_links_from_html(resp.text, listing_url, pattern)
        new_links = [u for u in page1_links if u not in already_seen and u not in collected]
        collected.extend(new_links)
        log.info("  page 1: %d links (%d new) — %s", len(page1_links), len(new_links), listing_url)

    page = 2
    while page <= MAX_PAGES_PER_LISTING and len(collected) < target_remaining and stale_streak < STALE_PAGE_LIMIT:
        candidates = [working_template.format(base=listing_url.rstrip("/"), n=page)] if working_template \
            else paginated_urls_for(listing_url, page)

        page_new_links: list[str] = []
        for candidate_url in candidates:
            resp = polite_get(candidate_url)
            if not resp:
                continue
            page_links = extract_links_from_html(resp.text, candidate_url, pattern)
            new_links = [u for u in page_links if u not in already_seen and u not in collected]
            if page_links:
                # This template returned *something* structurally matching our
                # article regex — treat it as the working shape for this listing
                # going forward, even if some pages later happen to repeat.
                if working_template is None:
                    template_index = paginated_urls_for(listing_url, page).index(candidate_url) \
                        if candidate_url in paginated_urls_for(listing_url, page) else None
                    if template_index is not None:
                        working_template = PAGINATION_TEMPLATES[template_index]
                page_new_links = new_links
                break  # found a template that returned links for this page; stop trying others

        if page_new_links:
            collected.extend(page_new_links)
            stale_streak = 0
            log.info("  page %d: %d new links — %s", page, len(page_new_links), listing_url)
        else:
            stale_streak += 1
            log.info("  page %d: no new links (stale streak %d/%d) — %s",
                      page, stale_streak, STALE_PAGE_LIMIT, listing_url)

        page += 1

    return collected


# --------------------------------------------------------------------------
# Article extraction (unchanged logic from v1)
# --------------------------------------------------------------------------
def extract_headline(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    if soup.title:
        return soup.title.get_text(strip=True)
    return ""


def extract_body_text(html: str, url: str) -> str:
    try:
        extracted = trafilatura.extract(html, url=url, include_comments=False, favor_precision=True)
        return extracted.strip() if extracted else ""
    except Exception as e:
        log.warning("trafilatura extraction failed for %s: %s", url, e)
        return ""


def isolate_claim_text(full_text: str, boundary_regex: str | None) -> str:
    if not boundary_regex:
        return full_text
    match = re.search(boundary_regex, full_text, flags=re.IGNORECASE)
    if match and match.start() > 40:
        return full_text[: match.start()].strip()
    return full_text


def code_mix_score(text: str) -> float:
    tokens = re.findall(r"[A-Za-zऀ-ஹௐ-௿]+", text)
    if not tokens:
        return 0.0
    latin_tokens = [t for t in tokens if re.match(r"^[A-Za-z]+$", t)]
    return round(len(latin_tokens) / len(tokens), 3)


def make_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Resumable state
# --------------------------------------------------------------------------
def load_seen_urls(csv_path: str) -> set[str]:
    seen = set()
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("url"):
                    seen.add(row["url"])
    except FileNotFoundError:
        pass
    if seen:
        log.info("Resuming: %d URLs already in %s will be skipped", len(seen), csv_path)
    return seen


_csv_lock = Lock()


def append_to_csv(articles: list[Article], path: str):
    if not articles:
        return
    with _csv_lock:
        try:
            with open(path, "r", encoding="utf-8"):
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(articles[0]).keys()))
            if not file_exists:
                writer.writeheader()
            for a in articles:
                writer.writerow(asdict(a))


# --------------------------------------------------------------------------
# Per-source scrape loop (target-driven)
# --------------------------------------------------------------------------
def scrape_source(source: dict, seen_urls: set[str], seen_lock: Lock, target: int,
                   output_csv: str) -> int:
    """Scrape one source up to `target` new articles. Returns count scraped."""
    scraped_count = 0
    seen_hashes: set[str] = set()

    for listing_url in source["listing_urls"]:
        if scraped_count >= target:
            break
        remaining = target - scraped_count
        log.info("=== [%s] listing: %s (need up to %d more) ===",
                  source["name"], listing_url, remaining)

        with seen_lock:
            snapshot_seen = set(seen_urls)
        candidate_links = get_article_links_paginated(
            listing_url, source["article_url_pattern"], snapshot_seen, remaining
        )

        for url in candidate_links:
            if scraped_count >= target:
                break
            with seen_lock:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            resp = polite_get(url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            headline = extract_headline(soup)
            body = extract_body_text(resp.text, url)

            if not body or len(body) < 80:
                log.info("Skipping (too short / extraction failed): %s", url)
                continue

            if source["label_type"] == "fake":
                body = isolate_claim_text(body, source.get("claim_boundary_regex"))
                if len(body) < 40:
                    log.info("Claim isolation left too little text, skipping: %s", url)
                    continue

            text_hash = hashlib.md5(re.sub(r"\s+", " ", body).strip().lower().encode()).hexdigest()
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            article = Article(
                id=make_id(url),
                source_name=source["name"],
                source_type=source["label_type"],
                url=url,
                published_date="",
                headline=headline,
                text=body,
                label=source["label_type"],
                lang_mix_score=code_mix_score(body),
                scraped_at=datetime.utcnow().isoformat(),
            )
            append_to_csv([article], output_csv)
            scraped_count += 1
            log.info("[%s] scraped %d/%d: %s", source["name"], scraped_count, target, headline[:60])

    log.info("=== [%s] done: %d articles ===", source["name"], scraped_count)
    return scraped_count


def build_per_source_targets(sources: list[dict], total_target: int) -> dict[str, int]:
    real_sources = [s for s in sources if s["label_type"] == "real"]
    fake_sources = [s for s in sources if s["label_type"] == "fake"]

    real_target = int(total_target * 0.65)
    fake_target = total_target - real_target

    targets = {}

    if real_sources:
        per_real = real_target // len(real_sources)
        for s in real_sources:
            targets[s["name"]] = per_real

    if fake_sources:
        per_fake = fake_target // len(fake_sources)
        for s in fake_sources:
            targets[s["name"]] = per_fake

    return targets


def main():
    global MAX_PAGES_PER_LISTING
    parser = argparse.ArgumentParser(description="Scrape a Tamil-English codemix fake news dataset.")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL_DEFAULT,
                        help=f"Total articles to collect across all sources (default {TARGET_TOTAL_DEFAULT})")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_LISTING,
                        help=f"Max pages to paginate per listing (default {MAX_PAGES_PER_LISTING})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of sources to scrape in parallel (default 4)")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV_DEFAULT,
                        help=f"Output CSV path (default {OUTPUT_CSV_DEFAULT})")
    args = parser.parse_args()
    MAX_PAGES_PER_LISTING = args.max_pages

    seen_urls = load_seen_urls(args.output)
    seen_lock = Lock()
    per_source_targets = build_per_source_targets(SOURCES, args.target)

    log.info("Per-source targets: %s", per_source_targets)

    total_scraped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                scrape_source, source, seen_urls, seen_lock,
                per_source_targets.get(source["name"], 0), args.output
            ): source["name"]
            for source in SOURCES
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                count = future.result()
                total_scraped += count
            except Exception as e:
                log.error("Source %s raised an exception: %s", name, e)

    log.info("Done. Scraped %d new articles this run -> %s", total_scraped, args.output)
    log.info("Total URLs on file (including prior runs): %d", len(seen_urls))


if __name__ == "__main__":
    main()