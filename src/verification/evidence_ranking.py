"""
Module 5: Evidence ranking.

Scores and sorts successfully-extracted evidence (module 4's output) by:
  1. relevance to the claim   - TF-IDF cosine similarity (default) or BM25
  2. source authority         - curated trusted-domain list
  3. freshness                - how recent the publication date is, if known

This module does NOT judge truth/falsity - it only orders evidence so the
LLM verifier (module 6) sees the most useful items first. Evidence that
failed extraction is dropped here (never passed on as if it were real
content).

Standalone test:
    python src/verification/evidence_ranking.py
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from schemas import Evidence, RankedEvidence
from utils import setup_logger

logger = setup_logger("verification")

# Not exhaustive - a starting point matching the project brief's "prefer
# authoritative sources" list. Extend freely as you find more sources
# relevant to your claim domains.
TRUSTED_DOMAINS = {
    # Government / official India
    "rbi.org.in": 1.0, "pib.gov.in": 1.0, "eci.gov.in": 1.0, "india.gov.in": 1.0,
    "mohfw.gov.in": 1.0, "sci.gov.in": 1.0, "meity.gov.in": 1.0,
    # International orgs
    "who.int": 1.0, "un.org": 1.0,
    # Fact-checking organizations
    "altnews.in": 0.95, "boomlive.in": 0.95, "factcheck.afp.com": 0.95,
    "vishvasnews.com": 0.9, "factchecker.in": 0.9,
    # Reputable news organizations
    "bbc.com": 0.9, "reuters.com": 0.9, "apnews.com": 0.9,
    "thehindu.com": 0.85, "ndtv.com": 0.85, "indianexpress.com": 0.85,
    "livemint.com": 0.8, "hindustantimes.com": 0.8,
    "timesofindia.indiatimes.com": 0.75,
}
DEFAULT_AUTHORITY = 0.4  # unknown domains: neutral-low, not zeroed out


def authority_score(domain: str) -> float:
    domain = (domain or "").lower().replace("www.", "")
    return TRUSTED_DOMAINS.get(domain, DEFAULT_AUTHORITY)


def freshness_score(published_date: Optional[str]) -> float:
    """Simple recency decay. Missing/unparseable dates get a neutral 0.5 -
    a page shouldn't be penalized as "old" just because we couldn't parse
    its date."""
    if not published_date:
        return 0.5
    parsed = _try_parse_date(published_date)
    if parsed is None:
        return 0.5

    age_days = (datetime.now(timezone.utc) - parsed).days
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.85
    if age_days <= 180:
        return 0.65
    if age_days <= 730:
        return 0.45
    return 0.3


def _try_parse_date(date_str: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d",
                "%B %d, %Y", "%d %B %Y", "%b %d, %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\u0B80-\u0BFF]+", text.lower())


def relevance_scores(claim: str, texts: List[str], method: str = "tfidf") -> List[float]:
    """Returns a relevance score in [0, 1] per text, comparing each
    against `claim`. Degrades gracefully if optional ML deps aren't
    installed, so this module works even with zero extra dependencies."""
    if not texts:
        return []

    if method == "bm25":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed, falling back to TF-IDF")
            method = "tfidf"
        else:
            tokenized_corpus = [_tokenize(t) for t in texts]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(_tokenize(claim))
            max_score = max(scores) if max(scores) > 0 else 1.0
            return [s / max_score for s in scores]

    if method == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("scikit-learn not installed, falling back to keyword overlap")
        else:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform([claim] + texts)
            sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            return [float(s) for s in sims]

    # naive last-resort fallback: fraction of claim keywords present in each text
    claim_tokens = set(_tokenize(claim))
    scores = []
    for t in texts:
        text_tokens = set(_tokenize(t))
        overlap = len(claim_tokens & text_tokens) / max(len(claim_tokens), 1)
        scores.append(overlap)
    return scores


def rank_evidence(
    claim: str,
    evidence_list: List[Evidence],
    weights: tuple = (0.5, 0.3, 0.2),  # (relevance, authority, freshness)
    method: str = "tfidf",
) -> List[RankedEvidence]:
    usable = [e for e in evidence_list if e.extraction_success and e.extracted_text]
    if not usable:
        return []

    texts = [e.extracted_text for e in usable]
    rel_scores = relevance_scores(claim, texts, method=method)

    w_rel, w_auth, w_fresh = weights
    ranked = []
    for i, ev in enumerate(usable):
        auth = authority_score(ev.result.source_domain)
        fresh = freshness_score(ev.result.published_date)
        final = w_rel * rel_scores[i] + w_auth * auth + w_fresh * fresh
        ranked.append(RankedEvidence(
            evidence=ev,
            evidence_id=f"E{i + 1}",  # temporary, re-assigned below after sort
            relevance_score=round(float(rel_scores[i]), 4),
            authority_score=round(auth, 4),
            freshness_score=round(fresh, 4),
            final_score=round(final, 4),
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    for i, r in enumerate(ranked):
        r.evidence_id = f"E{i + 1}"  # E1 = best evidence after sorting
    return ranked


if __name__ == "__main__":
    from schemas import SearchResult

    claim = "RBI has released a new ₹5000 currency note."
    dummy_evidence = [
        Evidence(result=SearchResult(title="RBI press release", url="https://rbi.org.in/x",
                                      snippet="", source_domain="rbi.org.in", published_date="2026-07-01"),
                 extracted_text="The Reserve Bank of India has not issued any new 5000 rupee note "
                                "and urges the public to ignore such rumors.",
                 extraction_success=True),
        Evidence(result=SearchResult(title="Random old blog", url="https://randomblog.com/x",
                                      snippet="", source_domain="randomblog.com", published_date="2019-01-01"),
                 extracted_text="A brief unrelated history of Indian currency design over the decades.",
                 extraction_success=True),
    ]
    for r in rank_evidence(claim, dummy_evidence):
        print(f"{r.evidence_id}  final={r.final_score}  "
              f"(rel={r.relevance_score}, auth={r.authority_score}, fresh={r.freshness_score})  "
              f"{r.evidence.result.source_domain}")
