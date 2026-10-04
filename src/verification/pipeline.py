"""
Orchestrates the full evidence-aware verification pipeline:

  comment -> clean -> claim_extraction -> query_generation -> web_search
  -> webpage_extraction -> evidence_ranking -> llm_verifier -> verdict

Every stage's raw output is kept on the returned PipelineResult (not just
the final verdict), so run_demo.py can show each intermediate step -
important for a research project where you need to show your work, not
just the final label.

The pipeline stops early and reports exactly why whenever a stage can't
proceed (LLM not configured, no claim detected, search not configured, no
search results) rather than guessing or fabricating downstream stages.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent))
from claim_extraction import extract_claim
from evidence_ranking import rank_evidence
from llm_client import LLMClient, LLMUnavailableError
from llm_verifier import verify_claim
from query_generation import generate_queries
from schemas import Evidence, ExtractedClaim, RankedEvidence, SearchResult, Verdict
from web_search import SearchProviderError, get_search_provider
from webpage_extraction import extract_from_url

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import clean_comment, setup_logger

logger = setup_logger("verification")


@dataclass
class PipelineResult:
    original_text: str
    cleaned_text: str
    claim: ExtractedClaim
    queries: List[str]
    search_results: List[SearchResult]
    evidence: List[Evidence]
    ranked_evidence: List[RankedEvidence]
    verdict: Optional[Verdict]
    stage_reached: str  # see STAGE_* constants below
    notes: List[str]


STAGE_LLM_UNAVAILABLE = "llm_unavailable"
STAGE_NO_CLAIM = "no_claim_detected"
STAGE_SEARCH_UNAVAILABLE = "search_unavailable"
STAGE_NO_RESULTS = "no_search_results"
STAGE_COMPLETE = "complete"


def run_pipeline(raw_text: str, num_queries: int = 3, results_per_query: int = 4,
                  top_k_evidence: int = 5) -> PipelineResult:
    notes: List[str] = []
    cleaned = clean_comment(raw_text, keep_emoji=False)

    def _stub(claim=None, queries=None, results=None, evidence=None, ranked=None, verdict=None, stage=""):
        return PipelineResult(
            original_text=raw_text, cleaned_text=cleaned,
            claim=claim or ExtractedClaim(cleaned, False),
            queries=queries or [], search_results=results or [],
            evidence=evidence or [], ranked_evidence=ranked or [],
            verdict=verdict, stage_reached=stage, notes=notes,
        )

    # Stage 0: LLM must be configured for stages 1, 2, and 6
    try:
        client = LLMClient()
    except LLMUnavailableError as e:
        notes.append(str(e))
        return _stub(stage=STAGE_LLM_UNAVAILABLE)

    # Stage 1: claim extraction
    claim = extract_claim(cleaned, client=client)
    if not claim.has_claim or not claim.normalized_claim:
        notes.append("No checkable factual claim detected in this comment.")
        return _stub(claim=claim, stage=STAGE_NO_CLAIM)

    # Stage 2: query generation
    queries = generate_queries(claim.normalized_claim, client=client, n=num_queries)

    # Stage 3: web search
    provider = get_search_provider()
    all_results: List[SearchResult] = []
    try:
        for q in queries:
            all_results.extend(provider.search(q, num_results=results_per_query))
    except SearchProviderError as e:
        notes.append(f"Search unavailable: {e}")
        return _stub(claim=claim, queries=queries, stage=STAGE_SEARCH_UNAVAILABLE)

    if not all_results:
        notes.append("Web search returned no results for any generated query.")
        return _stub(claim=claim, queries=queries, stage=STAGE_NO_RESULTS)

    # de-dupe by URL (different queries often surface the same page)
    seen = set()
    deduped = []
    for r in all_results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            deduped.append(r)

    # Stage 4: webpage extraction
    evidence = [extract_from_url(r) for r in deduped]
    n_ok = sum(e.extraction_success for e in evidence)
    if n_ok == 0:
        notes.append("Found search results, but no page's text could be extracted "
                     "(paywalls, timeouts, or unsupported layouts).")

    # Stage 5: ranking
    ranked = rank_evidence(claim.normalized_claim, evidence)[:top_k_evidence]

    # Stage 6: LLM verification (verify_claim itself handles the "no ranked evidence" case)
    verdict = verify_claim(claim.normalized_claim, ranked, client=client)

    return _stub(claim=claim, queries=queries, results=deduped, evidence=evidence,
                 ranked=ranked, verdict=verdict, stage=STAGE_COMPLETE)


if __name__ == "__main__":
    result = run_pipeline("RBI 5000 rupees note release pannitanga nu news la varudhu")
    print(f"stage_reached: {result.stage_reached}")
    print(f"claim: {result.claim.normalized_claim}")
    if result.verdict:
        print(f"verdict: {result.verdict.verdict} ({result.verdict.confidence})")
    print(f"notes: {result.notes}")
