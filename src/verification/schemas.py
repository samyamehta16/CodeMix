"""
Shared data structures passed between the verification pipeline modules.

Plain dataclasses on purpose (no ORM/pydantic) - keeps every module
independently readable/testable for a student research project, and
`dataclasses.asdict()` gives free JSON-serialization for caching/logging.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedClaim:
    """Output of claim_extraction.py - detection + normalization only,
    never a truth judgement."""
    original_text: str
    has_claim: bool
    normalized_claim: Optional[str] = None
    language_note: Optional[str] = None
    raw_llm_response: Optional[dict] = None


@dataclass
class SearchResult:
    """One organic result from a web search provider (module 3)."""
    title: str
    url: str
    snippet: str
    source_domain: str
    published_date: Optional[str] = None
    rank: Optional[int] = None


@dataclass
class Evidence:
    """A SearchResult plus the full page text extracted from its URL
    (module 4). `extraction_success=False` means don't trust
    `extracted_text` - it may be None or stale."""
    result: SearchResult
    extracted_text: Optional[str] = None
    extraction_success: bool = False
    extraction_error: Optional[str] = None


@dataclass
class RankedEvidence:
    """Evidence + its ranking scores (module 5). `evidence_id` (E1, E2, ...)
    is what the LLM verifier cites back in its JSON output."""
    evidence: Evidence
    evidence_id: str
    relevance_score: float
    authority_score: float
    freshness_score: float
    final_score: float


@dataclass
class Verdict:
    """Output of llm_verifier.py (module 6) - the ONLY place a truth
    judgement is made in the whole pipeline."""
    verdict: str  # REAL | FAKE | MISLEADING | UNVERIFIED
    confidence: float
    evidence_relation: str  # SUPPORTS | CONTRADICTS | INSUFFICIENT
    explanation: str
    evidence_ids: List[str] = field(default_factory=list)
    raw_llm_response: Optional[dict] = None
    error: Optional[str] = None
