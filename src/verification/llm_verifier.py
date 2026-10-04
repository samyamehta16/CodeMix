"""
Module 6: LLM evidence verification - the final reasoning step, and the
ONLY module in this pipeline allowed to make a truth judgement. Modules
3-5 (search, extraction, ranking) never decide true/false; they only
retrieve and order evidence.

Given a claim and its top-ranked evidence, asks the LLM to determine
whether the evidence SUPPORTS / CONTRADICTS / is INSUFFICIENT for the
claim, then classify the claim as REAL / FAKE / MISLEADING / UNVERIFIED.

Standalone test:
    python src/verification/llm_verifier.py
"""
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent))
from llm_client import LLMClient, LLMUnavailableError
from schemas import RankedEvidence, Verdict

VALID_VERDICTS = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}
VALID_RELATIONS = {"SUPPORTS", "CONTRADICTS", "INSUFFICIENT"}

_PROMPT_TEMPLATE = """You are a careful fact-checking assistant. You will be given a CLAIM \
and a numbered list of EVIDENCE snippets retrieved from the web. Base your \
answer ONLY on the evidence provided below - never use outside knowledge, \
never invent facts, sources, or URLs that are not in the evidence list.

CLAIM:
{claim}

EVIDENCE:
{evidence_block}

Instructions:
1. Determine whether the evidence, taken together, SUPPORTS the claim, \
CONTRADICTS the claim, or is INSUFFICIENT to decide either way.
2. Based on that, classify the claim as one of:
   - REAL: evidence clearly supports the claim as stated
   - FAKE: evidence clearly contradicts the claim
   - MISLEADING: the claim has some true elements but is exaggerated, \
missing key context, or distorts what the evidence actually says
   - UNVERIFIED: evidence is insufficient or absent to decide either way \
(this is NOT the same as FAKE - lack of evidence does not mean false)
3. Cite only the evidence_ids you actually relied on.
4. Your explanation must reference specific evidence, not general knowledge.

Return ONLY valid JSON, no markdown fences, matching exactly this schema:
{{"verdict": "REAL|FAKE|MISLEADING|UNVERIFIED", "confidence": <0.0-1.0>, "evidence_relation": "SUPPORTS|CONTRADICTS|INSUFFICIENT", "explanation": "<grounded in the evidence above>", "evidence_ids": ["E1", ...]}}
"""


def _format_evidence_block(ranked: List[RankedEvidence]) -> str:
    lines = []
    for r in ranked:
        res = r.evidence.result
        lines.append(
            f"[{r.evidence_id}] {res.title} ({res.source_domain}, "
            f"{res.published_date or 'date unknown'})\n"
            f"URL: {res.url}\n"
            f"Text: {r.evidence.extracted_text[:1500]}\n"
        )
    return "\n".join(lines)


def verify_claim(claim: str, ranked_evidence: List[RankedEvidence], client: LLMClient = None) -> Verdict:
    if not ranked_evidence:
        return Verdict(
            verdict="UNVERIFIED",
            confidence=0.0,
            evidence_relation="INSUFFICIENT",
            explanation="No usable evidence was retrieved/extracted for this claim.",
            evidence_ids=[],
        )

    client = client or LLMClient()
    prompt = _PROMPT_TEMPLATE.format(claim=claim, evidence_block=_format_evidence_block(ranked_evidence))

    data = client.generate_json(prompt)
    if data is None:
        return Verdict(
            verdict="UNVERIFIED",
            confidence=0.0,
            evidence_relation="INSUFFICIENT",
            explanation="LLM verification call failed or returned malformed JSON after retries.",
            evidence_ids=[],
            error="llm_call_failed_or_malformed_json",
        )

    verdict = str(data.get("verdict", "")).upper()
    relation = str(data.get("evidence_relation", "")).upper()
    if verdict not in VALID_VERDICTS or relation not in VALID_RELATIONS:
        return Verdict(
            verdict="UNVERIFIED",
            confidence=0.0,
            evidence_relation="INSUFFICIENT",
            explanation=f"LLM returned an invalid verdict/relation enum: {data!r}",
            evidence_ids=[],
            raw_llm_response=data,
            error="invalid_enum_value",
        )

    valid_ids = {r.evidence_id for r in ranked_evidence}
    cited_ids = [eid for eid in data.get("evidence_ids", []) if eid in valid_ids]

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Verdict(
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        evidence_relation=relation,
        explanation=data.get("explanation", ""),
        evidence_ids=cited_ids,
        raw_llm_response=data,
    )


if __name__ == "__main__":
    from schemas import Evidence, SearchResult
    from evidence_ranking import rank_evidence

    claim = "RBI has released a new ₹5000 currency note."
    dummy_evidence = [
        Evidence(result=SearchResult(title="RBI clarification", url="https://rbi.org.in/x",
                                      snippet="", source_domain="rbi.org.in", published_date="2026-07-01"),
                 extracted_text="The Reserve Bank of India has clarified that no new 5000 rupee note "
                                "has been approved or issued, calling such claims false rumors "
                                "circulating on social media.",
                 extraction_success=True),
    ]
    ranked = rank_evidence(claim, dummy_evidence)

    try:
        verdict = verify_claim(claim, ranked)
    except LLMUnavailableError as e:
        print(f"[UNAVAILABLE] {e}")
    else:
        print(verdict)
