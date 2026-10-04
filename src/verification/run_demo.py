"""
Interactive test script for the evidence-aware verification pipeline.
Runs ONE comment/claim through every stage and prints each stage's output
- exactly the "type a claim, see every step" test harness requested for
this project.

Run:
    python src/verification/run_demo.py
    python src/verification/run_demo.py --text "RBI has released a new ₹5000 currency note."
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from pipeline import (
    STAGE_LLM_UNAVAILABLE, STAGE_NO_CLAIM, STAGE_NO_RESULTS,
    STAGE_SEARCH_UNAVAILABLE, run_pipeline,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="Comment/claim to test")
    args = p.parse_args()

    text = args.text
    if not text:
        text = input("Enter a comment or claim to verify: ").strip()
    if not text:
        text = "RBI has released a new ₹5000 currency note."
        print(f"(no input given - using default example: {text!r})")

    print("=" * 72)
    print(f"INPUT: {text}")
    print("=" * 72)

    result = run_pipeline(text)

    print(f"\n[1] CLEANED TEXT\n    {result.cleaned_text}")

    print(f"\n[2] CLAIM EXTRACTION")
    print(f"    has_claim:        {result.claim.has_claim}")
    print(f"    normalized_claim: {result.claim.normalized_claim}")
    print(f"    reasoning:        {result.claim.language_note}")

    if result.stage_reached in (STAGE_LLM_UNAVAILABLE, STAGE_NO_CLAIM):
        _print_stop(result)
        return

    print(f"\n[3] GENERATED SEARCH QUERIES")
    for q in result.queries:
        print(f"    - {q}")

    if result.stage_reached in (STAGE_SEARCH_UNAVAILABLE, STAGE_NO_RESULTS):
        _print_stop(result)
        return

    print(f"\n[4] RETRIEVED URLs ({len(result.search_results)})")
    for r in result.search_results:
        print(f"    - [{r.source_domain}] {r.title}\n      {r.url}")

    n_ok = sum(e.extraction_success for e in result.evidence)
    print(f"\n[5] EXTRACTED EVIDENCE ({n_ok}/{len(result.evidence)} pages extracted successfully)")
    for e in result.evidence:
        status = "OK" if e.extraction_success else f"FAILED ({e.extraction_error})"
        print(f"    - {e.result.url} -> {status}")

    print(f"\n[6] RANKED EVIDENCE (top {len(result.ranked_evidence)})")
    for r in result.ranked_evidence:
        print(f"    {r.evidence_id}  final={r.final_score}  "
              f"(relevance={r.relevance_score}, authority={r.authority_score}, freshness={r.freshness_score})")
        print(f"       {r.evidence.result.title} ({r.evidence.result.source_domain})")

    print(f"\n[7] FINAL VERDICT")
    v = result.verdict
    if v:
        print(f"    verdict:           {v.verdict}")
        print(f"    confidence:        {v.confidence}")
        print(f"    evidence_relation: {v.evidence_relation}")
        print(f"    evidence_ids:      {v.evidence_ids}")
        print(f"    explanation:       {v.explanation}")
        if v.error:
            print(f"    error:             {v.error}")
    else:
        print("    No verdict produced.")

    if result.notes:
        print(f"\nNOTES:")
        for n in result.notes:
            print(f"  - {n}")


def _print_stop(result):
    print(f"\nSTOPPED at stage: {result.stage_reached}")
    for n in result.notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
