"""
Module 1: Claim extraction.

Given a raw (possibly Tamil-English code-mixed) social media comment,
asks the LLM:
  1. Does this comment contain a checkable factual/news claim?
  2. If yes, normalize it into ONE clear, self-contained English
     statement suitable for web search and evidence comparison.

This module only detects + rephrases - it never judges truth. That
happens later, in llm_verifier.py, after evidence has been retrieved.

Standalone test:
    python src/verification/claim_extraction.py
    python src/verification/claim_extraction.py "your comment here"
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from llm_client import LLMClient, LLMUnavailableError
from schemas import ExtractedClaim

_PROMPT_TEMPLATE = """You are analyzing a social media comment that may be written in \
Tamil-English code-mixed text (Tamil script, romanized Tamil, English, or a mix of both).

Your job has two parts:
1. Decide whether the comment makes or repeats a FACTUAL / NEWS-STYLE CLAIM \
that could in principle be checked against evidence (an announcement, \
statistic, event, policy, health claim, quote attributed to someone, etc). \
Opinions, jokes, greetings, song lyrics, insults, and pure emotional \
reactions are NOT claims - mark them has_claim: false.
2. If it IS a claim, rewrite it as ONE clear, self-contained, neutral \
English sentence stating exactly what is being claimed (the "normalized \
claim"). Do not add information that isn't in the original comment. Do \
NOT resolve whether it's true - only restate what is being claimed.

Return ONLY valid JSON, no markdown fences, matching exactly this schema:
{{"has_claim": true or false, "normalized_claim": "<string, or null if has_claim is false>", "reasoning": "<one short sentence explaining your decision>"}}

Comment:
\"\"\"
{comment_text}
\"\"\"
"""


def extract_claim(comment_text: str, client: LLMClient = None) -> ExtractedClaim:
    client = client or LLMClient()
    prompt = _PROMPT_TEMPLATE.format(comment_text=comment_text)

    data = client.generate_json(prompt)
    if data is None:
        return ExtractedClaim(
            original_text=comment_text,
            has_claim=False,
            normalized_claim=None,
            language_note="LLM call failed after retries - could not determine claim status "
                           "(treated as no-claim rather than guessing)",
        )

    return ExtractedClaim(
        original_text=comment_text,
        has_claim=bool(data.get("has_claim", False)),
        normalized_claim=data.get("normalized_claim"),
        language_note=data.get("reasoning"),
        raw_llm_response=data,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("comment", nargs="?",
                    default="RBI 5000 rupees note release pannitanga nu news la varudhu")
    args = p.parse_args()

    try:
        result = extract_claim(args.comment)
    except LLMUnavailableError as e:
        print(f"[UNAVAILABLE] {e}")
    else:
        print(f"original_text:    {result.original_text}")
        print(f"has_claim:        {result.has_claim}")
        print(f"normalized_claim: {result.normalized_claim}")
        print(f"reasoning:        {result.language_note}")
