"""
Module 2: Search query generation.

Given a normalized claim, asks the LLM for several short, diverse web
search queries likely to surface relevant evidence - different phrasings
and keyword combinations, rather than just re-using the claim sentence
verbatim as the only query (which tends to retrieve poorly on search
engines built for keyword queries, not full sentences).

Standalone test:
    python src/verification/query_generation.py
    python src/verification/query_generation.py "RBI has released a new ₹5000 currency note."
"""
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent))
from llm_client import LLMClient, LLMUnavailableError

_PROMPT_TEMPLATE = """Given the following factual claim, generate {n} short, diverse web \
search engine queries (3-8 words each) that would help find news articles, \
official statements, or fact-checks about it. Vary the phrasing and \
keyword choice across queries rather than producing near-duplicates.

CLAIM: {claim}

Return ONLY valid JSON, no markdown fences, matching exactly this schema:
{{"queries": ["<query 1>", "<query 2>", ...]}}
"""


def generate_queries(claim: str, client: LLMClient = None, n: int = 3) -> List[str]:
    client = client or LLMClient()
    prompt = _PROMPT_TEMPLATE.format(claim=claim, n=n)

    data = client.generate_json(prompt)
    if data is None or not data.get("queries"):
        # Graceful fallback so the pipeline can still proceed with at least
        # one query instead of dead-ending on an LLM hiccup.
        return [claim]

    queries = [q.strip() for q in data["queries"] if isinstance(q, str) and q.strip()]
    return queries[:n] if queries else [claim]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("claim", nargs="?", default="RBI has released a new ₹5000 currency note.")
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    try:
        queries = generate_queries(args.claim, n=args.n)
    except LLMUnavailableError as e:
        print(f"[UNAVAILABLE] {e}")
    else:
        for q in queries:
            print(f"- {q}")
