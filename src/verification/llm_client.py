"""
Generic Gemini wrapper for the verification pipeline: "send a prompt, get
text or strict JSON back, with retries and caching". Separate from
src/gemini_annotator.py, which is specific to the token-level LID
annotation prompt/schema.

Used by claim_extraction.py, query_generation.py, and llm_verifier.py so
all three share identical retry/JSON-parsing/error-handling behavior.
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils import setup_logger

import cache

logger = setup_logger("verification")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


class LLMUnavailableError(RuntimeError):
    """Raised when GEMINI_API_KEY isn't configured. Callers must handle
    this explicitly and report unavailability - never fabricate a result."""


class LLMClient:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if not api_key:
            raise LLMUnavailableError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_text(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """Returns raw model text, or None if every retry failed (rate
        limit, timeout, transient API error, etc.) - never raises for
        these expected failure modes, so callers can degrade gracefully."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": 2048},
                )
                return response.text
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning(f"Gemini call failed (attempt {attempt}/{max_retries}): {e}. Waiting {wait}s")
                time.sleep(wait)
        return None

    def generate_json(self, prompt: str, temperature: float = 0.1, max_retries: int = 3,
                       use_cache: bool = True) -> Optional[dict]:
        """Calls the model and parses a strict-JSON response, retrying on
        malformed JSON as well as on call failure. Returns None if nothing
        usable came back after all retries - callers must handle this
        (e.g. fall back to has_claim=False / verdict=UNVERIFIED), never
        invent a result to fill the gap."""
        if use_cache:
            cached = cache.get("llm_json", prompt)
            if cached is not None:
                return cached

        result = None
        for attempt in range(1, max_retries + 1):
            raw = self.generate_text(prompt, temperature=temperature, max_retries=1)
            if raw is None:
                time.sleep(min(2 ** attempt, 30))
                continue
            cleaned = _JSON_FENCE_RE.sub("", raw).strip()
            try:
                result = json.loads(cleaned)
                break
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON from LLM on attempt {attempt}: {raw[:200]!r}")

        if result is not None and use_cache:
            cache.set("llm_json", prompt, result)
        return result


if __name__ == "__main__":
    client = LLMClient()
    out = client.generate_json('Return ONLY this JSON: {"ok": true, "note": "hello"}')
    print(out)
