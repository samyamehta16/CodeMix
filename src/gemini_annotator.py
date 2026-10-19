"""
Wraps the Gemini API call for one comment -> token-level language labels.

Why an LLM here instead of a fine-tuned tagger: we don't have a Tamil-English
LID dataset or GPU time to fine-tune IndicBERT/MuRIL from scratch. Gemini is
used as a *silver-label generator* - its output becomes the training data for
a future MuRIL/IndicBERT fine-tune (see convert_to_lid_format.py). This is a
documented, defensible workaround - not a replacement for eventually training
a real tagger, and a human-reviewed sample (review_sample.py) should be
reported alongside it to show label quality.
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

sys.path.append(str(Path(__file__).resolve().parent))
from config import GEMINI_API_KEY, GEMINI_MODEL, LABELS
from utils import setup_logger

logger = setup_logger()

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "annotation_prompt.txt"
_PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


class GeminiAnnotator:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, comment_text: str) -> str:
        return _PROMPT_TEMPLATE.replace("{COMMENT_TEXT}", comment_text)

    def _parse_response(self, raw_text: str) -> Optional[list]:
        cleaned = _JSON_FENCE_RE.sub("", raw_text).strip()
        try:
            data = json.loads(cleaned)
            tokens = data.get("tokens", [])
            # basic validation - drop tokens with bad/missing labels
            valid = [t for t in tokens
                     if isinstance(t, dict) and t.get("label") in LABELS and "token" in t]
            return valid
        except (json.JSONDecodeError, AttributeError):
            return None

    def annotate(self, comment_text: str, max_retries: int = 3) -> Optional[list]:
        """
        Returns a list of {"token": ..., "label": ...} dicts, or None if it
        failed after retries (caller should log and skip, not crash the run).
        """
        prompt = self._build_prompt(comment_text)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                tokens = self._parse_response(response.text)
                if tokens:
                    return tokens
                logger.warning(f"Unparseable/empty response on attempt {attempt}, retrying...")
            except Exception as e:
                # covers rate-limit errors, transient API errors, etc.
                wait = min(2 ** attempt, 30)
                logger.warning(f"Gemini call failed (attempt {attempt}/{max_retries}): {e}. "
                                f"Waiting {wait}s")
                time.sleep(wait)
                continue

        logger.error(f"Giving up on comment after {max_retries} attempts: {comment_text[:60]!r}")
        return None
