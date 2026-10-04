"""
Small helpers for text cleaning and logging.
"""
import re
import logging

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
# collapse 3+ repeated LETTERS -> 2 (soooo -> soo) to reduce noise, keeps meaning.
# Explicitly excludes digits/underscore ([^\W\d_]) so numbers like "5000" or
# "1000" (which have 3+ repeated digits) are never mangled into "500"/"100" -
# this matters a lot once numbers are part of claims being fact-checked.
REPEAT_CHAR_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)


def setup_logger(name: str = "annotator") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def clean_comment(text: str, keep_emoji: bool = True) -> str:
    """
    Cleans a single YouTube comment while preserving Tamil-English
    code-mixed content. Does NOT lowercase (case can carry meaning/tone
    in social text) and does NOT strip Tamil unicode.
    """
    if not isinstance(text, str):
        return ""

    text = URL_RE.sub(" ", text)                  # remove URLs
    text = text.replace("\n", " ").replace("\r", " ")
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)       # soooo -> soo
    text = WHITESPACE_RE.sub(" ", text).strip()    # collapse whitespace

    if not keep_emoji:
        import emoji
        text = emoji.replace_emoji(text, replace="")
        text = WHITESPACE_RE.sub(" ", text).strip()

    return text


# Zero-width joiner / variation selectors - these glue together or modify
# emoji (e.g. "heart" + ZWJ + "fire" -> a single combined emoji glyph).
# `emoji.replace_emoji` can fail to match a *dangling* base emoji when its
# ZWJ/variation-selector partner ended up split into a separate token by
# the LLM's tokenization, so strip these connector chars first.
_ZWJ_VARIATION_RE = re.compile(r"[\u200d\ufe0e\ufe0f]")


def strip_emoji(token: str) -> str:
    """Removes emoji characters from a token; returns '' if nothing is left
    (i.e. the token was emoji-only)."""
    token = _ZWJ_VARIATION_RE.sub("", token)
    return emoji.replace_emoji(token, replace="").strip()


def is_low_quality(text: str, min_len: int = 2) -> bool:
    """Flags empty / near-empty / punctuation-only comments for removal."""
    if not text or len(text.strip()) < min_len:
        return True
    if not re.search(r"[A-Za-z\u0B80-\u0BFF]", text):  # no Latin or Tamil chars
        return True
    return False
