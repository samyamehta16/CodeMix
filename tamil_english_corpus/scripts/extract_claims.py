"""
extract_claims.py
=================
Conservative claim-text extraction for the Tamil-English fake/real news dataset.

Column written: claim_text
Column written: extraction_status
  Values:
    extracted_from_ithil     - claim pulled from after ithil block
    extracted_leading_quote  - claim from the leading title quote before intro verb
    retained_direct          - real rows or rows with no wrapper -> text retained as-is
    manual_review            - could not safely isolate the claim

Rules (fake rows only):
  STRUCTURE A  ''TITLE'' intro_verb, \n...ithil, ''ACTUAL_CLAIM'' ...
               -> Extract ACTUAL_CLAIM from the ithil block
  STRUCTURE B  Direct claim text, no leading quote wrapper, no ithil
               -> Retain text as-is (retained_direct)
  STRUCTURE C  Starts with intro prose, then ithil, ''CLAIM''
               -> Extract quoted passage after ithil as claim
  STRUCTURE D  Social media intro -> try ithil extraction, else manual_review

Real rows:
  -> claim_text = text (retained_direct), no modification
"""

import sys, io, re, pandas as pd

# ---------------------------------------------------------------------------
# Unicode quote characters actually present in the dataset
# ---------------------------------------------------------------------------
# U+2018 LEFT SINGLE QUOTATION MARK  (open curly single:  \u2018)
# U+2019 RIGHT SINGLE QUOTATION MARK (close curly single: \u2019)
# U+201C LEFT DOUBLE QUOTATION MARK  (open curly double:  \u201c)
# U+201D RIGHT DOUBLE QUOTATION MARK (close curly double: \u201d)
#
# The dataset uses a compound pair U+2018 + U+2019 placed side-by-side as
# both the open-delimiter AND the close-delimiter (mimicking '' and '').
# We must match this two-char compound FIRST, before the single-char patterns,
# so the regex doesn't split it into an open + an immediate close.

_CQ_OPEN_SGL  = "\u2018"   # ' (left single)
_CQ_CLOSE_SGL = "\u2019"   # ' (right single)
_CQ_OPEN_DBL  = "\u201c"   # " (left double)
_CQ_CLOSE_DBL = "\u201d"   # " (right double)
_COMPOUND_SGL = "\u2018\u2019"   # '' used as compound open/close

# Regex alternation: try compound pair first, then two straight singles,
# then curly doubles, then straight double.
_QOPEN  = (
    r"(?:"
    + re.escape(_COMPOUND_SGL)      # '' (compound, highest priority)
    + r"|'{2}"                      # '' (two straight singles)
    + r"|" + re.escape(_CQ_OPEN_DBL)  # "
    + r"|" + re.escape(_CQ_OPEN_SGL)  # '
    + r"|\")"                        # " (straight double)
)
_QCLOSE = (
    r"(?:"
    + re.escape(_COMPOUND_SGL)
    + r"|'{2}"
    + r"|" + re.escape(_CQ_CLOSE_DBL)
    + r"|" + re.escape(_CQ_CLOSE_SGL)
    + r"|\")"
)

_QUOTE_BLOCK = re.compile(_QOPEN + r"(.+?)" + _QCLOSE, re.DOTALL)

# ---------------------------------------------------------------------------
# Other patterns
# ---------------------------------------------------------------------------

# After ithil we may see spaces or \n before the quote
_ITHIL_RE = re.compile(r"\u0b87\u0ba4\u0bbf\u0bb2\u0bcd,?\s*")   # இதில்,?

# Trailing fact-check noise appended after the embedded claim
_TRAILING_NOISE = re.compile(
    r"\s*(?:"
    r"\u0b8e\u0ba9\u0bcd\u0bb1\u0bc1\s+\u0b8e\u0bb4\u0bc1\u0ba4\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1"  # என்று எழுதப்பட்டுள்ளது
    r"|\u0b8e\u0ba9\u0bcd\u0bb1\u0bc1\s+\u0b8e\u0bb4\u0bc1\u0ba4\u0bbf\s+\u0b87\u0ba3\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1"  # என்று எழுதி இணைக்கப்பட்டுள்ளது
    r"|\u0baa\u0bb2\u0bb0\u0bc1\u0bae\u0bcd\s+\u0b87\u0ba4\u0ba9\u0bc8\s+\u0b89\u0ba3\u0bcd\u0bae\u0bc8"  # பலரும் இதனை உண்மை
    r"|\u0b9a\u0bae\u0bc2\u0b95\s+\u0bb5\u0bb2\u0bc8\u0ba4\u0bb3\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd"  # சமூக வலைதளங்களில்
    r"|\u0b9a\u0bae\u0bc2\u0b95\s+\u0b8a\u0b9f\u0b95\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd"  # சமூக ஊடகங்களில்
    r")",
    re.DOTALL
)

# Verbs that signal boilerplate after a leading title quote
_INTRO_VERBS_PAT = (
    r"\u0b95\u0bc2\u0bb1\u0bbf"  # கூறி
    r"|\u0baa\u0bb0\u0bb5\u0bc1\u0bae\u0bcd"  # பரவும்
    r"|\u0baa\u0bb0\u0baa\u0bcd\u0baa\u0bbf"  # பரப்பி
    r"|\u0bb5\u0bc8\u0bb0\u0bb2\u0bbe\u0b95\u0bbf"  # வைரலாகி
    r"|\u0bb5\u0bc6\u0bb3\u0bbf\u0baf\u0bbf\u0b9f\u0bcd\u0b9f\u0ba4\u0bbe\u0b95"  # வெளியிட்டதாக
    r"|\u0ba4\u0bbf\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf"  # திரிக்கி
    r"|\u0b95\u0bc2\u0bb1\u0bbf\u0baf\u0ba4\u0bbe\u0b95"  # கூறியதாக
    r"|\u0bb5\u0ba8\u0bcd\u0ba4\u0ba4\u0bc1"  # வந்தது
    r"|\u0baa\u0bb0\u0baa\u0bcd\u0baa\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1"  # பரப்பப்படுகிறது
    r"|\u0bb5\u0bc8\u0bb0\u0bb2\u0bcd"  # வைரல்
    r"|\u0b8e\u0ba9\u0bcd\u0bb1\u0bc1"  # என்று
)

_LEADING_TITLE_RE = re.compile(
    r"^" + _QOPEN + r"(.+?)" + _QCLOSE + r"\s+\u0b8e\u0ba9\u0bcd\u0bb1\u0bc1\s+(?:" + _INTRO_VERBS_PAT + r")",
    re.DOTALL
)

# Social-media intro check (first 200 chars)
_SOCIAL_INTRO_RE = re.compile(
    r"\u0b9a\u0bae\u0bc2\u0b95\s+(?:\u0bb5\u0bb2\u0bc8\u0ba4\u0bb3\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd|\u0b8a\u0b9f\u0b95\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd)"  # சமூக வலைதளங்களில் / சமூக ஊடகங்களில்
    r"|\u0b83\u0baa\u0bc7\u0b9a\u0bcd\u0baa\u0bc1\u0b95\u0bcd\u0b95\u0bbf\u0bb2\u0bcd\s+\u0baa\u0ba4\u0bbf\u0bb5\u0bbf\u0b9f\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1"  # ஃபேஸ்புக்கில் பதிவிடப்பட்டுள்ளது
)


def _strip_quote(s: str) -> str:
    """Remove wrapping and orphan leading/trailing quote chars."""
    s = s.strip()
    # Full-pair strip (compound and non-compound)
    for pair in [
        (_COMPOUND_SGL, _COMPOUND_SGL),
        ("''", "''"),
        ('"', '"'),
        (_CQ_OPEN_DBL, _CQ_CLOSE_DBL),
        (_CQ_OPEN_SGL, _CQ_CLOSE_SGL),
    ]:
        if s.startswith(pair[0]) and s.endswith(pair[1]):
            return s[len(pair[0]):-len(pair[1])].strip()
    # Partial strip: opening only
    for q in [_COMPOUND_SGL, "''", '"', _CQ_OPEN_DBL, _CQ_OPEN_SGL]:
        if s.startswith(q):
            s = s[len(q):].strip()
            break
    # Partial strip: trailing only
    for q in [_COMPOUND_SGL, "''", '"', _CQ_CLOSE_DBL, _CQ_CLOSE_SGL]:
        if s.endswith(q):
            s = s[:-len(q)].strip()
            break
    return s


def _clean_claim(text: str) -> str:
    """Strip trailing fact-check noise, leading markdown, and normalise whitespace."""
    text = _TRAILING_NOISE.sub("", text).strip()
    text = _strip_quote(text)
    # Remove leading markdown bold/italic asterisks from social-media copy-pastes
    text = re.sub(r"^[\*_\s]+", "", text).strip()
    # Collapse internal whitespace runs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_claim(row) -> tuple[str, str]:
    """Return (claim_text, extraction_status) for one row."""
    text = str(row["text"]).strip()

    if not text or text == "nan":
        return "", "manual_review"

    label = int(row.get("label", -1))

    # ── REAL ROWS ─────────────────────────────────────────────────────────
    if label == 1:
        return text, "retained_direct"

    # ── FAKE ROWS ─────────────────────────────────────────────────────────

    # Step 1: Locate ithil block
    ithil_match = _ITHIL_RE.search(text)

    if ithil_match:
        after_ithil = text[ithil_match.end():]
        qm = _QUOTE_BLOCK.search(after_ithil)
        if qm:
            candidate = _clean_claim(qm.group(1))
            if len(candidate) >= 10:
                return candidate, "extracted_from_ithil"
        # No quote found after ithil -> try plain text after ithil
        plain_after = _TRAILING_NOISE.sub("", after_ithil).strip()
        plain_after = re.sub(r"\s+", " ", plain_after).strip()
        if len(plain_after) >= 10:
            return plain_after, "extracted_from_ithil"

    # Step 2: Try leading title quote as claim (only when ithil not found)
    if _LEADING_TITLE_RE.match(text):
        qm = _QUOTE_BLOCK.match(text)
        if qm:
            candidate = _clean_claim(qm.group(1))
            if len(candidate) >= 10:
                return candidate, "extracted_leading_quote"

    # Step 3: No wrapper -> treat as direct claim
    if not _SOCIAL_INTRO_RE.search(text[:200]):
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned, "retained_direct"

    # Step 4: Could not safely isolate claim
    return text, "manual_review"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    src = "data/final/tamil_news_dataset_1400_balanced.csv"
    out = "data/final/tamil_news_dataset_1400_balanced.csv"

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from {src}")

    # Preserve original text column
    if "text_original" not in df.columns:
        df["text_original"] = df["text"]

    # Apply extraction
    results = df.apply(extract_claim, axis=1)
    df["claim_text"]        = [r[0] for r in results]
    df["extraction_status"] = [r[1] for r in results]

    print("\n=== Extraction Status Breakdown ===")
    print(df["extraction_status"].value_counts())

    manual = df[df["extraction_status"] == "manual_review"]
    print(f"\nManual review count: {len(manual)}")

    df["_claim_len"] = df["claim_text"].str.len()
    print(f"\nclaim_text length stats:\n{df['_claim_len'].describe()}")

    empty = df[df["claim_text"].str.strip() == ""]
    print(f"\nRows with empty claim_text: {len(empty)}")

    df.drop(columns=["_claim_len"], inplace=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved -> {out}")

    print("\n=== SAMPLE EXTRACTED CLAIMS (fake, extracted_from_ithil) ===")
    sample = df[(df["label"] == 0) & (df["extraction_status"] == "extracted_from_ithil")].head(8)
    for _, r in sample.iterrows():
        print(f"  ORIG  : {str(r['text'])[:120]}")
        print(f"  CLAIM : {r['claim_text'][:120]}")
        print()

    print("\n=== SAMPLE MANUAL REVIEW ROWS ===")
    for _, r in manual.head(5).iterrows():
        print(f"  TEXT  : {str(r['text'])[:150]}")
        print()


if __name__ == "__main__":
    main()
