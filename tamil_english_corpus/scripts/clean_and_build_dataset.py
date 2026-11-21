"""
Tamil Real vs Fake News Dataset Pipeline
=========================================
Builds a curated, de-leaked, strictly deduplicated, and balanced
Tamil / Tamil-English Real vs Fake News dataset for Hugging Face.

Key Features:
1. De-leakage: Isolates the viral claim/rumor into `text` and quarantines
   the fact-checker's investigation/debunking into `investigation_details`.
2. Content Filtering: Removes advertisements, horoscopes, calculators,
   stock market tickers, and sports match summaries from real news.
3. Strict Deduplication: Deduplicates by URL, headline, and text content.
4. Stratified Splits: Creates 80/10/10 train/validation/test splits
   with zero cross-split leakage and fixed random seed (42).
"""

import hashlib
import json
import logging
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# Windows UTF-8 console output setup
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_pipeline")

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_INPUT = ROOT_DIR / "data" / "cleaned" / "news_dataset_1500.csv"
RAW_INPUT = ROOT_DIR / "scripts" / "tamil_fakenews_dataset.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "final"
OUTPUT_FULL_CSV = OUTPUT_DIR / "tamil_news_dataset_1400_balanced.csv"
OUTPUT_TRAIN_CSV = OUTPUT_DIR / "train.csv"
OUTPUT_VAL_CSV = OUTPUT_DIR / "validation.csv"
OUTPUT_TEST_CSV = OUTPUT_DIR / "test.csv"

RANDOM_SEED = 42
TARGET_PER_CLASS = 700


def make_id(text_or_url: str) -> str:
    """Generate deterministic 16-char SHA-256 identifier."""
    norm = unicodedata.normalize("NFKC", str(text_or_url).strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def clean_headline(hl: str) -> str:
    """Clean headline of trailing news brand pipes and fact-check tags."""
    if not isinstance(hl, str):
        return ""
    h = hl.strip()
    # Remove leading fact check prefix tags
    h = re.sub(
        r"^(?:Fact Check|உண்மை சரிபார்ப்பு|FactCheck|போலிச் செய்தி)\s*[:|-]\s*",
        "",
        h,
        flags=re.IGNORECASE,
    )
    # Remove trailing portal branding
    h = re.sub(
        r"\s*\|\s*(?:Fact Crescendo|Dinamalar|Puthiya Thalaimurai|Vikatan|BOOM|Newschecker|OneIndia).*$",
        "",
        h,
        flags=re.IGNORECASE,
    )
    h = re.sub(r"[ \t]+", " ", h)
    return h.strip()


def clean_real_text(text: str, headline: str) -> str:
    """Clean real news article body, removing publication timestamps and ads."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if headline and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()
    # Remove metadata timestamps e.g. "ADDED : ...", "UPDATED : ..."
    t = re.sub(r"^(?:ADDED|UPDATED)\s*:[^\n]*\n?", "", t, flags=re.MULTILINE)
    # Remove newspaper subscription / ad notices
    t = re.sub(r"தினமலர் ஆண்ட்ராய்டு ஆப் டவுன்லோட் செய்ய.*", "", t, flags=re.DOTALL)
    t = re.sub(r"விகடன் சந்தா சேர.*", "", t, flags=re.DOTALL)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
    return t


def segment_fake_article(text: str, headline: str) -> tuple[str, str]:
    """
    Separate viral claim text from fact-checker debunking investigation.
    Returns: (claim_text, investigation_details)
    """
    if not isinstance(text, str):
        return "", ""

    t = text.strip()
    if headline and t.startswith(headline.strip()):
        t = t[len(headline.strip()):].strip()

    claim_text = ""
    investigation_text = ""

    # Pattern A: FactCrescendoTamil structure with 'தகவலின் விவரம்:'
    if "தகவலின் விவரம்:" in t:
        parts = t.split("தகவலின் விவரம்:", 1)
        intro = parts[0].strip()
        details = parts[1].strip()

        # Split at investigation start
        inv_split = re.split(
            r"(?:உண்மை அறிவோம்:|ஆய்வு:|ஆய்வு முறை:|எங்கள் ஆய்வில்|நாம் மேற்கொண்ட ஆய்வில்|பின்னணி:|முடிவு:)",
            details,
            1,
        )
        claim_section = inv_split[0].strip()
        if len(inv_split) > 1:
            investigation_text = inv_split[1].strip()

        # Strip investigative framing from claim intro
        clean_intro = re.sub(
            r"(?:இது தொடர்பாக|இது குறித்து|சமூக வலைதளங்களில்|சமூக ஊடகங்களில்).*?ஆய்வு.*",
            "",
            intro,
            flags=re.DOTALL,
        ).strip()

        # Strip tipline / whatsapp solicitation from claim section
        clean_claim = re.sub(
            r"இதனை வாசகர்கள் நமக்கு வாட்ஸ்ஆப்.*?கேட்டனர்\.?",
            "",
            claim_section,
            flags=re.DOTALL,
        ).strip()
        clean_claim = re.sub(
            r"ஃபேஸ்புக் பதிவு.*?காணலாம்.*", "", clean_claim, flags=re.DOTALL
        ).strip()

        if clean_intro and clean_claim:
            claim_text = f"{clean_intro}\n\n{clean_claim}".strip()
        elif clean_claim:
            claim_text = clean_claim
        else:
            claim_text = clean_intro

    # Pattern B: Newschecker / BOOM 'Claim' vs 'Fact' structure
    elif "Claim" in t and "Fact" in t:
        m = re.search(
            r"Claim\s*[:]?\s*(.*?)(?:Fact\s*[:]?\s*(.*)|$)", t, flags=re.DOTALL
        )
        if m:
            claim_text = m.group(1).strip()
            investigation_text = m.group(2).strip() if m.group(2) else ""

    # Fallback: Strip conclusion and footer
    if not claim_text:
        clean_t = re.sub(r"முடிவு:.*", "", t, flags=re.DOTALL)
        clean_t = re.sub(
            r"(?:எங்கள் ஆய்வில்|நாம் மேற்கொண்ட ஆய்வில்).* உறுதி செய்யப்படுகிறது.*",
            "",
            clean_t,
            flags=re.DOTALL,
        )
        claim_text = clean_t.strip()
        investigation_text = t

    # Final cleanup of claim text
    claim_text = re.sub(
        r"(?:Claim\s*Link|Archived?\s*Link|Archive\s*Link)\s*\d*(?:\s*[l|I/]\s*(?:Claim\s*Link|Archived?\s*Link|Archive\s*Link)\s*\d*)*",
        "",
        claim_text,
        flags=re.IGNORECASE,
    )
    claim_text = re.sub(r"https?://\S+", "", claim_text)
    claim_text = re.sub(r"[ \t]+", " ", claim_text)
    claim_text = re.sub(r"\n\s*\n+", "\n\n", claim_text).strip()

    return claim_text, investigation_text


def is_noise_real_article(url: str, headline: str, text: str) -> bool:
    """Filter out non-comparable categories (ads, sports scores, finance tables, horoscopes)."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    cat = parts[1] if len(parts) > 1 and parts[0] == "news" else (parts[0] if parts else "")

    noise_categories = {
        "calculators",
        "horoscope",
        "bundle",
        "collection",
        "personal-finance",
        "business-share-market",
        "business-savings",
        "business-banking-and-finance",
        "sports-cricket",
        "sports-badminton",
        "sports-tennis",
        "sports-football",
        "sports-others",
        "sports",
    }
    if cat in noise_categories:
        return True

    # Check for commercial advertisements / shopping events
    ad_phrases = ["ஷாப்பர்ஸ் எக்ஸ்போ", "மெகா ஷாப்பிங்", "தினமலர் ஸ்மார்ட் ஷாப்பர்ஸ்"]
    for phrase in ad_phrases:
        if phrase in headline or phrase in text[:200]:
            return True

    return False


def load_real_records() -> list[dict]:
    """Extract and clean high-quality real news records."""
    log.info("Loading real news records...")
    candidates = []

    # 1. Load from cleaned 1500 CSV
    if CLEANED_INPUT.exists():
        df_clean = pd.read_csv(CLEANED_INPUT, encoding="utf-8")
        clean_reals = df_clean[df_clean["label"] == 1]
        for _, r in clean_reals.iterrows():
            candidates.append({
                "url": str(r.get("url", "")).strip(),
                "headline": str(r.get("headline", "")).strip(),
                "text": str(r.get("text", "")).strip(),
                "source": str(r.get("source", "RealNews")).strip(),
            })

    # 2. Also load from raw CSV to gather any missed genuine political/social news
    if RAW_INPUT.exists():
        try:
            df_raw = pd.read_csv(RAW_INPUT, encoding="utf-8", encoding_errors="replace")
            raw_reals = df_raw[df_raw["label"] == "real"]
            for _, r in raw_reals.iterrows():
                candidates.append({
                    "url": str(r.get("url", "")).strip(),
                    "headline": str(r.get("headline", "")).strip(),
                    "text": str(r.get("text", "")).strip(),
                    "source": str(r.get("source_name", "RealNews")).strip(),
                })
        except Exception as e:
            log.warning("Could not read raw CSV: %s", e)

    log.info("Total real candidate rows loaded: %d", len(candidates))

    seen_urls = set()
    seen_hl_norms = set()
    seen_text_hashes = set()
    records = []

    for c in candidates:
        url = c["url"]
        raw_hl = c["headline"]
        raw_text = c["text"]
        source = c["source"]

        if not url or not raw_hl or not raw_text:
            continue
        if is_noise_real_article(url, raw_hl, raw_text):
            continue

        hl = clean_headline(raw_hl)
        body = clean_real_text(raw_text, hl)

        if len(hl) < 10 or len(body) < 80:
            continue
        if url in seen_urls:
            continue

        norm_hl = re.sub(r"[\s\W_]+", "", hl.lower())
        if norm_hl in seen_hl_norms:
            continue

        text_hash = hashlib.md5(body[:250].encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            continue

        seen_urls.add(url)
        seen_hl_norms.add(norm_hl)
        seen_text_hashes.add(text_hash)

        records.append({
            "id": make_id(url),
            "headline": hl,
            "text": body,
            "investigation_details": "",  # Real news has no debunking investigation
            "label": 1,
            "label_name": "real",
            "source": source,
            "url": url,
            "char_count": len(body),
            "word_count": len(body.split()),
        })

    log.info("Total cleaned, deduplicated, relevant real news records: %d", len(records))
    return records


def load_fake_records() -> list[dict]:
    """Extract, de-leak, and clean fake news records."""
    log.info("Loading fake news records...")
    candidates = []

    if CLEANED_INPUT.exists():
        df_clean = pd.read_csv(CLEANED_INPUT, encoding="utf-8")
        clean_fakes = df_clean[df_clean["label"] == 0]
        for _, r in clean_fakes.iterrows():
            candidates.append({
                "url": str(r.get("url", "")).strip(),
                "headline": str(r.get("headline", "")).strip(),
                "text": str(r.get("text", "")).strip(),
                "source": str(r.get("source", "FactCheck")).strip(),
            })

    log.info("Total fake candidate rows loaded: %d", len(candidates))

    seen_urls = set()
    seen_hl_norms = set()
    seen_text_hashes = set()
    records = []

    for c in candidates:
        url = c["url"]
        raw_hl = c["headline"]
        raw_text = c["text"]
        source = c["source"]

        if not url or not raw_hl or not raw_text:
            continue

        hl = clean_headline(raw_hl)
        claim, investigation = segment_fake_article(raw_text, hl)

        if len(hl) < 10 or len(claim) < 30:
            continue
        if url in seen_urls:
            continue

        norm_hl = re.sub(r"[\s\W_]+", "", hl.lower())
        if norm_hl in seen_hl_norms:
            continue

        text_hash = hashlib.md5(claim[:250].encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            continue

        seen_urls.add(url)
        seen_hl_norms.add(norm_hl)
        seen_text_hashes.add(text_hash)

        records.append({
            "id": make_id(url),
            "headline": hl,
            "text": claim,
            "investigation_details": investigation,
            "label": 0,
            "label_name": "fake",
            "source": source,
            "url": url,
            "char_count": len(claim),
            "word_count": len(claim.split()),
        })

    log.info("Total cleaned, de-leaked, deduplicated fake news records: %d", len(records))
    return records


def check_cross_split_similarity(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str, threshold: float = 0.88) -> int:
    """Check for near-duplicate headline leakage across splits."""
    hl_a = df_a["headline"].tolist()
    hl_b = df_b["headline"].tolist()
    leak_count = 0

    for i, a in enumerate(hl_a):
        for j, b in enumerate(hl_b):
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= threshold:
                leak_count += 1
                log.warning(
                    "Potential near-duplicate between %s and %s (ratio=%.2f):\n  A: %s\n  B: %s",
                    name_a,
                    name_b,
                    ratio,
                    a,
                    b,
                )
    return leak_count


def build_stratified_splits(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 80/10/10 train/validation/test splits."""
    # Deterministic shuffle
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    reals = df_shuffled[df_shuffled["label"] == 1].reset_index(drop=True)
    fakes = df_shuffled[df_shuffled["label"] == 0].reset_index(drop=True)

    n_real = len(reals)
    n_fake = len(fakes)

    # 80% train, 10% val, 10% test
    n_real_train = int(n_real * 0.80)
    n_real_val = int(n_real * 0.10)

    n_fake_train = int(n_fake * 0.80)
    n_fake_val = int(n_fake * 0.10)

    train_df = pd.concat([
        reals.iloc[:n_real_train],
        fakes.iloc[:n_fake_train],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_df = pd.concat([
        reals.iloc[n_real_train:n_real_train + n_real_val],
        fakes.iloc[n_fake_train:n_fake_train + n_fake_val],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_df = pd.concat([
        reals.iloc[n_real_train + n_real_val:],
        fakes.iloc[n_fake_train + n_fake_val:],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    log.info("=== Starting Tamil News Dataset Pipeline ===")

    # 1. Load curated records
    real_records = load_real_records()
    fake_records = load_fake_records()

    # Determine balanced size
    balanced_per_class = min(len(real_records), len(fake_records), TARGET_PER_CLASS)
    log.info(
        "Balancing dataset: selecting %d Real and %d Fake rows (Total = %d).",
        balanced_per_class,
        balanced_per_class,
        balanced_per_class * 2,
    )

    final_real = real_records[:balanced_per_class]
    final_fake = fake_records[:balanced_per_class]

    combined = final_real + final_fake
    df = pd.DataFrame(combined)

    # Global final deduplication safeguard
    df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["headline"]).drop_duplicates(subset=["text"])
    log.info("Final dataset size after strict deduplication: %d rows", len(df))

    # Deterministic shuffle
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # 2. Build 80/10/10 stratified splits
    train_df, val_df, test_df = build_stratified_splits(df, seed=RANDOM_SEED)

    # 3. Cross-split leakage verification
    log.info("Verifying zero cross-split near-duplicate leakage...")
    leak_train_val = check_cross_split_similarity(train_df, val_df, "Train", "Val", threshold=0.92)
    leak_train_test = check_cross_split_similarity(train_df, test_df, "Train", "Test", threshold=0.92)
    leak_val_test = check_cross_split_similarity(val_df, test_df, "Val", "Test", threshold=0.92)
    total_leaks = leak_train_val + leak_train_test + leak_val_test
    log.info("Cross-split leakage check complete: %d high-similarity pairs found.", total_leaks)

    # 4. Save to files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "headline",
        "text",
        "label",
        "label_name",
        "source",
        "url",
        "char_count",
        "word_count",
        "investigation_details",
    ]

    df[cols].to_csv(OUTPUT_FULL_CSV, index=False, encoding="utf-8-sig")
    train_df[cols].to_csv(OUTPUT_TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df[cols].to_csv(OUTPUT_VAL_CSV, index=False, encoding="utf-8-sig")
    test_df[cols].to_csv(OUTPUT_TEST_CSV, index=False, encoding="utf-8-sig")

    log.info("Saved full dataset to: %s", OUTPUT_FULL_CSV)
    log.info("Saved train split to:    %s (%d rows)", OUTPUT_TRAIN_CSV, len(train_df))
    log.info("Saved val split to:      %s (%d rows)", OUTPUT_VAL_CSV, len(val_df))
    log.info("Saved test split to:     %s (%d rows)", OUTPUT_TEST_CSV, len(test_df))

    # 5. Print comprehensive pipeline report
    print("\n" + "=" * 65)
    print("TAMIL REAL VS FAKE NEWS DATASET - PIPELINE REPORT")
    print("=" * 65)
    print(f"Total Rows:            {len(df)}")
    print(f"Real News (1):         {(df['label'] == 1).sum()} ({(df['label'] == 1).mean() * 100:.1f}%)")
    print(f"Fake News (0):         {(df['label'] == 0).sum()} ({(df['label'] == 0).mean() * 100:.1f}%)")
    print(f"Unique URLs:           {df['url'].nunique()}")
    print(f"Unique Headlines:      {df['headline'].nunique()}")
    print(f"Unique Model Texts:    {df['text'].nunique()}")
    print(f"Cross-Split Leaks:     {total_leaks}")

    print("\nSplits Breakdown:")
    print(f"  Train (80%):         {len(train_df)} rows (Real: {(train_df['label'] == 1).sum()}, Fake: {(train_df['label'] == 0).sum()})")
    print(f"  Validation (10%):    {len(val_df)} rows (Real: {(val_df['label'] == 1).sum()}, Fake: {(val_df['label'] == 0).sum()})")
    print(f"  Test (10%):          {len(test_df)} rows (Real: {(test_df['label'] == 1).sum()}, Fake: {(test_df['label'] == 0).sum()})")

    print("\nSource Distribution across Full Dataset:")
    print(df["source"].value_counts().to_string())

    print("\nLength Statistics (Model Input Text):")
    print(f"  Mean Characters:     {df['char_count'].mean():.1f} (Real: {df[df['label'] == 1]['char_count'].mean():.1f}, Fake: {df[df['label'] == 0]['char_count'].mean():.1f})")
    print(f"  Mean Words:          {df['word_count'].mean():.1f} (Real: {df[df['label'] == 1]['word_count'].mean():.1f}, Fake: {df[df['label'] == 0]['word_count'].mean():.1f})")

    print("\nTarget Leakage Audit in Model Input `text` (% incidence):")
    audit_words = ["ஆய்வு", "தவறான", "போலி", "உறுதி செய்யப்படுகிறது", "Fact Check", "BOOM"]
    for w in audit_words:
        f_count = df[df["label"] == 0]["text"].str.contains(w, regex=False).sum()
        r_count = df[df["label"] == 1]["text"].str.contains(w, regex=False).sum()
        f_pct = f_count / (df["label"] == 0).sum() * 100
        r_pct = r_count / (df["label"] == 1).sum() * 100
        print(f"  {w:22s}: Fake={f_count:3d} ({f_pct:4.1f}%) | Real={r_count:3d} ({r_pct:4.1f}%)")

    print("=" * 65)


if __name__ == "__main__":
    main()
