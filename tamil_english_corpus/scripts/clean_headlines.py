"""
Tamil Headline Fact-Check Neutralization & Cleaning Pipeline
=============================================================
Safely cleans the `headline` column across dataset splits:
1. Strips verified fact-checking boilerplate trailers (e.g. "என்று பரவும் வீடியோ உண்மையா?").
2. Neutralizes unambiguous attribution inquiry verbs to declarative form (e.g. "கூறினாரா?" -> "கூறினார்").
3. Converts unambiguous passive allegation inquiries to declarative form (e.g. "வீசப்பட்டதா?" -> "வீசப்பட்டது").
4. Flags ambiguous / open interrogative headlines for manual review without fact distortion.
5. Preserves untouched original in `headline_original` and writes cleaned result to `headline_clean`.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"c:\Users\palak\OneDrive\Desktop\tamil_english_corpus")
DATA_DIR = ROOT_DIR / "data" / "final"

FILES_TO_PROCESS = [
    DATA_DIR / "train.csv",
    DATA_DIR / "validation.csv",
    DATA_DIR / "test.csv",
    DATA_DIR / "tamil_news_dataset_1400_balanced.csv",
]

# ----------------------------------------------------------------------
# 1. Boilerplate trailers to remove
# ----------------------------------------------------------------------
BOILERPLATE_TRAILERS = [
    # 1A. என்று/என பரவும்/வைரலாகும்... உண்மையா/போலியா?
    (
        r'\s*(?:என்று|என)\s+(?:பரவும்|வைரலாகும்|வைரலான|பகிரப்படும்|வெளியான|உலவும்)\s+(?:வீடியோ|புகைப்படம்|நியூஸ்\s*கார்டு|தகவல்|பதிவு|ஆடியோ|செய்தி|செய்திகள்|போட்டோ|காட்சி)\s+(?:உண்மையா|உண்மையானதா|உண்மையானவையா|போலியா|சரியா|பொய்யா)\s*\?*$',
        "boilerplate_viral_media_unmaiya"
    ),
    # 1B. என்று/என பரவும்/வைரலாகும்... தற்போது/தமிழ்நாட்டில் எடுக்கப்பட்டதா/நடந்ததா?
    (
        r'\s*(?:என்று|என)\s+(?:பரவும்|வைரலாகும்|வைரலான|பகிரப்படும்|வெளியான)\s+(?:வீடியோ|புகைப்படம்|நியூஸ்\s*கார்டு|தகவல்|பதிவு|ஆடியோ|செய்தி)\s*[-–—]?\s*(?:தற்போது|இப்போது|தமிழ்நாட்டில்|சமீபத்தில்)?\s*(?:எடுக்கப்பட்டதா|நடந்ததா|உண்மையா|போலியா)\s*\?*$',
        "boilerplate_viral_media_origin_check"
    ),
    # 1C. என்று/என பரவும்/வைரலாகும்... ? (without உண்மையா)
    (
        r'\s*(?:என்று|என)\s+(?:பரவும்|வைரலாகும்|வைரலான|பகிரப்படும்|வெளியான|உலவும்)\s+(?:வீடியோ|புகைப்படம்|நியூஸ்\s*கார்டு|தகவல்|பதிவு|ஆடியோ|செய்தி|செய்திகள்|போட்டோ|காட்சி)\s*\?*$',
        "boilerplate_viral_media_trailer"
    ),
    # 1D. என்று/என பரவும் வதந்தி / போலிச் செய்தி?
    (
        r'\s*(?:என்று|என)\s+(?:பரவும்|வைரலாகும்|பகிரப்படும்)\s+(?:வதந்தி|போலிச்\s*செய்தி)\s*\?*$',
        "boilerplate_viral_rumor_trailer"
    ),
    # 1E. ... என்று வெளியிட்டதாக பரவும் நியூஸ் கார்டு/தகவல் உண்மையா/போலியா?
    (
        r'\s*என்று\s+வெளியிட்டதாக\s+பரவும்\s+(?:நியூஸ்\s*கார்டு|தகவல்|பதிவு|செய்தி)\s+(?:உண்மையா|போலியா)\s*\?*$',
        "boilerplate_published_rumor_unmaiya"
    ),
    # 1F. வைரலாகும் வீடியோ உண்மையா / பரவும் தகவல் உண்மையா at the end
    (
        r'[\s,;-]+(?:வைரலாகும்|பரவும்|பகிரப்படும்)\s+(?:வீடியோ|புகைப்படம்|தகவல்|பதிவு|செய்தி)\s+(?:உண்மையா|போலியா)\s*\?*$',
        "boilerplate_viral_unmaiya"
    ),
    # 1G. Standalone காட்சி இதுவா? / புகைப்படம் இதுவா? / ஐடி கார்டு இதுவா?
    (
        r'\s+இதுவா\s*\?*$',
        "boilerplate_is_this_footage"
    ),
    # 1H. Standalone உண்மையா? / போலியா? / உண்மையானதா? at the end
    (
        r'[\s,;-]+(?:உண்மையா|உண்மையானதா|உண்மையானவையா|போலியா|போலியானதா|சரியா)\s*\?*$',
        "boilerplate_standalone_unmaiya"
    ),
]

ADVERBIAL_VERB_CONVERSIONS = [
    (r'தாக்கப்பட்டதாக$', 'தாக்கப்பட்டார்'),
    (r'செய்ததாக$', 'செய்தார்'),
    (r'நடந்ததாக$', 'நடந்தது'),
    (r'தேங்கியுள்ளதாக$', 'தேங்கியுள்ளது'),
    (r'தொடங்கிவிட்டதாக$', 'தொடங்கிவிட்டனர்'),
    (r'தாக்குவதாக$', 'தாக்குகின்றனர்'),
    (r'வெளியிட்டதாக$', 'வெளியிட்டது'),
]

# ----------------------------------------------------------------------
# 2. Unambiguous attribution verbs
# ----------------------------------------------------------------------
ATTRIBUTION_VERBS = [
    (r'\bகூறினாரா\s*\?*$', 'கூறினார்', 'verb_neutralized_kuri'),
    (r'\bகூறியுள்ளாரா\s*\?*$', 'கூறியுள்ளார்', 'verb_neutralized_kuri_pres'),
    (r'\bசெய்தாரா\s*\?*$', 'செய்தார்', 'verb_neutralized_seitha'),
    (r'\bசெய்துள்ளாரா\s*\?*$', 'செய்துள்ளார்', 'verb_neutralized_seitha_pres'),
    (r'\bபேசினாரா\s*\?*$', 'பேசினார்', 'verb_neutralized_pesi'),
    (r'\bபேசியுள்ளாரா\s*\?*$', 'பேசியுள்ளார்', 'verb_neutralized_pesi_pres'),
    (r'\bவெளியிட்டதா\s*\?*$', 'வெளியிட்டது', 'verb_neutralized_veliyi'),
    (r'\bவெளியிட்டுள்ளதா\s*\?*$', 'வெளியிட்டுள்ளது', 'verb_neutralized_veliyi_pres'),
    (r'\bதெரிவித்தாரா\s*\?*$', 'தெரிவித்தார்', 'verb_neutralized_therivi'),
    (r'\bதெரிவித்துள்ளாரா\s*\?*$', 'தெரிவித்துள்ளார்', 'verb_neutralized_therivi_pres'),
    (r'\bஅறிவித்தாரா\s*\?*$', 'அறிவித்தார்', 'verb_neutralized_arivi'),
    (r'\bஅறிவித்துள்ளாரா\s*\?*$', 'அறிவித்துள்ளார்', 'verb_neutralized_arivi_pres'),
    (r'\bஅறிவித்தார்களா\s*\?*$', 'அறிவித்தார்கள்', 'verb_neutralized_arivi_pl'),
    (r'\bஅளித்தாரா\s*\?*$', 'அளித்தார்', 'verb_neutralized_alitha'),
    (r'\bகேட்டாரா\s*\?*$', 'கேட்டார்', 'verb_neutralized_keta'),
    (r'\bமறுத்தாரா\s*\?*$', 'மறுத்தார்', 'verb_neutralized_marutha'),
    (r'\bஎச்சரித்தாரா\s*\?*$', 'எச்சரித்தார்', 'verb_neutralized_echaritha'),
    (r'\bஒப்புக்கொண்டாரா\s*\?*$', 'ஒப்புக்கொண்டார்', 'verb_neutralized_oppu'),
    (r'\bஉறுதிப்படுத்தியதா\s*\?*$', 'உறுதிப்படுத்தியது', 'verb_neutralized_urudhi'),
    (r'\bசெய்ததா\s*\?*$', 'செய்தது', 'verb_neutralized_seithatha'),
    (r'\bபாராட்டினாரா\s*\?*$', 'பாராட்டினார்', 'verb_neutralized_paaraatu'),
    (r'\bமிரட்டினாரா\s*\?*$', 'மிரட்டினார்', 'verb_neutralized_mirattu'),
    (r'\bசெலுத்தினாரா\s*\?*$', 'செலுத்தினார்', 'verb_neutralized_seluthu'),
    (r'\bபார்த்தனரா\s*\?*$', 'பார்த்தனர்', 'verb_neutralized_paarthanar'),
    (r'\bசென்றனரா\s*\?*$', 'சென்றனர்', 'verb_neutralized_sentranar'),
    (r'\bசுட்டனரா\s*\?*$', 'சுட்டனர்', 'verb_neutralized_suttanar'),
    (r'\bசெய்தனரா\s*\?*$', 'செய்தனர்', 'verb_neutralized_seithanar'),
]

INVERSION_VERBS = [
    (r'\bகூறினாரா\s+([^\?]+)\s*\?*$', r'கூறினார் \1', 'verb_inversion_kuri'),
    (r'\bசெய்தாரா\s+([^\?]+)\s*\?*$', r'செய்தார் \1', 'verb_inversion_seitha'),
    (r'\bபேசினாரா\s+([^\?]+)\s*\?*$', r'பேசினார் \1', 'verb_inversion_pesi'),
    (r'\bதெரிவித்தாரா\s+([^\?]+)\s*\?*$', r'தெரிவித்தார் \1', 'verb_inversion_therivi'),
    (r'\bஅறிவித்தாரா\s+([^\?]+)\s*\?*$', r'அறிவித்தார் \1', 'verb_inversion_arivi'),
    (r'\bவிமர்சித்தாரா\s+([^\?]+)\s*\?*$', r'விமர்சித்தார் \1', 'verb_inversion_vimar'),
    (r'\bகிண்டல்\s+செய்தாரா\s+([^\?]+)\s*\?*$', r'கிண்டல் செய்தார் \1', 'verb_inversion_kindal'),
]

PASSIVE_EVENT_VERBS = [
    (r'\bவீசப்பட்டதா\s*\?*$', 'வீசப்பட்டது', 'passive_thrown'),
    (r'\bவெளியிடப்பட்டதா\s*\?*$', 'வெளியிடப்பட்டது', 'passive_released'),
    (r'\bவழங்கப்பட்டதா\s*\?*$', 'வழங்கப்பட்டது', 'passive_provided'),
    (r'\bநடத்தப்பட்டதா\s*\?*$', 'நடத்தப்பட்டது', 'passive_conducted'),
    (r'\bதாக்கப்பட்டதா\s*\?*$', 'தாக்கப்பட்டது', 'passive_attacked'),
    (r'\bஎரிக்கப்பட்டதா\s*\?*$', 'எரிக்கப்பட்டது', 'passive_burnt'),
    (r'\bவிழுந்ததா\s*\?*$', 'விழுந்தது', 'passive_fell'),
    (r'\bவெடித்ததா\s*\?*$', 'வெடித்தது', 'passive_exploded'),
    (r'\bஉயர்த்தப்பட்டதா\s*\?*$', 'உயர்த்தப்பட்டது', 'passive_raised'),
    (r'\bவிநியோகிக்கப்பட்டதா\s*\?*$', 'விநியோகிக்கப்பட்டது', 'passive_distributed'),
    (r'\bவைக்கப்பட்டிருந்ததா\s*\?*$', 'வைக்கப்பட்டிருந்தது', 'passive_placed_past'),
    (r'\bவைக்கப்பட்டதா\s*\?*$', 'வைக்கப்பட்டது', 'passive_placed'),
    (r'\bஅமைக்கப்பட்டதா\s*\?*$', 'அமைக்கப்பட்டது', 'passive_installed'),
    (r'\bவரவழைக்கப்பட்டதா\s*\?*$', 'வரவழைக்கப்பட்டது', 'passive_brought'),
    (r'\bஇடைநீக்கம்\s+செய்யப்பட்டாரா\s*\?*$', 'இடைநீக்கம் செய்யப்பட்டார்', 'passive_suspended'),
    (r'\bதடை\s+விதிக்கப்பட்டுள்ளதா\s*\?*$', 'தடை விதிக்கப்பட்டுள்ளது', 'passive_banned'),
]


def clean_headline_text(headline: str) -> dict:
    """Clean a single headline deterministically without using labels or metadata."""
    if not isinstance(headline, str) or not headline.strip():
        return {
            "headline_original": "",
            "headline_clean": "",
            "action": "UNCHANGED",
            "action_type": "empty",
            "review_flag": False,
            "review_reason": ""
        }

    orig = headline.strip()
    curr = orig

    # 1. Verified boilerplate removal
    for pat, tag in BOILERPLATE_TRAILERS:
        m = re.search(pat, curr)
        if m:
            curr = curr[:m.start()].strip()
            curr = re.sub(r'[\s,;-]+(?:என்று|என)$', '', curr).strip()
            # Convert any dangling adverbial verbal ending to declarative
            for adv_pat, adv_repl in ADVERBIAL_VERB_CONVERSIONS:
                curr = re.sub(adv_pat, adv_repl, curr).strip()
            curr = curr.rstrip('?').strip()
            return {
                "headline_original": orig,
                "headline_clean": curr,
                "action": "MODIFIED",
                "action_type": tag,
                "review_flag": False,
                "review_reason": ""
            }

    # 2. Inversion verb neutralization
    for pat, repl, tag in INVERSION_VERBS:
        if re.search(pat, curr):
            curr = re.sub(pat, repl, curr).strip()
            curr = curr.rstrip('?').strip()
            return {
                "headline_original": orig,
                "headline_clean": curr,
                "action": "MODIFIED",
                "action_type": tag,
                "review_flag": False,
                "review_reason": ""
            }

    # 3. Direct attribution verb neutralization
    # Check claim indicator or explicit verb ending
    has_claim_indicator = bool(
        " என்று " in curr or " என " in curr or re.search(r'["\'‘“].*?["\'’”]', curr) or " பற்றி " in curr
    )
    if has_claim_indicator:
        for pat, repl, tag in ATTRIBUTION_VERBS:
            if re.search(pat, curr):
                curr = re.sub(pat, repl, curr).strip()
                curr = curr.rstrip('?').strip()
                return {
                    "headline_original": orig,
                    "headline_clean": curr,
                    "action": "MODIFIED",
                    "action_type": tag,
                    "review_flag": False,
                    "review_reason": ""
                }

    # 4. Passive event verb neutralization
    for pat, repl, tag in PASSIVE_EVENT_VERBS:
        if re.search(pat, curr):
            curr = re.sub(pat, repl, curr).strip()
            curr = curr.rstrip('?').strip()
            return {
                "headline_original": orig,
                "headline_clean": curr,
                "action": "MODIFIED",
                "action_type": tag,
                "review_flag": False,
                "review_reason": ""
            }

    # 5. Trailing punctuation fix (e.g. "..?" or "!?")
    if curr.endswith('.?') or curr.endswith('!?'):
        curr = curr[:-1].rstrip('.').strip()
        return {
            "headline_original": orig,
            "headline_clean": curr,
            "action": "MODIFIED",
            "action_type": "trailing_punctuation_cleaned",
            "review_flag": False,
            "review_reason": ""
        }

    # 6. Flag open/speculative interrogative headlines
    if curr.endswith('?'):
        return {
            "headline_original": orig,
            "headline_clean": orig,
            "action": "FLAGGED_FOR_REVIEW",
            "action_type": "interrogative_requires_manual_review",
            "review_flag": True,
            "review_reason": "Contains open/speculative interrogative phrasing without verified boilerplate."
        }

    # 7. Unchanged already declarative
    return {
        "headline_original": orig,
        "headline_clean": orig,
        "action": "UNCHANGED",
        "action_type": "already_declarative",
        "review_flag": False,
        "review_reason": ""
    }


def process_dataset_file(filepath: Path) -> dict:
    print(f"\nProcessing {filepath.name}...")
    df = pd.read_csv(filepath, encoding="utf-8")
    orig_len = len(df)

    results = [clean_headline_text(h) for h in df["headline"]]
    res_df = pd.DataFrame(results)

    # Insert / assign columns
    df["headline_original"] = res_df["headline_original"]
    df["headline_clean"] = res_df["headline_clean"]
    df["headline"] = res_df["headline_clean"]  # Keep active headline column synced
    df["review_flag"] = res_df["review_flag"]
    df["review_reason"] = res_df["review_reason"]
    df["cleaning_action"] = res_df["action"]

    # Reorder columns logically: id, headline, headline_clean, headline_original, text, label, ...
    cols = df.columns.tolist()
    priority_order = [
        "id",
        "headline",
        "headline_clean",
        "headline_original",
        "text",
        "label",
        "label_name",
        "source",
        "url",
        "char_count",
        "word_count",
        "review_flag",
        "review_reason",
        "cleaning_action",
        "investigation_details",
    ]
    ordered_cols = [c for c in priority_order if c in cols] + [c for c in cols if c not in priority_order]
    df = df[ordered_cols]

    # Save cleanly with UTF-8
    df.to_csv(filepath, index=False, encoding="utf-8")
    assert len(df) == orig_len, f"Row count mismatch in {filepath.name}!"

    action_counts = res_df["action"].value_counts().to_dict()
    flagged_count = int(res_df["review_flag"].sum())
    print(f"  Rows preserved: {len(df)} (Zero row loss)")
    print(f"  Actions: {action_counts}")
    print(f"  Flagged for review: {flagged_count}")

    return {
        "file": filepath.name,
        "total": len(df),
        "modified": action_counts.get("MODIFIED", 0),
        "unchanged": action_counts.get("UNCHANGED", 0),
        "flagged": action_counts.get("FLAGGED_FOR_REVIEW", 0),
    }


def main():
    summary = []
    for fp in FILES_TO_PROCESS:
        if fp.exists():
            summary.append(process_dataset_file(fp))
        else:
            print(f"File not found: {fp}")

    print("\n================ FINAL PIPELINE SUMMARY ================")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
