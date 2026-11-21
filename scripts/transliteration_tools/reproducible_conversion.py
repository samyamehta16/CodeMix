#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproducible_conversion.py

Reproducible, auditable pipeline for Tamil-to-Tanglish Romanisation
and schema standardization for Tamil Fake News Research.

Key Principles:
1. Strict read-only handling of raw dataset (SHA-256 verification).
2. Separation of mechanical cleanup, phonological transliteration, and translation.
3. Decoupling of English translations into 'headline_en' column.
4. Complete label-blindness (label columns are never read during text conversion).
5. Full traceability: preserves original Tamil text, IDs, sources, URLs, and splits.
"""

import os
import sys
import argparse
import hashlib
import re
import json
import unicodedata
import pandas as pd

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add scripts directory to path to load engine
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import tanglish_engine as te

# Extended Canonical Entity and Acronym Mapping for Research Reproducibility
CANONICAL_ACRONYMS = {
    'ஆர்எஸ்எஸ்': 'RSS',
    'ஆர்.எஸ்.எஸ்': 'RSS',
    'ஏடிஎம்': 'ATM',
    'ஏ.டி.எம்': 'ATM',
    'சி.ஏ.ஏ': 'CAA',
    'சிஏஏ': 'CAA',
    'திமுக': 'DMK',
    'தி.மு.க': 'DMK',
    'அதிமுக': 'ADMK',
    'அ.தி.மு.க': 'ADMK',
    'பாஜக': 'BJP',
    'பா.ஜ.க': 'BJP',
    'டிவிகே': 'TVK',
    'தவெக': 'TVK',
    'த.வெ.க': 'TVK',
    'சிஎஸ்கே': 'CSK',
    'சி.எஸ்.கே': 'CSK',
    'சிசிடிவி': 'CCTV',
    'சி.சி.டி.வி': 'CCTV',
    'பிஐபி': 'PIB',
    'சிஎன்என்': 'CNN',
    'பிடிஐ': 'PTI',
    'ஏஎன்ஐ': 'ANI',
    'எஸ்ஐஆர்': 'SIR',
    'என்டிஏ': 'NDA',
}

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_headline_components(headline_str):
    """
    Parses headline to separate Tamil headline, English translations, and format tags.
    Returns: (cleaned_headline_ta, headline_en, format_tag)
    """
    if not headline_str or pd.isna(headline_str):
        return "", "", ""
        
    s = str(headline_str).strip()
    if '|' not in s:
        return s, "", ""
        
    parts = [p.strip() for p in s.split('|')]
    head_ta = parts[0]
    head_en = ""
    tag = ""
    
    for p in parts[1:]:
        # Detect if part is an English translation/summary (>3 words, predominantly Latin)
        latin_words = re.findall(r'[A-Za-z]+', p)
        has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', p))
        
        if len(latin_words) >= 3 and not has_tamil:
            head_en = p
        elif len(p) < 25 and any(k in p.lower() for k in ['album', 'interview', 'photo', 'கழுகார்']):
            tag = p
        else:
            # Topic tag or continuation
            if not head_en:
                head_ta = f"{head_ta} - {p}"
                
    return head_ta, head_en, tag

def convert_record_text(text_str, apply_acronym_standardization=True):
    """
    Converts Tamil text to Romanised Tanglish without conditioning on labels.
    """
    if not text_str or pd.isna(text_str):
        return text_str
        
    t = str(text_str)
    
    # Optional pre-pass: canonical acronym replacement
    if apply_acronym_standardization:
        for tam_acr, eng_acr in CANONICAL_ACRONYMS.items():
            pattern = r'(?<![\u0B80-\u0BFF])' + re.escape(tam_acr) + r'(?![\u0B80-\u0BFF])'
            t = re.sub(pattern, eng_acr, t)
            
    # Core transliteration
    return te.transliterate_text(t)

def build_split_lookup(raw_df=None):
    """
    Constructs deterministic row_id -> split mapping.
    Maps from data/fake_news_1400_hard (Source-Disjoint Hard Split)
    or data/fake_news_1400_hard/fake_news_1400 (Approach A Random Stratified).
    """
    split_dir = 'data/fake_news_1400_hard'
    split_map = {}
    
    # Try Approach B (Source Disjoint Hard)
    meta_path = os.path.join(split_dir, 'split_meta.json')
    if os.path.exists(meta_path) and raw_df is not None:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            train_srcs = set(meta.get('train_sources', []))
            test_srcs = set(meta.get('test_sources', []))
            
            # Map by headline matching to train/dev/test CSVs
            for sname in ['train', 'dev', 'test']:
                sp = os.path.join(split_dir, f"{sname}.csv")
                if os.path.exists(sp):
                    sdf = pd.read_csv(sp)
                    s_texts = set(sdf['text'].str.strip())
                    for idx, r in raw_df.iterrows():
                        hl_clean = str(r['headline']).split('|')[0].strip()
                        if str(r['headline']).strip() in s_texts or hl_clean in s_texts:
                            split_map[r['id']] = sname
        except Exception as e:
            print(f"[!] Warning building split lookup: {e}")
            
    return split_map

def run_conversion(
    raw_path='data/raw/tamil_news_dataset_1400_balanced.csv',
    output_path='output/tamil_news_candidate_v2_sample100.csv',
    num_rows=100,
    exclusions_path='proposed_exclusions.csv',
    apply_acronym_standardization=True,
    filter_exclusions=False
):
    print("=" * 75)
    print("REPRODUCIBLE TAMIL TO TANGLISH CONVERSION PIPELINE")
    print("=" * 75)
    
    assert os.path.exists(raw_path), f"Input file not found: {raw_path}"
    sha_input_pre = compute_sha256(raw_path)
    print(f"[*] Input File: {raw_path}")
    print(f"[*] Input SHA-256 (pre): {sha_input_pre}")
    
    df_raw = pd.read_csv(raw_path)
    total_raw_len = len(df_raw)
    print(f"[*] Total Raw Records: {total_raw_len}")
    
    # Exclusions handling
    excluded_ids = set()
    if exclusions_path and os.path.exists(exclusions_path):
        df_ex = pd.read_csv(exclusions_path)
        excluded_ids = set(df_ex['id'])
        print(f"[*] Loaded {len(excluded_ids)} proposed exclusions from {exclusions_path}")
        
    # Select slice
    if num_rows is not None and num_rows > 0:
        target_df = df_raw.iloc[:num_rows].copy()
        print(f"[*] Mode: Processing first {num_rows} records (Evaluation Sample)")
    else:
        target_df = df_raw.copy()
        print(f"[*] Mode: Processing ALL {total_raw_len} records (Full Dataset)")
        
    # Load split mapping
    split_lookup = build_split_lookup(raw_df=df_raw)
    print(f"[*] Loaded split mapping for {len(split_lookup)} records.")
    
    # Process records
    processed_records = []
    skipped_count = 0
    
    for idx, r in target_df.iterrows():
        rec_id = r['id']
        if filter_exclusions and rec_id in excluded_ids:
            skipped_count += 1
            continue
            
        hl_raw = r['headline']
        tx_raw = r['text']
        inv_raw = r.get('investigation_details', None)
        
        # Parse headline components
        hl_ta_clean, hl_en, fmt_tag = extract_headline_components(hl_raw)
        
        # Transliterate text fields
        hl_rom = convert_record_text(hl_ta_clean, apply_acronym_standardization)
        tx_rom = convert_record_text(tx_raw, apply_acronym_standardization)
        inv_rom = convert_record_text(inv_raw, apply_acronym_standardization) if pd.notna(inv_raw) else None
        
        # Word and character counts based on Romanised text
        c_count = len(tx_rom) if isinstance(tx_rom, str) else 0
        w_count = len(tx_rom.split()) if isinstance(tx_rom, str) else 0
        
        # Determine split
        rec_split = split_lookup.get(rec_id, "unassigned")
        
        rec = {
            'id': rec_id,
            'headline_ta': hl_ta_clean,
            'headline_romanised': hl_rom,
            'headline_en': hl_en,
            'text_ta': tx_raw,
            'text_romanised': tx_rom,
            'investigation_details_ta': inv_raw if pd.notna(inv_raw) else None,
            'investigation_details_romanised': inv_rom,
            'label': r['label'],
            'label_name': r['label_name'],
            'source': r['source'],
            'url': r['url'],
            'split': rec_split,
            'char_count': c_count,
            'word_count': w_count,
            'format_tag': fmt_tag,
            'is_flagged_exclusion': (rec_id in excluded_ids)
        }
        processed_records.append(rec)
        
    out_df = pd.DataFrame(processed_records)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding='utf-8')
    
    sha_out = compute_sha256(output_path)
    sha_input_post = compute_sha256(raw_path)
    
    assert sha_input_pre == sha_input_post, "FATAL: Raw dataset was modified during conversion!"
    
    print("-" * 75)
    print(f"[*] Processed Records: {len(out_df)} (Skipped exclusions: {skipped_count})")
    print(f"[*] Output Saved To:   {output_path}")
    print(f"[*] Output SHA-256:    {sha_out}")
    print(f"[*] Input SHA-256 (post): {sha_input_post} (VERIFIED UNCHANGED)")
    print("=" * 75)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproducible Tamil Romanisation Pipeline")
    parser.add_argument('--raw', default='data/raw/tamil_news_dataset_1400_balanced.csv')
    parser.add_argument('--output', default='output/tamil_news_candidate_v2_sample100.csv')
    parser.add_argument('--rows', type=int, default=100, help="Number of rows (default 100 sample; set 0 or -1 for full)")
    parser.add_argument('--exclude-path', default='proposed_exclusions.csv')
    parser.add_argument('--filter-exclusions', action='store_true', default=False)
    parser.add_argument('--standardize-acronyms', action='store_true', default=True)
    
    args = parser.parse_args()
    num_r = None if args.rows in [0, -1] else args.rows
    run_conversion(
        raw_path=args.raw,
        output_path=args.output,
        num_rows=num_r,
        exclusions_path=args.exclude_path,
        apply_acronym_standardization=args.standardize_acronyms,
        filter_exclusions=args.filter_exclusions
    )
