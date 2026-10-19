#!/usr/bin/env python3
"""Tokenizer length stats, near-duplicate headlines, date-range note."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_CKPT, OUT, RAW_CSV, ensure_dirs, load_raw  # noqa: E402


def token_stats():
    raw = load_raw()
    headlines = raw["headline"].astype(str).tolist()
    rows = []
    for key, ckpt in MODEL_CKPT.items():
        tok = AutoTokenizer.from_pretrained(ckpt)
        lengths = [len(tok.encode(h, add_special_tokens=True)) for h in headlines]
        truncated = sum(1 for L in lengths if L > 64)
        rows.append({
            "tokenizer": key,
            "checkpoint": ckpt,
            "median_tokens_with_special": float(pd.Series(lengths).median()),
            "mean_tokens": float(pd.Series(lengths).mean()),
            "max_tokens": int(max(lengths)),
            "pct_truncated_at_64": 100.0 * truncated / len(lengths),
            "n_truncated": truncated,
            "n": len(lengths),
        })
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "data_stats" / "tokenizer_stats.csv", index=False)
    (OUT / "data_stats" / "tokenizer_stats.json").write_text(json.dumps(rows, indent=2))


def near_duplicates(threshold: float = 0.9):
    """Character-level SequenceMatcher ratio on headlines."""
    from difflib import SequenceMatcher

    raw = load_raw()
    texts = raw["headline"].astype(str).tolist()
    ids = raw["id"].tolist()
    pairs = []
    n = len(texts)
    # O(n^2) on 1400 is fine (~1M compares)
    for i in range(n):
        for j in range(i + 1, n):
            r = SequenceMatcher(None, texts[i], texts[j]).ratio()
            if r > threshold:
                pairs.append({
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "ratio": round(r, 4),
                    "headline_a": texts[i],
                    "headline_b": texts[j],
                    "source_a": raw.loc[i, "source"],
                    "source_b": raw.loc[j, "source"],
                    "label_a": raw.loc[i, "label_name"],
                    "label_b": raw.loc[j, "label_name"],
                })
    pd.DataFrame(pairs).to_csv(OUT / "data_stats" / "near_duplicates.csv", index=False, encoding="utf-8-sig")
    (OUT / "data_stats" / "near_duplicates_meta.json").write_text(json.dumps({
        "threshold": threshold,
        "n_pairs": len(pairs),
        "method": "difflib.SequenceMatcher.ratio",
    }, indent=2))
    print(f"Near-duplicate pairs >{threshold}: {len(pairs)}")


def date_note():
    raw = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    date_like = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()]
    note = {
        "date_columns_found": date_like,
        "message": (
            "No publication date column in tamil_news_dataset_1400_balanced.csv; "
            "cannot report date range per class."
            if not date_like else "see columns"
        ),
        "columns": list(raw.columns),
    }
    (OUT / "data_stats" / "date_range.json").write_text(json.dumps(note, indent=2))
    print(note)


def main():
    ensure_dirs()
    token_stats()
    near_duplicates(0.9)
    date_note()


if __name__ == "__main__":
    main()
