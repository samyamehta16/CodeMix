#!/usr/bin/env python3
"""Export HuggingFace-ready CSV/Parquet + dataset card (no article body)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HARD_DIR, OUT, RANDOM_DIR, RAW_CSV, ensure_dirs, load_raw  # noqa: E402


def split_map(data_dir: Path) -> dict[str, str]:
    """Map cleaned text -> split name."""
    m = {}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(data_dir / f"{split}.csv", encoding="utf-8-sig")
        for t in df["text"].astype(str):
            m[t.strip()] = split
    return m


def main():
    ensure_dirs()
    raw = load_raw()
    # rebuild cleaned text the same way preprocess does
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from utils import clean_comment

    raw = raw.copy()
    raw["headline_clean"] = raw["headline"].astype(str).map(lambda t: clean_comment(t, keep_emoji=False))
    rand_map = split_map(RANDOM_DIR)
    hard_map = split_map(HARD_DIR)

    rows = []
    for _, r in raw.iterrows():
        text = str(r["headline_clean"]).strip()
        rs = rand_map.get(text, "unused")
        hs = hard_map.get(text, "unused")
        # hard train undersampled FAKE — many FactCrescendo rows unused
        label_name = str(r["label_name"]).strip().lower()
        rows.append({
            "id": str(r["id"]),
            "headline": str(r["headline"]),
            "label": 0 if label_name == "fake" else 1,
            "label_name": label_name,
            "source": str(r["source"]),
            "url": str(r.get("url", "")),
            "char_count": int(r["char_count"]) if pd.notna(r.get("char_count")) else len(str(r["headline"])),
            "word_count": int(r["word_count"]) if pd.notna(r.get("word_count")) else len(str(r["headline"]).split()),
            "random_split": rs,
            "hard_split": hs,
        })
    df = pd.DataFrame(rows)
    out_dir = OUT / "huggingface"
    csv_path = out_dir / "tamil_news_dataset_1400_balanced_hf.csv"
    pq_path = out_dir / "tamil_news_dataset_1400_balanced_hf.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        (out_dir / "parquet_error.txt").write_text(str(e))
        pq_path = None

    readme = f"""---
pretty_name: Tamil News Veracity 1400 (TruthGuard)
language:
- ta
- en
license: other
task_categories:
- text-classification
size_categories:
- 1K<n<10K
---

# Tamil News Veracity 1400 (TruthGuard)

Self-collected balanced Tamil headline veracity corpus used in the TruthGuard project.

## Dataset summary

| Item | Value |
|---|---|
| Rows | {len(df)} |
| Classes | 700 `fake` / 700 `real` |
| Text unit | Headline only (article body **not** included in this release) |
| Language | Tamil (native script) |

## Columns

- `id`: original integer id
- `headline`: Tamil headline text
- `label`: 0 = fake, 1 = real
- `label_name`: `fake` / `real`
- `source`: publishing outlet
- `url`: source URL when available
- `char_count`, `word_count`
- `random_split`: `train` / `dev` / `test` (stratified 80/10/10, seed 42)
- `hard_split`: `train` / `dev` / `test` / `unused` (source-disjoint protocol; unused = held out of hard train due to FAKE undersampling or unassigned)

## Important confound (read before training)

Every source is class-pure (fact-checkers → FAKE; mainstream news → REAL). Random-split near-ceiling accuracy largely reflects **source/style separability**. Prefer the **hard** (source-disjoint) split for primary claims.

Hard test sources: PuthiyaThalaimurai, Vikatan-TamilNadu (REAL); BOOMTamil-FactCheck, NewscheckerTamil (FAKE).

## Recommended citation

TruthGuard: Dual-pipeline Tamil/Tanglish fake-news detection (VIT SCOPE). Please cite the accompanying paper when available.

## Ethical notes

Public web pages only. No social-media user handles. Downstream use for defensive fact-checking research.

## License

`other` — verify redistribution rights for outlet content before public Hugging Face upload; prefer CC-BY-4.0 if cleared, else research-only.
"""
    (out_dir / "README.md").write_text(readme)
    meta = {
        "n": len(df),
        "random_split_counts": df["random_split"].value_counts().to_dict(),
        "hard_split_counts": df["hard_split"].value_counts().to_dict(),
        "csv": str(csv_path),
        "parquet": str(pq_path) if pq_path else None,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
