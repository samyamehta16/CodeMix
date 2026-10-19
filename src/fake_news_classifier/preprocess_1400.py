"""
Prepares data/raw/tamil_news_dataset_1400_balanced.csv for MuRIL/IndicBERT.

Keeps `source` so leakage-aware splits are possible.

Two split modes:
  --split random   : stratified 80/10/10 (easy; source leakage intact) -> data/fake_news_1400/
  --split hard     : source-disjoint train/test (main paper eval)     -> data/fake_news_1400_hard/

Hard split (no shared sources between train and test):
  Train REAL : Dinamalar, Dinamalar-TamilNadu, Dinamalar-India
  Train FAKE : FactCrescendoTamil
  Test  REAL : PuthiyaThalaimurai, Vikatan-TamilNadu
  Test  FAKE : BOOMTamil-FactCheck, NewscheckerTamil
  Dev        : 10% stratified holdout from train

Run:
    python src/fake_news_classifier/preprocess_1400.py --split hard
    python src/fake_news_classifier/preprocess_1400.py --split random
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FAKE_NEWS_1400_DATA_DIR,
    FAKE_NEWS_1400_HARD_DATA_DIR,
    FAKE_NEWS_1400_RAW_CSV,
)
from utils import clean_comment, setup_logger

logger = setup_logger("fake_news_classifier_1400")

LABEL_NAME_TO_LABEL = {"fake": "FAKE", "real": "REAL"}

HARD_TRAIN_SOURCES = {
    "Dinamalar",
    "Dinamalar-TamilNadu",
    "Dinamalar-India",
    "FactCrescendoTamil",
}
HARD_TEST_SOURCES = {
    "PuthiyaThalaimurai",
    "Vikatan-TamilNadu",
    "BOOMTamil-FactCheck",
    "NewscheckerTamil",
}


def load_and_clean(text_column: str = "headline") -> pd.DataFrame:
    df = pd.read_csv(FAKE_NEWS_1400_RAW_CSV, encoding="utf-8-sig")
    n_start = len(df)

    df["label"] = df["label_name"].astype(str).str.strip().str.lower().map(LABEL_NAME_TO_LABEL)
    if text_column == "headline":
        df["text"] = df["headline"].astype(str)
    elif text_column == "body":
        df["text"] = df["text"].astype(str)
    elif text_column == "headline_body":
        df["text"] = df["headline"].astype(str) + " " + df["text"].astype(str)
    else:
        raise ValueError(f"Unknown text_column: {text_column}")

    df["text"] = df["text"].apply(lambda t: clean_comment(t, keep_emoji=False))
    df["source"] = df["source"].astype(str).str.strip()

    df = df.dropna(subset=["text", "label", "source"])
    df = df[df["label"].isin(["REAL", "FAKE"])]
    df = df[df["text"].str.len() >= 5]
    df = df.drop_duplicates(subset="text")

    logger.info(f"Loaded {n_start} rows, kept {len(df)} after clean/dedup")
    logger.info("Label distribution:\n" + df["label"].value_counts().to_string())
    logger.info("Source x label:\n" + pd.crosstab(df["source"], df["label"]).to_string())
    return df[["text", "label", "source"]].reset_index(drop=True)


def _save_splits(out_dir: Path, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(
            f"{name}: {len(split_df)} "
            f"{split_df['label'].value_counts().to_dict()} "
            f"sources={sorted(split_df['source'].unique())} -> {path}"
        )
    meta_path = out_dir / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Wrote {meta_path}")


def split_random(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    _save_splits(
        FAKE_NEWS_1400_DATA_DIR,
        train_df,
        dev_df,
        test_df,
        {
            "mode": "random_stratified",
            "note": "Sources may appear in both train and test (leakage-prone).",
            "train_n": len(train_df),
            "dev_n": len(dev_df),
            "test_n": len(test_df),
        },
    )


def split_hard(df: pd.DataFrame):
    """Source-disjoint split: no outlet appears in both train and test."""
    known = HARD_TRAIN_SOURCES | HARD_TEST_SOURCES
    missing = set(df["source"].unique()) - known
    if missing:
        logger.warning(f"Sources not assigned to hard split (dropped): {sorted(missing)}")

    train_pool = df[df["source"].isin(HARD_TRAIN_SOURCES)].copy()
    test_df = df[df["source"].isin(HARD_TEST_SOURCES)].copy()

    overlap = set(train_pool["source"]) & set(test_df["source"])
    if overlap:
        raise RuntimeError(f"Source overlap between train and test: {overlap}")

    # Balance train by undersampling majority class (FactCrescendo dominates FAKE)
    n_real = (train_pool["label"] == "REAL").sum()
    n_fake = (train_pool["label"] == "FAKE").sum()
    if n_fake > n_real:
        real_part = train_pool[train_pool["label"] == "REAL"]
        fake_part = train_pool[train_pool["label"] == "FAKE"].sample(n=n_real, random_state=42)
        train_pool = pd.concat([real_part, fake_part], ignore_index=True)
        logger.info(f"Balanced train: undersampled FAKE to {n_real} (was {n_fake})")

    train_df, dev_df = train_test_split(
        train_pool, test_size=0.1, stratify=train_pool["label"], random_state=42
    )

    meta = {
        "mode": "source_disjoint_hard",
        "train_sources": sorted(HARD_TRAIN_SOURCES),
        "test_sources": sorted(HARD_TEST_SOURCES),
        "note": (
            "Primary paper evaluation. Train and test share no sources, so the model "
            "cannot rely on outlet identity. Test FAKE is small (BOOM+Newschecker only) "
            "because FactCrescendo dominates FAKE; rely on macro-F1 and per-class metrics."
        ),
        "train_n": len(train_df),
        "dev_n": len(dev_df),
        "test_n": len(test_df),
        "test_label_counts": test_df["label"].value_counts().to_dict(),
        "train_label_counts": train_df["label"].value_counts().to_dict(),
    }
    _save_splits(FAKE_NEWS_1400_HARD_DATA_DIR, train_df, dev_df, test_df, meta)


def main(text_column: str = "headline", split: str = "hard"):
    if not FAKE_NEWS_1400_RAW_CSV.exists():
        logger.error(f"{FAKE_NEWS_1400_RAW_CSV} not found.")
        sys.exit(1)

    df = load_and_clean(text_column=text_column)
    if split == "random":
        split_random(df)
    else:
        split_hard(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text-column", choices=["headline", "body", "headline_body"], default="headline")
    p.add_argument("--split", choices=["hard", "random"], default="hard")
    args = p.parse_args()
    main(text_column=args.text_column, split=args.split)
