"""
Prepares data/raw/news_dataset_1500.csv for fine-tuning MuRIL/IndicBERT
as REAL-vs-FAKE classifiers.

Dataset: 1,500 Tamil news/fact-check articles, balanced 750 fake / 750 real.
Columns: id, headline, text, label, label_name, source, url, char_count, word_count.

Label mapping (from label_name — do NOT use numeric `label` alone without
checking convention):
  label_name=fake / label=0 -> FAKE
  label_name=real / label=1 -> REAL

IMPORTANT caveats for reporting:
1. Source perfectly predicts class (FactCrescendo/BOOM/Newschecker = fake;
   Dinamalar/PuthiyaThalaimurai/Vikatan = real). Source leakage risk.
2. Fake rows are fact-check articles with stock phrases ("உண்மை அறிவோம்",
   "ஆய்வு மேற்கொண்டோம்") absent from real news — body-text leakage.
   Default uses HEADLINE only to reduce that (still some style mismatch).

Run:
    python src/fake_news_classifier/preprocess_1500.py
    python src/fake_news_classifier/preprocess_1500.py --text-column body
    python src/fake_news_classifier/preprocess_1500.py --text-column headline_body
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FAKE_NEWS_1500_DATA_DIR, FAKE_NEWS_1500_RAW_CSV
from utils import clean_comment, setup_logger

logger = setup_logger("fake_news_classifier_1500")

LABEL_NAME_TO_LABEL = {"fake": "FAKE", "real": "REAL"}


def load_and_clean(text_column: str = "headline") -> pd.DataFrame:
    df = pd.read_csv(FAKE_NEWS_1500_RAW_CSV, encoding="utf-8-sig")
    n_start = len(df)

    # Prefer explicit label_name; fall back to numeric convention 0=fake, 1=real
    if "label_name" in df.columns:
        df["label"] = df["label_name"].astype(str).str.strip().str.lower().map(LABEL_NAME_TO_LABEL)
    else:
        df["label"] = df["label"].map({0: "FAKE", 1: "REAL", "0": "FAKE", "1": "REAL"})

    if text_column == "headline":
        df["text"] = df["headline"].astype(str)
    elif text_column == "body":
        df["text"] = df["text"].astype(str)
    elif text_column == "headline_body":
        df["text"] = (df["headline"].astype(str) + " " + df["text"].astype(str))
    else:
        raise ValueError(f"Unknown text_column: {text_column}")

    df["text"] = df["text"].apply(lambda t: clean_comment(t, keep_emoji=False))

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(["REAL", "FAKE"])]
    df = df[df["text"].str.len() >= 5]
    df = df.drop_duplicates(subset="text")

    logger.info(
        f"Loaded {n_start} rows, kept {len(df)} after clean/dedup "
        f"(dropped {before - len(df)} from filter; {n_start - before} nulls)"
    )
    logger.info("Label distribution:\n" + df["label"].value_counts().to_string())
    if "source" in df.columns:
        logger.info(
            "Source x label:\n"
            + pd.crosstab(df["source"], df["label"]).to_string()
        )
    return df[["text", "label"]].reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    FAKE_NEWS_1500_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    for name, split_df in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        out_path = FAKE_NEWS_1500_DATA_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"{name}: {len(split_df)} rows -> {out_path}")


def main(text_column: str = "headline"):
    if not FAKE_NEWS_1500_RAW_CSV.exists():
        logger.error(f"{FAKE_NEWS_1500_RAW_CSV} not found.")
        sys.exit(1)

    df = load_and_clean(text_column=text_column)
    split_and_save(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--text-column",
        choices=["headline", "body", "headline_body"],
        default="headline",
        help="What to classify on (default: headline — less fact-check body leakage)",
    )
    args = p.parse_args()
    main(text_column=args.text_column)
