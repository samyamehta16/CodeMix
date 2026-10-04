"""
Prepares data/raw/Tamil-News-Headlines.csv for fine-tuning MuRIL/IndicBERT
as REAL-vs-FAKE headline classifiers (the "baseline classifier" comparison
point against the evidence-aware LLM pipeline in src/verification/).

Source dataset columns: Unnamed index, "English Version", "Label" (topic
category - politics/tech/sport/etc, NOT the fake/real label), "News"
(Tamil headline text), "Author", "Date", "Authenticity" (0/1).

Inspecting the data: Authenticity=0 rows are real tech/general news
headlines (dated ~2022); Authenticity=1 rows are Tamil translations of
known fact-checked FALSE claims (COVID/political misinformation, dated
~2018-2019 in the source). So: Authenticity 0 -> REAL, 1 -> FAKE.

KNOWN LIMITATION (disclose in your report): the REAL and FAKE rows come
from visibly different sources/topics/eras, not the same claims verified
true vs false. A classifier trained on this can partly learn to
distinguish "2022 tech news style" from "2018 political-claim style"
rather than "fake-ness" in general - a common criticism of this style of
fake-news dataset. Worth stating explicitly rather than presenting
accuracy numbers as if they measure pure fake-news detection ability.

ALSO NOTE: "News" is formal, pure Tamil (Tamil script) - not the
romanized Tamil-English code-mixed text your target application (YouTube
comments) actually looks like. A classifier trained here may not transfer
well to code-mixed social media text - worth testing/reporting explicitly
rather than assuming it will.

Run:
    python src/fake_news_classifier/preprocess.py
    python src/fake_news_classifier/preprocess.py --text-column english   # use English Version instead of Tamil
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FAKE_NEWS_DATA_DIR, FAKE_NEWS_LABELS, FAKE_NEWS_RAW_CSV
from utils import clean_comment, setup_logger

logger = setup_logger("fake_news_classifier")

AUTHENTICITY_TO_LABEL = {0: "REAL", 1: "FAKE"}


def load_and_clean(text_column: str = "tamil") -> pd.DataFrame:
    df = pd.read_csv(FAKE_NEWS_RAW_CSV)

    source_col = "News" if text_column == "tamil" else "English Version"
    df = df.rename(columns={source_col: "text"})
    df = df[["text", "Authenticity"]].copy()

    df["text"] = df["text"].apply(lambda t: clean_comment(t, keep_emoji=False))
    df["label"] = df["Authenticity"].map(AUTHENTICITY_TO_LABEL)

    n_start = len(df)
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.len() >= 5]  # drop near-empty headlines
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Loaded {n_start} rows, kept {len(df)} after dedup/cleaning "
                f"(dropped {n_start - len(df)})")

    logger.info("Label distribution:\n" + df["label"].value_counts().to_string())
    return df[["text", "label"]].reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    FAKE_NEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    for name, split_df in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        out_path = FAKE_NEWS_DATA_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"{name}: {len(split_df)} rows -> {out_path}")


def main(text_column: str = "tamil"):
    if not FAKE_NEWS_RAW_CSV.exists():
        logger.error(f"{FAKE_NEWS_RAW_CSV} not found.")
        sys.exit(1)

    df = load_and_clean(text_column=text_column)
    split_and_save(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text-column", choices=["tamil", "english"], default="tamil",
                    help="Use the Tamil 'News' column (default) or the 'English Version' column")
    args = p.parse_args()
    main(text_column=args.text_column)
