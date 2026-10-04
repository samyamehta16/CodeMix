"""
Step 1: Clean the raw YouTube comment CSV.

Input : data/raw/youtube_comments.csv
         columns -> video_id, comment_id, text, author, like_count, published_at
Output: data/processed/cleaned_comments.csv
         same columns + 'clean_text', duplicates/empties/low-quality rows dropped

Run:
    python src/preprocess.py
    python src/preprocess.py --keep-emoji false
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_CSV, CLEANED_CSV, PROCESSED_DIR
from utils import clean_comment, is_low_quality, setup_logger

logger = setup_logger()


def main(keep_emoji: bool = True, input_csv: Path = RAW_CSV, output_csv: Path = CLEANED_CSV):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        logger.error(f"Raw CSV not found at {input_csv}. Put your file there "
                      f"(or pass --input) with columns: video_id, comment_id, "
                      f"text, author, like_count, published_at")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    required_cols = {"video_id", "comment_id", "text", "author", "like_count", "published_at"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Missing expected columns: {missing}")
        sys.exit(1)

    n_start = len(df)
    logger.info(f"Loaded {n_start} raw comments")

    # 1. drop exact duplicate comment_ids
    df = df.drop_duplicates(subset="comment_id", keep="first")

    # 2. drop duplicate text (spam/copy-paste comments)
    df = df.drop_duplicates(subset="text", keep="first")

    # 3. clean text
    df["clean_text"] = df["text"].apply(lambda t: clean_comment(t, keep_emoji=keep_emoji))

    # 4. drop empty / low-quality rows (emoji-only, punctuation-only, single char)
    mask_ok = ~df["clean_text"].apply(is_low_quality)
    dropped_low_quality = (~mask_ok).sum()
    df = df[mask_ok]

    n_end = len(df)
    logger.info(f"Dropped {n_start - n_end} rows "
                f"(duplicates + low-quality, of which {dropped_low_quality} low-quality)")
    logger.info(f"Remaining: {n_end} comments")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved cleaned data -> {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keep-emoji", type=str, default="true",
                    help="true/false - keep emojis in clean_text")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    keep_emoji = args.keep_emoji.lower() in ("true", "1", "yes")
    input_csv = Path(args.input) if args.input else RAW_CSV
    output_csv = Path(args.output) if args.output else CLEANED_CSV

    main(keep_emoji=keep_emoji, input_csv=input_csv, output_csv=output_csv)
