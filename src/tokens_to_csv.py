"""
Flatten annotated_comments.jsonl into a per-token CSV: one row per token,
with its predicted language label - the format you'd want for spreadsheet
review, error analysis, or as a stepping stone to the CoNLL export.

Failed/empty annotations are skipped automatically (nothing to flatten).
Emoji tokens are dropped by default (pass --keep-emoji to keep them).

Run:
    python src/tokens_to_csv.py
    python src/tokens_to_csv.py --keep-emoji
    python src/tokens_to_csv.py --input data/annotated/annotated_comments.jsonl --output data/annotated/tokens_flat.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import ANNOTATED_JSONL, ANNOTATED_DIR
from utils import setup_logger, strip_emoji

logger = setup_logger()


def flatten(input_path: Path, keep_emoji: bool = False) -> pd.DataFrame:
    rows = []
    n_skipped = 0
    n_emoji_dropped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("annotation_failed") or not rec.get("tokens"):
                n_skipped += 1
                continue

            idx = 0
            for tok in rec["tokens"]:
                token_text = tok["token"]
                if not keep_emoji:
                    cleaned = strip_emoji(token_text)
                    if not cleaned:
                        n_emoji_dropped += 1
                        continue
                    token_text = cleaned

                rows.append({
                    "comment_id": rec["comment_id"],
                    "video_id": rec.get("video_id", ""),
                    "token_index": idx,
                    "token": token_text,
                    "label": tok["label"],
                })
                idx += 1

    if n_skipped:
        logger.info(f"Skipped {n_skipped} failed/empty annotations")
    if n_emoji_dropped:
        logger.info(f"Dropped {n_emoji_dropped} emoji-only tokens")

    return pd.DataFrame(rows)


def main(input_path: Path, output_path: Path, keep_emoji: bool = False):
    if not input_path.exists():
        logger.error(f"{input_path} not found. Run run_annotation.py first.")
        sys.exit(1)

    df = flatten(input_path, keep_emoji=keep_emoji)
    if df.empty:
        logger.error("No usable tokens found - nothing to write.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"Wrote {len(df)} token rows from {df['comment_id'].nunique()} comments -> {output_path}")
    logger.info("Label counts:\n" + df["label"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Path to annotated jsonl (default: config.ANNOTATED_JSONL)")
    p.add_argument("--output", type=str, default=None, help="Path to write the flat token CSV")
    p.add_argument("--keep-emoji", action="store_true", help="Keep emoji tokens instead of dropping them")
    args = p.parse_args()

    input_path = Path(args.input) if args.input else ANNOTATED_JSONL
    output_path = Path(args.output) if args.output else ANNOTATED_DIR / "tokens_flat.csv"

    main(input_path, output_path, keep_emoji=args.keep_emoji)
