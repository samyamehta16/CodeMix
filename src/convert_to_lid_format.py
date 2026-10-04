"""
Step 3: Convert the Gemini-annotated JSONL into a CoNLL-style token-tagging
dataset (one token + label per line, blank line between comments) - the
standard format used by LinCE / DravidianLangTech style LID datasets and
directly usable with HuggingFace token-classification fine-tuning scripts
for MuRIL / IndicBERT.

Emoji tokens are dropped by default (pass --keep-emoji to keep them) -
they're not linguistically TA/EN/MIX content and just add noise to a LID
tagger; kept consistent with tokens_to_csv.py's default behavior.

Also produces a train/dev/test split.

Run:
    python src/convert_to_lid_format.py
    python src/convert_to_lid_format.py --keep-emoji
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import ANNOTATED_JSONL, LID_DATASET_DIR
from utils import setup_logger, strip_emoji

logger = setup_logger()

random.seed(42)
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def load_annotations():
    """Returns (usable_sentences, n_failed)."""
    sentences = []
    n_failed = 0
    with open(ANNOTATED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["annotation_failed"] or not rec["tokens"]:
                n_failed += 1
                continue
            sentences.append(rec)
    return sentences, n_failed


def write_conll(sentences, path: Path, keep_emoji: bool = False) -> int:
    """Returns the number of emoji-only tokens dropped."""
    n_emoji_dropped = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in sentences:
            wrote_any = False
            for tok in rec["tokens"]:
                token_text = tok["token"]
                if not keep_emoji:
                    cleaned = strip_emoji(token_text)
                    if not cleaned:
                        n_emoji_dropped += 1
                        continue
                    token_text = cleaned
                f.write(f"{token_text}\t{tok['label']}\n")
                wrote_any = True
            if wrote_any:
                f.write("\n")  # sentence boundary
    return n_emoji_dropped


def main(keep_emoji: bool = False):
    LID_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    sentences, n_failed = load_annotations()
    if not sentences:
        logger.error(f"No usable annotations found in {ANNOTATED_JSONL}. "
                      f"Run run_annotation.py first.")
        sys.exit(1)

    random.shuffle(sentences)
    n = len(sentences)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_dev = int(n * SPLIT_RATIOS["dev"])

    splits = {
        "train": sentences[:n_train],
        "dev": sentences[n_train:n_train + n_dev],
        "test": sentences[n_train + n_dev:],
    }

    total_emoji_dropped = 0
    for split_name, split_data in splits.items():
        out_path = LID_DATASET_DIR / f"{split_name}.conll"
        total_emoji_dropped += write_conll(split_data, out_path, keep_emoji=keep_emoji)
        logger.info(f"{split_name}: {len(split_data)} sentences -> {out_path}")

    logger.info(f"Total usable sentences: {n} (dropped {n_failed} failed annotations)")
    if not keep_emoji:
        logger.info(f"Dropped {total_emoji_dropped} emoji-only tokens from the LID dataset")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keep-emoji", action="store_true", help="Keep emoji tokens instead of dropping them")
    args = p.parse_args()
    main(keep_emoji=args.keep_emoji)
