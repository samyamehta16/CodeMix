"""
Step 2: Run Gemini annotation over the cleaned comments.

Resumable by design: every result is appended to a JSONL checkpoint file
line-by-line as it's produced. If your free-tier quota runs out or the
script crashes, just re-run it - already-annotated comment_ids are skipped.

Run:
    python src/run_annotation.py
    python src/run_annotation.py --limit 50      # test on a small batch first
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLEANED_CSV, ANNOTATED_JSONL, ANNOTATED_CSV, ANNOTATED_DIR, SECONDS_BETWEEN_CALLS
from gemini_annotator import GeminiAnnotator
from utils import setup_logger

logger = setup_logger()


def load_done_ids(jsonl_path: Path) -> set:
    """Reads the checkpoint file to find comment_ids already annotated."""
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(row["comment_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main(limit: int = None):
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEANED_CSV.exists():
        logger.error(f"{CLEANED_CSV} not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_CSV)
    if limit:
        df = df.head(limit)

    done_ids = load_done_ids(ANNOTATED_JSONL)
    todo = df[~df["comment_id"].isin(done_ids)]
    logger.info(f"{len(done_ids)} already annotated, {len(todo)} remaining "
                f"(rate: ~{60/SECONDS_BETWEEN_CALLS:.0f} req/min -> "
                f"est. {len(todo)*SECONDS_BETWEEN_CALLS/60:.1f} min)")

    if len(todo) == 0:
        logger.info("Nothing to do. All rows already annotated.")
        return

    annotator = GeminiAnnotator()

    with open(ANNOTATED_JSONL, "a", encoding="utf-8") as out_f:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Annotating"):
            text = row["clean_text"]
            tokens = annotator.annotate(text)

            record = {
                "comment_id": row["comment_id"],
                "video_id": row["video_id"],
                "text": text,
                "tokens": tokens if tokens is not None else [],
                "annotation_failed": tokens is None,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # persist immediately - crash-safe

            time.sleep(SECONDS_BETWEEN_CALLS)  # respect free-tier RPM limit

    export_flat_csv()
    logger.info("Annotation run complete.")


def export_flat_csv():
    """Also writes a flat CSV (one row per comment, tokens as a JSON string)
    for easy viewing in Excel / to show your professor."""
    rows = []
    with open(ANNOTATED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            token_str = " ".join(f"{t['token']}/{t['label']}" for t in rec["tokens"])
            rows.append({
                "comment_id": rec["comment_id"],
                "video_id": rec["video_id"],
                "text": rec["text"],
                "annotated_tokens": token_str,
                "n_tokens": len(rec["tokens"]),
                "annotation_failed": rec["annotation_failed"],
            })
    pd.DataFrame(rows).to_csv(ANNOTATED_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Flat CSV export -> {ANNOTATED_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                    help="Only annotate the first N rows (useful for testing)")
    args = p.parse_args()
    main(limit=args.limit)
