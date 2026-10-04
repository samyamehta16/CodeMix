"""
Step 4 (recommended, for academic rigor): export a random sample of
annotated comments to Excel so a human (you / your professor / a Tamil
speaker) can spot-check Gemini's labels and estimate accuracy before
trusting this as training data for MuRIL/IndicBERT.

Run:
    python src/review_sample.py --n 50
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import ANNOTATED_JSONL, ANNOTATED_DIR
from utils import setup_logger

logger = setup_logger()
random.seed(7)


def main(n: int):
    if not ANNOTATED_JSONL.exists():
        logger.error(f"{ANNOTATED_JSONL} not found. Run run_annotation.py first.")
        sys.exit(1)

    records = []
    with open(ANNOTATED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if not rec["annotation_failed"]:
                    records.append(rec)

    sample = random.sample(records, min(n, len(records)))

    rows = []
    for rec in sample:
        annotated_str = " ".join(f"{t['token']}/{t['label']}" for t in rec["tokens"])
        rows.append({
            "comment_id": rec["comment_id"],
            "original_text": rec["text"],
            "gemini_annotation": annotated_str,
            "is_correct(Y/N)": "",       # reviewer fills this in
            "corrected_annotation": "",  # reviewer fills this in if wrong
            "notes": "",
        })

    out_path = ANNOTATED_DIR / "review_sample.xlsx"
    pd.DataFrame(rows).to_excel(out_path, index=False)
    logger.info(f"Wrote {len(rows)}-comment review sample -> {out_path}")
    logger.info("Have a Tamil speaker fill 'is_correct' + 'corrected_annotation', "
                "then you can report token-level accuracy / agreement to your professor.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    main(n=args.n)
