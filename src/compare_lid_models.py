"""
Compare fine-tuned LID taggers (MuRIL vs IndicBERT, and optionally more)
by reading each model's models/<key>/test_metrics.json, produced by
src/train_lid_tagger.py, and printing/saving a side-by-side table.

Run (after training at least one model):
    python src/compare_lid_models.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import MODEL_REGISTRY, TRAINED_MODELS_DIR
from utils import setup_logger

logger = setup_logger()

METRICS_TO_SHOW = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]


def main():
    rows = []
    for key in MODEL_REGISTRY:
        metrics_path = TRAINED_MODELS_DIR / key / "test_metrics.json"
        if not metrics_path.exists():
            logger.info(f"Skipping '{key}' - {metrics_path} not found (not trained yet)")
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"model": key, "checkpoint": MODEL_REGISTRY[key]}
        row.update({m: metrics.get(m) for m in METRICS_TO_SHOW})
        rows.append(row)

    if not rows:
        logger.error("No trained models found yet. Run src/train_lid_tagger.py first "
                     "(for at least one of --model muril / --model indicbert).")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    out_path = TRAINED_MODELS_DIR / "model_comparison.csv"
    df.to_csv(out_path, index=False)

    logger.info("Model comparison (test set):\n" + df.to_string(index=False))
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
