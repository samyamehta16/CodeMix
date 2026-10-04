"""
Compare fine-tuned fake-news classifiers (MuRIL vs IndicBERT, and
optionally more) by reading each model's
models_fake_news/<key>/test_metrics.json, produced by train_classifier.py.

Run (after training at least one model):
    python src/fake_news_classifier/compare_classifiers.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FAKE_NEWS_MODELS_DIR, MODEL_REGISTRY
from utils import setup_logger

logger = setup_logger("fake_news_classifier")

METRICS_TO_SHOW = ["test_accuracy", "test_precision_fake", "test_recall_fake",
                    "test_f1_fake", "test_f1_macro"]


def main():
    rows = []
    for key in MODEL_REGISTRY:
        metrics_path = FAKE_NEWS_MODELS_DIR / key / "test_metrics.json"
        if not metrics_path.exists():
            logger.info(f"Skipping '{key}' - {metrics_path} not found (not trained yet)")
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"model": key, "checkpoint": MODEL_REGISTRY[key]}
        row.update({m: metrics.get(m) for m in METRICS_TO_SHOW})
        rows.append(row)

    if not rows:
        logger.error("No trained classifiers found yet. Run train_classifier.py first "
                     "(for at least one of --model muril / --model indicbert).")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    out_path = FAKE_NEWS_MODELS_DIR / "classifier_comparison.csv"
    df.to_csv(out_path, index=False)

    logger.info("Fake-news classifier comparison (test set):\n" + df.to_string(index=False))
    logger.info(f"Saved -> {out_path}")
    logger.info("Next: compare these numbers against the evidence-aware LLM pipeline's "
                "verdicts on the same test.csv headlines (not automated yet - see docs).")


if __name__ == "__main__":
    main()
