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
from config import FAKE_NEWS_1500_MODELS_DIR, FAKE_NEWS_MODELS_DIR, MODEL_REGISTRY
from utils import setup_logger

logger = setup_logger("fake_news_classifier")

METRICS_TO_SHOW = ["test_accuracy", "test_precision_fake", "test_recall_fake",
                    "test_f1_fake", "test_f1_macro"]


def compare(models_dir: Path, label: str):
    rows = []
    for key in MODEL_REGISTRY:
        metrics_path = models_dir / key / "test_metrics.json"
        if not metrics_path.exists():
            logger.info(f"Skipping '{key}' - {metrics_path} not found (not trained yet)")
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"model": key, "checkpoint": MODEL_REGISTRY[key]}
        row.update({m: metrics.get(m) for m in METRICS_TO_SHOW})
        rows.append(row)

    if not rows:
        logger.error(
            f"No trained classifiers in {models_dir}. Run train_classifier.py first."
        )
        return False

    df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    out_path = models_dir / "classifier_comparison.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(f"Fake-news classifier comparison ({label}):\n" + df.to_string(index=False))
    logger.info(f"Saved -> {out_path}")
    return True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override models dir (default: try models_fake_news_1500 then models_fake_news)",
    )
    args = p.parse_args()

    if args.models_dir:
        ok = compare(Path(args.models_dir), str(args.models_dir))
        sys.exit(0 if ok else 1)

    # Prefer the 1500-run dir if any metrics exist there
    if any((FAKE_NEWS_1500_MODELS_DIR / k / "test_metrics.json").exists() for k in MODEL_REGISTRY):
        ok = compare(FAKE_NEWS_1500_MODELS_DIR, "news_dataset_1500")
    else:
        ok = compare(FAKE_NEWS_MODELS_DIR, "Tamil-News-Headlines")
    if ok:
        logger.info(
            "Next: compare these numbers against the evidence-aware LLM pipeline's "
            "verdicts on the same test.csv headlines (not automated yet - see docs)."
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
