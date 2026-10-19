#!/usr/bin/env python3
"""Aggregate multi-seed metrics + deterministic TF-IDF baselines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HARD_DIR,
    OUT,
    RANDOM_DIR,
    TRANSFORMERS,
    compute_metrics,
    ensure_dirs,
    load_split,
    metrics_row,
    save_metrics,
    save_preds,
)


def run_tfidf(which: str):
    train = pd.read_csv((RANDOM_DIR if which == "random" else HARD_DIR) / "train.csv", encoding="utf-8-sig")
    test = load_split("test", which)
    models = {
        "TF-IDF + MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("clf", MultinomialNB()),
        ]),
        "TF-IDF + LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
    }
    rows = []
    for name, pipe in models.items():
        pipe.fit(train["text"], train["label"])
        pred = pipe.predict(test["text"])
        m = compute_metrics(test["label"].tolist(), pred)
        slug = name.replace(" ", "_").replace("+", "plus").replace("/", "-")
        tag = f"{slug}_{which}_deterministic"
        save_preds(
            OUT / "predictions" / f"{tag}_test.csv",
            test["id"].tolist(),
            test["label"].tolist(),
            pred.tolist(),
            extra={"source": test["source"].tolist()},
        )
        save_metrics(OUT / "metrics" / f"{tag}_test.json", {"model": name, "split": which, **m})
        rows.append(metrics_row(f"{name} ({which})", m))
    return rows


def aggregate_seeds():
    rows = []
    for which in ("random", "hard"):
        for model in TRANSFORMERS:
            vals_f1, vals_bal, vals_acc = [], [], []
            for seed in range(1, 6):
                path = OUT / "metrics" / f"{model}_{which}_seed{seed}_test.json"
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                vals_f1.append(d["macro_f1"])
                vals_bal.append(d["balanced_accuracy"])
                vals_acc.append(d["accuracy"])
            if not vals_f1:
                continue
            rows.append({
                "Model": model,
                "Split": which,
                "n_seeds": len(vals_f1),
                "Macro-F1 mean": float(np.mean(vals_f1)),
                "Macro-F1 std": float(np.std(vals_f1, ddof=1)) if len(vals_f1) > 1 else 0.0,
                "Bal.Acc mean": float(np.mean(vals_bal)),
                "Bal.Acc std": float(np.std(vals_bal, ddof=1)) if len(vals_bal) > 1 else 0.0,
                "Acc mean": float(np.mean(vals_acc)),
                "Acc std": float(np.std(vals_acc, ddof=1)) if len(vals_acc) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    tfidf_rows = run_tfidf("random") + run_tfidf("hard")
    pd.DataFrame(tfidf_rows).to_csv(OUT / "multiseed" / "tfidf_deterministic.csv", index=False)
    agg = aggregate_seeds()
    agg.to_csv(OUT / "multiseed" / "transformer_mean_std.csv", index=False)
    print("TF-IDF:")
    print(pd.DataFrame(tfidf_rows).to_string(index=False))
    print("\nMulti-seed:")
    print(agg.to_string(index=False) if len(agg) else "(no seed runs yet)")


if __name__ == "__main__":
    main()
