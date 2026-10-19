#!/usr/bin/env python3
"""Style-only logistic regression baseline on random + hard splits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    OUT,
    compute_metrics,
    ensure_dirs,
    load_split,
    metrics_row,
    save_metrics,
    save_preds,
)

MARKERS_PARAVUM = "என்று பரவும்"
MARKERS_UNMAI = ("உண்மையா", "உண்மை")


def style_features(texts: pd.Series) -> np.ndarray:
    rows = []
    for t in texts.astype(str):
        has_q = int("?" in t or "？" in t)
        length = len(t)
        has_paravum = int(MARKERS_PARAVUM in t)
        has_unmai = int(any(m in t for m in MARKERS_UNMAI))
        rows.append([has_q, length, has_paravum, has_unmai])
    return np.asarray(rows, dtype=float)


def run_one(which: str) -> dict:
    train = load_split("train", which)
    # Fit only on train; evaluate on test (dev unused for selection — single model)
    test = load_split("test", which)
    X_tr, y_tr = style_features(train["text"]), train["label"].values
    X_te, y_te = style_features(test["text"]), test["label"].values
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    m = compute_metrics(y_te, pred)
    save_preds(
        OUT / "predictions" / f"style_lr_{which}_test.csv",
        test["id"].tolist(),
        y_te.tolist(),
        pred.tolist(),
        extra={"source": test["source"].tolist(), "text": test["text"].tolist()},
    )
    save_metrics(OUT / "metrics" / f"style_lr_{which}_test.json", {
        "model": "style_only_logistic",
        "split": which,
        "features": [
            "has_question_mark",
            "headline_length",
            f'contains "{MARKERS_PARAVUM}"',
            'contains "உண்மையா" or "உண்மை"',
        ],
        "coef": {
            "has_question_mark": float(clf.coef_[0][0]),
            "headline_length": float(clf.coef_[0][1]),
            "contains_paravum": float(clf.coef_[0][2]),
            "contains_unmai": float(clf.coef_[0][3]),
        },
        "intercept": float(clf.intercept_[0]),
        **m,
    })
    return metrics_row(f"Style-only LR ({which})", m)


def main():
    ensure_dirs()
    rows = [run_one("random"), run_one("hard")]
    pd.DataFrame(rows).to_csv(OUT / "style_baseline" / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
