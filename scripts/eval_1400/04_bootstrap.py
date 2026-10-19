#!/usr/bin/env python3
"""95% bootstrap CIs for macro-F1 and balanced Acc on hard-split predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, TRANSFORMERS, compute_metrics, ensure_dirs  # noqa: E402


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    f1s, bals = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx])
        f1s.append(m["macro_f1"])
        bals.append(m["balanced_accuracy"])
    f1s, bals = np.asarray(f1s), np.asarray(bals)
    return {
        "macro_f1_point": float(compute_metrics(y_true, y_pred)["macro_f1"]),
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "balanced_accuracy_point": float(compute_metrics(y_true, y_pred)["balanced_accuracy"]),
        "balanced_accuracy_ci95": [float(np.percentile(bals, 2.5)), float(np.percentile(bals, 97.5))],
        "n_boot": n_boot,
    }


def main():
    ensure_dirs()
    pred_dir = OUT / "predictions"
    candidates = []
    for m in TRANSFORMERS:
        candidates.append((m, pred_dir / f"{m}_hard_seed42_test.csv"))
    # TF-IDF + style if present
    for name in [
        "TF-IDF_plus_MultinomialNB_hard_deterministic_test.csv",
        "TF-IDF_plus_LogisticRegression_hard_deterministic_test.csv",
        "TF-IDF_plus_LinearSVC_hard_deterministic_test.csv",
        "style_lr_hard_test.csv",
    ]:
        candidates.append((name.replace("_hard_deterministic_test.csv", "").replace("_test.csv", ""), pred_dir / name))

    rows = []
    for label, path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ci = bootstrap_ci(df["true"].tolist(), df["pred"].tolist())
        row = {"Model": label, **ci}
        rows.append(row)
        save = OUT / "bootstrap" / f"{path.stem}_ci.json"
        save.write_text(json.dumps({"model": label, "file": str(path), **ci}, indent=2))
        print(label, "macroF1", ci["macro_f1_point"], ci["macro_f1_ci95"],
              "bal", ci["balanced_accuracy_point"], ci["balanced_accuracy_ci95"])

    pd.DataFrame(rows).to_csv(OUT / "bootstrap" / "hard_ci95_summary.csv", index=False)


if __name__ == "__main__":
    main()
