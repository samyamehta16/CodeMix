#!/usr/bin/env python3
"""Error analysis on hard split for IndicBERTv2 and TF-IDF+NB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ensure_dirs, load_split  # noqa: E402

MARKERS = {
    "has_question": lambda t: "?" in t or "？" in t,
    "has_paravum": lambda t: "என்று பரவும்" in t,
    "has_unmai": lambda t: ("உண்மையா" in t) or ("உண்மை" in t),
}


def analyze(pred_path: Path, name: str):
    test = load_split("test", "hard")
    preds = pd.read_csv(pred_path)
    # Prefer source from predictions if present; else join from test
    if "source" not in preds.columns:
        df = preds.merge(test[["id", "text", "source"]], on="id", how="left")
    else:
        df = preds.merge(test[["id", "text"]], on="id", how="left")
        if "source" not in df.columns and "source_x" in df.columns:
            df["source"] = df["source_x"]
    if "text" not in df.columns and "text_x" in df.columns:
        df["text"] = df["text_x"]
    # per-source FAKE recall for BOOM / Newschecker
    fake = df[df["true"] == "FAKE"]
    per_src = {}
    for src in ["BOOMTamil-FactCheck", "NewscheckerTamil"]:
        sub = fake[fake["source"] == src]
        if len(sub) == 0:
            per_src[src] = {"n": 0, "recall": None}
        else:
            tp = int((sub["pred"] == "FAKE").sum())
            per_src[src] = {"n": int(len(sub)), "recall": tp / len(sub), "TP": tp, "FN": int(len(sub) - tp)}

    fn = df[(df["true"] == "FAKE") & (df["pred"] == "REAL")].copy()
    fp = df[(df["true"] == "REAL") & (df["pred"] == "FAKE")].copy()
    # up to 10 each
    fn_s = fn.head(10).copy()
    fp_s = fp.head(10).copy()
    for col, fn_ in MARKERS.items():
        fn_s[col] = fn_s["text"].astype(str).map(fn_)
    out_dir = OUT / "error_analysis" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_s.to_csv(out_dir / "false_negatives_10.csv", index=False, encoding="utf-8-sig")
    fp_s[["id", "text", "source", "true", "pred"]].to_csv(
        out_dir / "false_positives_10.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "model": name,
        "fake_source_recall": per_src,
        "n_FN_total": int(len(fn)),
        "n_FP_total": int(len(fp)),
        "FN_marker_rates_all": {
            k: float(fn["text"].astype(str).map(fn_).mean()) if len(fn) else None
            for k, fn_ in MARKERS.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(name, summary)
    return summary


def main():
    ensure_dirs()
    pairs = [
        ("indicbert-v2", OUT / "predictions" / "indicbert-v2_hard_seed42_test.csv"),
        ("TF-IDF_NB", OUT / "predictions" / "TF-IDF_plus_MultinomialNB_hard_deterministic_test.csv"),
    ]
    all_sum = {}
    for name, path in pairs:
        if path.exists():
            all_sum[name] = analyze(path, name)
        else:
            print("MISSING", path)
    (OUT / "error_analysis" / "summary.json").write_text(
        json.dumps(all_sum, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
