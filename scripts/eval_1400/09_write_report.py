#!/usr/bin/env python3
"""Compile RESULTS.md with tables + surprises list."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, TRANSFORMERS, ensure_dirs, metrics_row  # noqa: E402


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def row_from_metrics(name: str, m: dict) -> dict:
    return metrics_row(name, m)


def table_md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_No rows_\n"
    cols = [c for c in ["Model", "Acc", "Bal.Acc", "P_F", "R_F", "F1_F", "Macro-F1", "FP", "FN"] if c in df.columns]
    df = df[cols]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ensure_dirs()
    surprises = []
    sections = ["# TruthGuard 1400 Evaluation Suite Results\n"]
    sections.append("FAKE = positive class. Existing seed-42 splits unchanged.\n")

    # Style
    sections.append("## 1. Style-only logistic baseline\n")
    rows = []
    for which in ("random", "hard"):
        m = load_metrics(OUT / "metrics" / f"style_lr_{which}_test.json")
        if m:
            rows.append(row_from_metrics(f"Style-only LR ({which})", m))
    sections.append(table_md(pd.DataFrame(rows)))

    # Seed-42 transformers + tfidf hard/random
    sections.append("## 2. Seed-42 models (reference; not overwritten)\n")
    for which in ("hard", "random"):
        sections.append(f"### {which} split\n")
        rows = []
        for mname in TRANSFORMERS:
            m = load_metrics(OUT / "metrics" / f"{mname}_{which}_seed42_test.json")
            if m:
                rows.append(row_from_metrics(mname, m))
        for slug, label in [
            ("TF-IDF_plus_MultinomialNB", "TF-IDF + MultinomialNB"),
            ("TF-IDF_plus_LogisticRegression", "TF-IDF + LogisticRegression"),
            ("TF-IDF_plus_LinearSVC", "TF-IDF + LinearSVC"),
        ]:
            m = load_metrics(OUT / "metrics" / f"{slug}_{which}_deterministic_test.json")
            if m:
                rows.append(row_from_metrics(label, m))
        sections.append(table_md(pd.DataFrame(rows)))

    # Suspicious
    sections.append("## 3. Suspicious-result checks\n")
    sus = OUT / "suspicious" / "report.json"
    if sus.exists():
        d = json.loads(sus.read_text())
        sections.append("```json\n" + json.dumps(d, indent=2)[:4000] + "\n```\n")
        mm = d.get("muril_vs_mbert", {})
        if mm.get("n_predictions_differ", 1) == 0 and not mm.get("identical_files"):
            surprises.append("MuRIL and mBERT hard predictions are identical despite different files.")
        if mm.get("identical_files"):
            surprises.append("CRITICAL: MuRIL and mBERT prediction files have identical SHA256.")
        x = d.get("xlm_roberta_hard", {})
        if x.get("share_real_predicted_fake", 0) and x["share_real_predicted_fake"] > 0.5:
            surprises.append(
                f"XLM-RoBERTa predicts FAKE for {x['share_real_predicted_fake']:.1%} of REAL hard-test headlines."
            )

    # XLM-R tuned
    sections.append("## 3b. XLM-RoBERTa extra run (lr=1e-5, warmup=0.1, epochs=6)\n")
    m = load_metrics(OUT / "metrics" / "xlm-roberta_hard_seed42_tuned_test.json")
    if m:
        sections.append(table_md(pd.DataFrame([row_from_metrics("XLM-R tuned (hard)", m)])))
    else:
        sections.append("_Not finished yet._\n")

    # Multiseed
    sections.append("## 4. Multi-seed mean ± std (seeds 1–5)\n")
    agg = OUT / "multiseed" / "transformer_mean_std.csv"
    if agg.exists() and agg.stat().st_size > 10:
        try:
            df = pd.read_csv(agg)
            if len(df):
                sections.append(table_md(df))
            else:
                sections.append("_Pending multiseed completion (empty aggregate)._ \n")
        except Exception:
            sections.append("_Pending multiseed completion._\n")
    else:
        sections.append("_Pending multiseed completion._\n")

    # Bootstrap
    sections.append("## 5. Hard-split 95% bootstrap CIs\n")
    bpath = OUT / "bootstrap" / "hard_ci95_summary.csv"
    if bpath.exists():
        sections.append(table_md(pd.read_csv(bpath)))

    # Error analysis
    sections.append("## 6. Error analysis (hard)\n")
    ea = OUT / "error_analysis" / "summary.json"
    if ea.exists():
        sections.append("```json\n" + ea.read_text()[:3000] + "\n```\n")
        sections.append("False +/- CSVs under `eval_results_1400/error_analysis/`.\n")

    # Data stats
    sections.append("## 7. Data / tokenizer stats\n")
    ts = OUT / "data_stats" / "tokenizer_stats.csv"
    if ts.exists():
        sections.append(table_md(pd.read_csv(ts)))
    nd = OUT / "data_stats" / "near_duplicates_meta.json"
    if nd.exists():
        sections.append("Near-duplicates: `" + nd.read_text() + "`\n")
    dr = OUT / "data_stats" / "date_range.json"
    if dr.exists():
        sections.append("Dates: `" + dr.read_text() + "`\n")

    # Romanisation
    sections.append("## 8. Romanisation OOD (no retrain)\n")
    rs = OUT / "romanisation" / "summary.csv"
    if rs.exists():
        sections.append(table_md(pd.read_csv(rs)))

    # HF
    sections.append("## 9. Hugging Face export\n")
    sections.append("Files in `eval_results_1400/huggingface/` (CSV/Parquet + README.md).\n")

    sections.append("## Surprises / things that looked wrong\n")
    if not surprises:
        surprises.append("See XLM-R hard Acc collapse vs random 100%; check MuRIL≈mBERT agreement rates.")
    for s in surprises:
        sections.append(f"- {s}\n")

    out = OUT / "RESULTS.md"
    out.write_text("".join(sections))
    print("Wrote", out)


if __name__ == "__main__":
    main()
