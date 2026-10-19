#!/usr/bin/env python3
"""Check MuRIL vs mBERT prediction identity; audit XLM-R; optional extra XLM-R run is separate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ensure_dirs, load_split  # noqa: E402


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ensure_dirs()
    muril = OUT / "predictions" / "muril_hard_seed42_test.csv"
    mbert = OUT / "predictions" / "mbert_hard_seed42_test.csv"
    xlmr = OUT / "predictions" / "xlm-roberta_hard_seed42_test.csv"
    report = {"checks": []}

    if not muril.exists() or not mbert.exists():
        report["error"] = "Run predict_existing.py first"
        (OUT / "suspicious" / "report.json").write_text(json.dumps(report, indent=2))
        print(report)
        return

    a = pd.read_csv(muril)
    b = pd.read_csv(mbert)
    assert len(a) == len(b) == 382
    # align by id
    merged = a.merge(b, on="id", suffixes=("_muril", "_mbert"))
    n_diff = int((merged["pred_muril"] != merged["pred_mbert"]).sum())
    n_same = int((merged["pred_muril"] == merged["pred_mbert"]).sum())
    sha_m = file_sha256(muril)
    sha_b = file_sha256(mbert)
    same_file = sha_m == sha_b
    report["muril_vs_mbert"] = {
        "n": len(merged),
        "n_predictions_differ": n_diff,
        "n_predictions_same": n_same,
        "agreement_rate": n_same / len(merged),
        "muril_sha256": sha_m,
        "mbert_sha256": sha_b,
        "identical_files": same_file,
        "muril_path": str(muril),
        "mbert_path": str(mbert),
        "conclusion": (
            "SAME FILE BY MISTAKE" if same_file
            else ("SUSPICIOUSLY IDENTICAL PREDS" if n_diff == 0 else "Distinct models with overlapping preds")
        ),
    }
    print("MuRIL vs mBERT: differ=", n_diff, "same_file=", same_file)

    # XLM-R: share of REAL predicted FAKE + train log parse
    if xlmr.exists():
        x = pd.read_csv(xlmr)
        real = x[x["true"] == "REAL"]
        share = float((real["pred"] == "FAKE").mean()) if len(real) else None
        report["xlm_roberta_hard"] = {
            "n_real": int(len(real)),
            "share_real_predicted_fake": share,
            "n_real_pred_fake": int((real["pred"] == "FAKE").sum()),
            "overall_pred_counts": x["pred"].value_counts().to_dict(),
        }
        print("XLM-R REAL→FAKE share:", share)

    # Parse training log for epoch losses if present
    log_path = Path("data/fake_news_1400_hard/_train_xlmr_log.txt")
    epoch_losses = []
    if log_path.exists():
        import re
        text = log_path.read_text(errors="ignore")
        # HuggingFace logs like {'loss': '0.2163', ... 'epoch': '2.5'}
        for m in re.finditer(r"\{'loss':\s*'?([0-9.eE+-]+)'?.*?['\"]epoch['\"]:\s*'?([0-9.]+)'?", text):
            epoch_losses.append({"loss": float(m.group(1)), "epoch": float(m.group(2))})
        # also eval_loss per epoch
        evals = []
        for m in re.finditer(
            r"'eval_loss':\s*'?([0-9.eE+-]+)'?.*?'eval_accuracy':\s*'?([0-9.eE+-]+)'?.*?'epoch':\s*'?([0-9.]+)'?",
            text,
        ):
            evals.append({
                "eval_loss": float(m.group(1)),
                "eval_accuracy": float(m.group(2)),
                "epoch": float(m.group(3)),
            })
        report["xlm_roberta_train_log"] = {
            "log_path": str(log_path),
            "logged_train_loss_points": epoch_losses,
            "logged_eval_points": evals,
        }
    else:
        report["xlm_roberta_train_log"] = {"error": f"missing {log_path}"}

    out = OUT / "suspicious" / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
