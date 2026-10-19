"""Shared helpers for the tamil_news_1400 evaluation suite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "tamil_news_dataset_1400_balanced.csv"
RANDOM_DIR = ROOT / "data" / "fake_news_1400"
HARD_DIR = ROOT / "data" / "fake_news_1400_hard"
OUT = ROOT / "eval_results_1400"
MODELS_HARD = ROOT / "models_fake_news_1400_hard"
MODELS_RANDOM = ROOT / "models_fake_news_1400"

TRANSFORMERS = ["indicbert-v2", "muril", "mbert", "xlm-roberta"]
MODEL_CKPT = {
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",
    "muril": "google/muril-base-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
}
LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {0: "REAL", 1: "FAKE"}


def ensure_dirs():
    for p in [
        OUT,
        OUT / "predictions",
        OUT / "metrics",
        OUT / "style_baseline",
        OUT / "suspicious",
        OUT / "multiseed",
        OUT / "bootstrap",
        OUT / "error_analysis",
        OUT / "data_stats",
        OUT / "romanisation",
        OUT / "huggingface",
        OUT / "logs",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df["headline"] = df["headline"].astype(str)
    return df


def attach_ids(split_df: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map cleaned split `text` back to raw `id` via cleaned-headline match."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from utils import clean_comment

    raw = load_raw() if raw is None else raw
    mapping = {}
    for _, row in raw.iterrows():
        cleaned = clean_comment(str(row["headline"]), keep_emoji=False).strip()
        mapping[cleaned] = str(row["id"])
        mapping[str(row["headline"]).strip()] = str(row["id"])
    out = split_df.copy()
    texts = out["text"].astype(str)
    ids = []
    for t in texts:
        tid = mapping.get(t.strip())
        if tid is None:
            key = " ".join(t.split())
            tid = next((mapping[k] for k in mapping if " ".join(k.split()) == key), "UNKNOWN")
        ids.append(tid)
    out["id"] = ids
    n_unk = sum(1 for i in ids if i == "UNKNOWN")
    if n_unk:
        print(f"WARNING: attach_ids unmatched {n_unk}/{len(ids)} rows")
    return out


def load_split(split_name: str, which: str) -> pd.DataFrame:
    """which in {random, hard}; split_name in train/dev/test."""
    data_dir = RANDOM_DIR if which == "random" else HARD_DIR
    df = pd.read_csv(data_dir / f"{split_name}.csv", encoding="utf-8-sig")
    return attach_ids(df)


def compute_metrics(y_true, y_pred) -> dict:
    """y_true/y_pred as REAL/FAKE strings. FAKE = positive."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="FAKE", zero_division=0
    )
    _, _, f1m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "precision_fake": float(p),
        "recall_fake": float(r),
        "f1_fake": float(f1),
        "macro_f1": float(f1m),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n": int(len(y_true)),
    }


def save_preds(path: Path, ids, y_true, y_pred, extra: dict | None = None):
    rows = {"id": list(ids), "true": list(y_true), "pred": list(y_pred)}
    if extra:
        rows.update(extra)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def metrics_row(name: str, m: dict) -> dict:
    return {
        "Model": name,
        "Acc": round(m["accuracy"], 4),
        "Bal.Acc": round(m["balanced_accuracy"], 4),
        "P_F": round(m["precision_fake"], 4),
        "R_F": round(m["recall_fake"], 4),
        "F1_F": round(m["f1_fake"], 4),
        "Macro-F1": round(m["macro_f1"], 4),
        "FP": m["FP"],
        "FN": m["FN"],
    }
