#!/usr/bin/env python3
"""Dump predictions for already-trained seed-42 models (does not retrain)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ID2LABEL,
    MODELS_HARD,
    MODELS_RANDOM,
    OUT,
    TRANSFORMERS,
    compute_metrics,
    ensure_dirs,
    load_split,
    save_metrics,
    save_preds,
)


def predict_dir(model_dir: Path, texts: list[str], batch_size: int = 32) -> list[str]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred_ids = logits.argmax(-1).cpu().tolist()
            preds.extend(ID2LABEL[j] for j in pred_ids)
    return preds


def dump_one(model_key: str, which: str):
    model_root = MODELS_HARD if which == "hard" else MODELS_RANDOM
    model_dir = model_root / model_key
    if not (model_dir / "config.json").exists():
        print(f"SKIP missing {model_dir}")
        return None
    test = load_split("test", which)
    preds = predict_dir(model_dir, test["text"].astype(str).tolist())
    m = compute_metrics(test["label"].tolist(), preds)
    tag = f"{model_key}_{which}_seed42"
    save_preds(
        OUT / "predictions" / f"{tag}_test.csv",
        test["id"].tolist(),
        test["label"].tolist(),
        preds,
        extra={"source": test["source"].tolist()},
    )
    save_metrics(OUT / "metrics" / f"{tag}_test.json", {"model": model_key, "split": which, "seed": 42, **m})
    print(tag, {k: m[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "FP", "FN"]})
    return m


def main():
    ensure_dirs()
    for which in ("hard", "random"):
        for m in TRANSFORMERS:
            dump_one(m, which)


if __name__ == "__main__":
    main()
