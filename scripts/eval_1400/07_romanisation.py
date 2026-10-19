#!/usr/bin/env python3
"""
Transliterate hard+random TEST headlines to Roman script and evaluate
already-trained seed-42 models WITHOUT retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    metrics_row,
    save_metrics,
    save_preds,
)


def get_transliterate_fn():
    """Prefer indic-transliteration; fallback simple pass-through with warning."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        def fn(text: str) -> str:
            try:
                return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
            except Exception:
                return text

        return fn, "indic_transliteration.sanscript TAMIL→ITRANS"
    except ImportError:
        pass
    try:
        # ai4bharat IndicXlit if installed
        from ai4bharat.transliteration import XlitEngine  # type: ignore

        eng = XlitEngine("ta", beam_width=4, rescore=False)

        def fn(text: str) -> str:
            try:
                out = eng.translit_sentence(text, lang_code="ta")
                if isinstance(out, dict):
                    # take first candidate
                    return next(iter(out.values()))[0] if out else text
                return str(out)
            except Exception:
                return text

        return fn, "ai4bharat IndicXlit"
    except Exception:
        raise RuntimeError(
            "Install indic-transliteration: pip install indic-transliteration"
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
            preds.extend(ID2LABEL[j] for j in logits.argmax(-1).cpu().tolist())
    return preds


def run_split(which: str, translit_fn, method: str):
    test = load_split("test", which)
    native = test["text"].astype(str).tolist()
    roman = [translit_fn(t) for t in native]
    # save 20 samples
    samples = pd.DataFrame({
        "id": test["id"].head(20),
        "native": native[:20],
        "roman": roman[:20],
        "label": test["label"].head(20),
        "source": test["source"].head(20),
    })
    samples.to_csv(
        OUT / "romanisation" / f"samples_20_{which}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # full transliterated test
    pd.DataFrame({
        "id": test["id"],
        "native": native,
        "roman": roman,
        "label": test["label"],
        "source": test["source"],
    }).to_csv(OUT / "romanisation" / f"test_transliterated_{which}.csv", index=False, encoding="utf-8-sig")

    model_root = MODELS_HARD if which == "hard" else MODELS_RANDOM
    rows = []
    for m in TRANSFORMERS:
        md = model_root / m
        if not (md / "config.json").exists():
            continue
        # native (from existing preds if available)
        native_pred_path = OUT / "predictions" / f"{m}_{which}_seed42_test.csv"
        if native_pred_path.exists():
            ndf = pd.read_csv(native_pred_path)
            m_native = compute_metrics(ndf["true"].tolist(), ndf["pred"].tolist())
        else:
            npred = predict_dir(md, native)
            m_native = compute_metrics(test["label"].tolist(), npred)

        rpred = predict_dir(md, roman)
        m_roman = compute_metrics(test["label"].tolist(), rpred)
        save_preds(
            OUT / "predictions" / f"{m}_{which}_seed42_roman_test.csv",
            test["id"].tolist(),
            test["label"].tolist(),
            rpred,
            extra={"source": test["source"].tolist(), "roman_text": roman},
        )
        save_metrics(
            OUT / "metrics" / f"{m}_{which}_seed42_roman_test.json",
            {
                "model": m,
                "split": which,
                "input": "romanised_tamil",
                "transliteration": method,
                "native_macro_f1": m_native["macro_f1"],
                "native_balanced_accuracy": m_native["balanced_accuracy"],
                **m_roman,
            },
        )
        rows.append({
            "Model": m,
            "Split": which,
            "Native Macro-F1": round(m_native["macro_f1"], 4),
            "Native Bal.Acc": round(m_native["balanced_accuracy"], 4),
            "Roman Macro-F1": round(m_roman["macro_f1"], 4),
            "Roman Bal.Acc": round(m_roman["balanced_accuracy"], 4),
            **{k: metrics_row(m, m_roman)[k] for k in ["Acc", "P_F", "R_F", "F1_F", "FP", "FN"]},
        })
        print(which, m, "native_f1", m_native["macro_f1"], "roman_f1", m_roman["macro_f1"])
    return rows


def main():
    ensure_dirs()
    fn, method = get_transliterate_fn()
    (OUT / "romanisation" / "method.json").write_text(json.dumps({"method": method}, indent=2))
    rows = run_split("hard", fn, method) + run_split("random", fn, method)
    pd.DataFrame(rows).to_csv(OUT / "romanisation" / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
