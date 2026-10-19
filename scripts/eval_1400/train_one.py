#!/usr/bin/env python3
"""
Train one transformer run with explicit seed; select best by DEV macro-F1 only.
Saves predictions CSV + metrics JSON. Does not overwrite seed-42 model dirs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HARD_DIR,
    ID2LABEL,
    LABEL2ID,
    MODEL_CKPT,
    OUT,
    RANDOM_DIR,
    compute_metrics,
    ensure_dirs,
    load_split,
    save_metrics,
    save_preds,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CKPT))
    p.add_argument("--split", required=True, choices=["random", "hard"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--tag", type=str, default="", help="optional suffix e.g. xlmr_tuned")
    args = p.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed as hf_set_seed,
    )

    hf_set_seed(args.seed)

    data_dir = RANDOM_DIR if args.split == "random" else HARD_DIR
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    dev_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")
    test_df = load_split("test", args.split)

    def to_ds(df):
        d = df.copy()
        d["labels"] = d["label"].map(LABEL2ID)
        return Dataset.from_pandas(d[["text", "labels"]], preserve_index=False)

    train_ds, dev_ds = to_ds(train_df), to_ds(dev_df)
    test_ds = Dataset.from_pandas(
        pd.DataFrame({"text": test_df["text"], "labels": test_df["label"].map(LABEL2ID)}),
        preserve_index=False,
    )

    ckpt = MODEL_CKPT[args.model]
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    dev_tok = dev_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics_hf(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=LABEL2ID["FAKE"], zero_division=0
        )
        _, _, f1m, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {"accuracy": acc, "f1_fake": f1, "f1_macro": f1m, "precision_fake": p, "recall_fake": r}

    tag = args.tag or f"{args.model}_{args.split}_seed{args.seed}"
    run_dir = OUT / "multiseed" / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5.x dropped warmup_ratio; use warmup_steps instead
    steps_per_epoch = max(1, (len(train_tok) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics_hf,
    )
    train_out = trainer.train()
    # DEV metrics for the loaded best checkpoint
    dev_metrics = trainer.evaluate(dev_tok, metric_key_prefix="dev")
    # TEST once at end
    pred_out = trainer.predict(test_tok)
    pred_ids = np.argmax(pred_out.predictions, axis=1)
    pred_labels = [ID2LABEL[i] for i in pred_ids]
    y_true = test_df["label"].tolist()
    m = compute_metrics(y_true, pred_labels)

    save_preds(
        OUT / "predictions" / f"{tag}_test.csv",
        test_df["id"].tolist(),
        y_true,
        pred_labels,
        extra={"source": test_df["source"].tolist()},
    )
    payload = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "checkpoint_selection": "best_dev_macro_f1",
        "train_loss": float(train_out.training_loss) if train_out else None,
        "dev_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in dev_metrics.items()},
        **m,
    }
    save_metrics(OUT / "metrics" / f"{tag}_test.json", payload)
    # keep a light copy of trainer state log history
    hist = trainer.state.log_history if trainer.state else []
    (run_dir / "log_history.json").write_text(json.dumps(hist, indent=2, default=str))
    print(json.dumps({k: payload[k] for k in ["model", "split", "seed", "accuracy", "macro_f1", "balanced_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
