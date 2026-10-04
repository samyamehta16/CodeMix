"""
Fine-tune MuRIL or IndicBERT as a SEQUENCE-classification model
(headline text -> REAL/FAKE) on data/fake_news/{train,dev,test}.csv.

This is the "baseline classifier" comparison point promised by the
project title ("...using IndicBERT, MuRIL and LLMs") - a classic
fine-tuned transformer classifier, evaluated against the same test
headlines your evidence-aware LLM pipeline (src/verification/) would
produce a verdict for.

Reuses the same MODEL_REGISTRY as src/train_lid_tagger.py (the LID
tagger) - same checkpoints, different task head
(AutoModelForSequenceClassification vs AutoModelForTokenClassification).

*** Written and ready, but NOT executed here - needs GPU + the deps in
*** requirements-train.txt. See docs/05_fake_news_baseline_classifiers.md.

Setup (on a GPU machine):
    pip install -r requirements.txt
    pip install -r requirements-train.txt

Run:
    python src/fake_news_classifier/train_classifier.py --model muril
    python src/fake_news_classifier/train_classifier.py --model indicbert

Output:
    models_fake_news/<model_key>/               - fine-tuned model + tokenizer
    models_fake_news/<model_key>/test_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FAKE_NEWS_DATA_DIR, FAKE_NEWS_LABELS, FAKE_NEWS_MODELS_DIR, MODEL_REGISTRY
from utils import setup_logger

logger = setup_logger("fake_news_classifier")

LABEL2ID = {label: i for i, label in enumerate(FAKE_NEWS_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(FAKE_NEWS_LABELS)}


def build_compute_metrics():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=LABEL2ID["FAKE"], zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "accuracy": acc,
            "precision_fake": precision, "recall_fake": recall, "f1_fake": f1,
            "precision_macro": precision_m, "recall_macro": recall_m, "f1_macro": f1_m,
        }

    return compute_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=64, help="Headlines are short; 64 sub-tokens is generous")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    model_checkpoint = MODEL_REGISTRY[args.model]
    output_dir = Path(args.output_dir) if args.output_dir else FAKE_NEWS_MODELS_DIR / args.model

    logger.info(f"Fine-tuning {args.model} ({model_checkpoint}) as a REAL/FAKE headline classifier")
    logger.info(f"CUDA available: {torch.cuda.is_available()}"
                + ("" if torch.cuda.is_available() else " - this will be very slow on CPU, GPU strongly recommended"))

    for split in ("train", "dev", "test"):
        csv_path = FAKE_NEWS_DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            logger.error(f"{csv_path} not found. Run src/fake_news_classifier/preprocess.py first.")
            sys.exit(1)

    def load_split(name: str) -> Dataset:
        df = pd.read_csv(FAKE_NEWS_DATA_DIR / f"{name}.csv")
        df["labels"] = df["label"].map(LABEL2ID)
        return Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)

    train_ds, dev_ds, test_ds = load_split("train"), load_split("dev"), load_split("test")
    logger.info(f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)} headlines")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    dev_tok = dev_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(FAKE_NEWS_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    compute_metrics = build_compute_metrics()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=data_collator,
        # transformers>=4.46 renamed this from `tokenizer=` to `processing_class=`
        # (transformers 5.x removed `tokenizer=` entirely).
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    logger.info("Evaluating on held-out test.csv ...")
    test_metrics = trainer.evaluate(test_tok, metric_key_prefix="test")
    logger.info(json.dumps(test_metrics, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Saved fine-tuned model + tokenizer + test_metrics.json -> {output_dir}")


if __name__ == "__main__":
    main()
