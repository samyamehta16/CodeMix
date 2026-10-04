"""
Fine-tune MuRIL or IndicBERT as a TOKEN-CLASSIFICATION model on the
Gemini-annotated Tamil-English LID dataset (data/lid_dataset/*.conll).

This is the step that was originally planned but blocked on (a) a labeled
LID dataset and (b) GPU time - both of which now exist: (a) is the output
of src/run_annotation.py + src/convert_to_lid_format.py, and (b) just
needs a machine with a GPU to actually execute this script.

*** This script is written and ready to run, but is NOT executed as part
*** of the annotation pipeline - it needs a GPU and the extra dependencies
*** in requirements-train.txt. See docs/03_finetuning_muril_indicbert.md.

Setup (on a GPU machine):
    pip install -r requirements-train.txt

Run:
    python src/train_lid_tagger.py --model muril
    python src/train_lid_tagger.py --model indicbert
    python src/train_lid_tagger.py --model muril --epochs 10 --batch-size 8

Output:
    models/<model_key>/           - fine-tuned model + tokenizer (HF format)
    models/<model_key>/test_metrics.json - final accuracy/precision/recall/F1 on test.conll
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import LABELS, LID_DATASET_DIR, MODEL_REGISTRY, TRAINED_MODELS_DIR
from lid_dataset_utils import ID2LABEL, LABEL2ID, conll_to_dataset
from utils import setup_logger

logger = setup_logger()


def tokenize_and_align_labels(examples, tokenizer, max_length: int):
    """Standard HF recipe: sub-word tokenize, then only label the FIRST
    sub-token of each original token; everything else (special tokens,
    continuation sub-tokens) gets -100 so the loss/metrics ignore them."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def build_compute_metrics():
    """Flat token-level Accuracy / Precision / Recall / F1 (macro + per-label),
    matching the evaluation metrics named in the project brief. Uses sklearn
    directly on the valid (non -100) token predictions rather than seqeval,
    since these are independent per-token language tags, not BIO entity spans."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels, true_preds = [], []
        for pred_row, label_row in zip(predictions, labels):
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_labels.append(ID2LABEL[l])
                true_preds.append(ID2LABEL[p])

        acc = accuracy_score(true_labels, true_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_preds, average="macro", zero_division=0, labels=LABELS
        )
        per_label_p, per_label_r, per_label_f1, _ = precision_recall_fscore_support(
            true_labels, true_preds, average=None, zero_division=0, labels=LABELS
        )

        metrics = {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }
        for label, p, r, f in zip(LABELS, per_label_p, per_label_r, per_label_f1):
            metrics[f"precision_{label}"] = p
            metrics[f"recall_{label}"] = r
            metrics[f"f1_{label}"] = f
        return metrics

    return compute_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True,
                   help="Which pretrained checkpoint to fine-tune")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=64,
                   help="Comments are short YouTube comments; 64 sub-tokens is generous")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    # Heavy deps imported lazily so `--help` and data-prep work without
    # torch/transformers installed.
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    model_checkpoint = MODEL_REGISTRY[args.model]
    output_dir = Path(args.output_dir) if args.output_dir else TRAINED_MODELS_DIR / args.model

    logger.info(f"Fine-tuning {args.model} ({model_checkpoint}) for Tamil-English token-level LID")
    logger.info(f"CUDA available: {torch.cuda.is_available()}"
                + ("" if torch.cuda.is_available() else " - this will be very slow on CPU, "
                                                          "GPU strongly recommended"))

    for split in ("train", "dev", "test"):
        conll_path = LID_DATASET_DIR / f"{split}.conll"
        if not conll_path.exists():
            logger.error(f"{conll_path} not found. Run src/convert_to_lid_format.py first "
                         f"(after annotating with src/run_annotation.py).")
            sys.exit(1)

    train_ds = conll_to_dataset(LID_DATASET_DIR / "train.conll")
    dev_ds = conll_to_dataset(LID_DATASET_DIR / "dev.conll")
    test_ds = conll_to_dataset(LID_DATASET_DIR / "test.conll")
    logger.info(f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer, args.max_length)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    dev_tok = dev_ds.map(tokenize_fn, batched=True, remove_columns=dev_ds.column_names)
    test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
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
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    logger.info("Evaluating on held-out test.conll ...")
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
