"""
Shared helpers for reading the CoNLL-style LID dataset
(data/lid_dataset/{train,dev,test}.conll, produced by convert_to_lid_format.py)
into a HuggingFace `datasets.Dataset` ready for token-classification
fine-tuning.

CoNLL format recap: one "token<TAB>label" per line, blank line = sentence
boundary (see convert_to_lid_format.py).
"""
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset

from config import LABELS

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def read_conll(path: Path) -> List[Tuple[List[str], List[str]]]:
    """Returns a list of (tokens, labels) tuples, one per sentence."""
    sentences = []
    tokens, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            token, label = line.rsplit("\t", 1)
            tokens.append(token)
            labels.append(label)
    if tokens:
        sentences.append((tokens, labels))
    return sentences


def conll_to_dataset(path: Path) -> Dataset:
    """Loads a .conll file into a HF Dataset with columns:
    - tokens: list[str] (raw tokens, one sentence per row)
    - ner_tags: list[int] (label ids, aligned with `tokens`)
    """
    sentences = read_conll(path)
    if not sentences:
        raise ValueError(f"No sentences found in {path} - did you run "
                          f"convert_to_lid_format.py yet?")

    tokens_col = [toks for toks, _ in sentences]
    tags_col = [[LABEL2ID[label] for label in labels] for _, labels in sentences]

    return Dataset.from_dict({"tokens": tokens_col, "ner_tags": tags_col})
