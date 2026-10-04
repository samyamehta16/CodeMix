# Fine-Tuning MuRIL / IndicBERT on the Gemini-Annotated LID Dataset

## 1. Where this fits in the project

This closes the loop described in `02_annotation_methodology.md` Section 6:
Gemini was used as a *silver-label annotator* specifically **because** we
didn't yet have a labeled Tamil-English LID dataset or GPU access to
fine-tune IndicBERT/MuRIL directly. Now that:

- the annotated dataset exists (`data/lid_dataset/{train,dev,test}.conll`,
  produced by `src/convert_to_lid_format.py`), and
- GPU access is available,

we can fine-tune **MuRIL** and **IndicBERT** the way the project originally
intended: as proper token-classification models, trained on this data
rather than prompted per-comment.

**Status: the code below is written and ready, but has not been executed**
(no GPU / heavy ML deps in this environment) — run it yourself on a
GPU machine (Colab, Kaggle, university cluster, etc.).

## 2. What was added

| File | Purpose |
|---|---|
| `requirements-train.txt` | Extra deps (torch, transformers, datasets, scikit-learn, accelerate) — kept separate from `requirements.txt` since they're heavy and only needed for this step |
| `src/lid_dataset_utils.py` | Reads `.conll` files into a HuggingFace `Dataset` |
| `src/train_lid_tagger.py` | Fine-tunes **either** MuRIL or IndicBERT as a token-classification model, evaluates on `test.conll`, saves the model |
| `src/compare_lid_models.py` | Reads the `test_metrics.json` from each trained model and prints/saves a side-by-side comparison table |

`config.py` now also has a `MODEL_REGISTRY` mapping short names to actual
HuggingFace checkpoints:

```python
MODEL_REGISTRY = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",   # newer/stronger, optional
}
```

## 3. How to run it (on a GPU machine)

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt

# make sure the LID dataset exists (regenerate if you've annotated more since)
python src/convert_to_lid_format.py

# fine-tune each model
python src/train_lid_tagger.py --model muril
python src/train_lid_tagger.py --model indicbert

# compare them
python src/compare_lid_models.py
```

Each run saves to `models/<model_key>/`: the fine-tuned model + tokenizer
in standard HuggingFace format (reloadable with
`AutoModelForTokenClassification.from_pretrained("models/muril")`), plus
`test_metrics.json` with the final held-out test scores.

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--max-length` (see
`python src/train_lid_tagger.py --help`). Defaults (8 epochs, batch size
16, lr 2e-5) are reasonable starting points for a small dataset; expect to
tune them once you see real train/dev curves.

## 4. Design choices (for your report / to explain to your professor)

- **Task framing:** this is treated as **flat multi-class token
  classification** (each token gets one of `TA/EN/MIX/NE/O`
  independently), not BIO-tagged span extraction — there's no notion of
  multi-token "entities" to chunk here, each token's language is a
  standalone label. This is why evaluation uses plain per-token
  Accuracy/Precision/Recall/F1 (via scikit-learn) rather than
  entity-level `seqeval` metrics.
- **Sub-word alignment:** MuRIL/IndicBERT tokenize each word into
  sub-word pieces. Following the standard HuggingFace recipe, only the
  **first sub-token** of each original token is given the real label; the
  rest (and special tokens like `[CLS]`/`[SEP]`) get `-100` so they're
  ignored by the loss and by evaluation.
- **Metrics reported:** overall accuracy, macro-averaged
  precision/recall/F1, **and** per-label precision/recall/F1 (so you can
  see e.g. whether `MIX` — the rarest, hardest class — is being learned at
  all, separately from the dominant `TA`/`EN` classes).
- **Train/dev/test split:** 80/10/10 at the sentence (comment) level,
  already produced by `convert_to_lid_format.py` with a fixed random seed
  for reproducibility.
- **Two models, one script:** `train_lid_tagger.py` is parameterized by
  `--model` rather than duplicated per model, so MuRIL and IndicBERT go
  through *identical* data processing/training/evaluation code — any
  performance difference you report is attributable to the model, not to
  inconsistent pipelines.

## 5. Known limitations to disclose

- **Dataset size.** As of the current annotation run, the dataset has
  ~500 annotated comments (~400 train sentences) — small for fine-tuning
  a transformer from scratch-ish; expect it to still work reasonably
  because MuRIL/IndicBERT are *pretrained* (this is fine-tuning, not
  training from scratch), but report this as a limitation and mention
  that completing the remaining annotation (~350 more comments, blocked
  on Gemini free-tier daily quota — see `02_annotation_methodology.md`)
  would strengthen results.
- **Labels are silver, not gold** (see `02_annotation_methodology.md`,
  Section 5) — any tagger trained here inherits Gemini's annotation
  quality/errors. The `review_sample.xlsx` human-checked subset is the
  honest way to bound how much noise that introduces.
- **`indicbert` (v1, `ai4bharat/indic-bert`) is an ALBERT-based model**
  with a different tokenizer/architecture than MuRIL (BERT-based) — if
  results differ substantially, that could reflect architecture as much
  as pretraining data; `indicbert-v2` (`ai4bharat/IndicBERTv2-MLM-only`,
  BERT-based) is included in the registry as a fairer architectural
  comparison if you want a third data point.

## 6. Next step after this

Once both models are fine-tuned and `compare_lid_models.py` has a
side-by-side table, the project moves to the **aggression detection**
phase (comment-level classification into Normal/Offensive/Aggressive/Hate
Speech/etc.) described in `02_annotation_methodology.md` Section 7 — not
yet started.
