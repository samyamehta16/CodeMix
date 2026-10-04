# Fake News Baseline Classifiers — MuRIL / IndicBERT on Tamil Headlines

## 1. Where this fits

Your project title is *"Evidence-Aware Fake News Detection ... using
IndicBERT, MuRIL and LLMs"* — this is the **IndicBERT/MuRIL half** of that
comparison: a classic fine-tuned transformer text classifier
(`headline → REAL/FAKE`), trained on a labeled dataset, evaluated with
Accuracy/Precision/Recall/F1, and meant to be compared against your
evidence-aware LLM pipeline (`src/verification/`) on the same test
examples.

This is a **different, complementary use of MuRIL/IndicBERT** from
`src/train_lid_tagger.py` (which fine-tunes them as *token*-classification
LID taggers on your YouTube-comment dataset). Same checkpoints, same
`MODEL_REGISTRY` in `config.py`, different task head
(`AutoModelForSequenceClassification` here vs
`AutoModelForTokenClassification` there).

## 2. The dataset: `data/raw/Tamil-News-Headlines.csv`

5,226 rows, columns: `English Version`, `Label` (topic category -
politics/tech/sport/entertainment/misc, **not** the fake/real label),
`News` (Tamil-script headline), `Author`, `Date`, `Authenticity` (0/1).

I inspected the actual content to determine what `Authenticity` means:

- **`Authenticity=0`** rows read like real tech/general news headlines,
  dated around 2022 (e.g. *"Redmi A1+ smartphone launched in India..."*).
- **`Authenticity=1`** rows read like Tamil translations of well-known
  fact-checked **false** claims (COVID-19 misinformation, political
  quote claims), dated around 2018-2019 in the source data (e.g. *"Says
  there is no COVID-19 virus"*, *"Says coronavirus is just the damn
  flu"*).

So: **`Authenticity 0 → REAL`, `Authenticity 1 → FAKE`** — this mapping is
hard-coded in `preprocess.py` (`AUTHENTICITY_TO_LABEL`).

### Known limitations to disclose in your report

1. **Source/topic/era mismatch between classes.** REAL and FAKE rows
   don't come from the same claims fact-checked true vs. false — they
   come from visibly different sources, topics, and time periods (2022
   tech news vs. 2018-19 political/health claims). A classifier trained
   on this can partly learn to distinguish *"2022 tech-news writing
   style"* from *"2018 political-claim writing style"* rather than
   "fake-ness" in a general sense. This is a well-known criticism of
   this style of fake-news dataset — state it explicitly rather than
   presenting accuracy as a pure measure of fake-news detection ability.
2. **Domain mismatch with your target application.** `News` is formal,
   pure Tamil-script text — not the romanized Tamil-English code-mixed
   social-media text your project is actually about (the YouTube
   comments). A classifier fine-tuned here may not transfer well to
   code-mixed comments. Worth explicitly testing (e.g. run the trained
   classifier on a few real YouTube comments and see how it does) and
   reporting rather than assuming it generalizes.
3. **48 duplicate headlines** were dropped during preprocessing; final
   counts after cleaning: 5,171 rows (2,893 FAKE / 2,278 REAL).

## 3. What was built

| File | Purpose |
|---|---|
| `src/fake_news_classifier/preprocess.py` | Cleans `Tamil-News-Headlines.csv`, maps `Authenticity` → `REAL`/`FAKE`, dedupes, splits 80/10/10 (stratified) |
| `src/fake_news_classifier/train_classifier.py` | Fine-tunes MuRIL or IndicBERT as a sequence classifier; reports Accuracy/Precision/Recall/F1 |
| `src/fake_news_classifier/compare_classifiers.py` | Side-by-side comparison table once both are trained |

**Already run** (lightweight, no GPU needed): preprocessing is done —
`data/fake_news/{train,dev,test}.csv` exist (4,136 / 517 / 518 rows).

**Not run yet** (needs GPU): the actual fine-tuning. Written and ready,
same as `src/train_lid_tagger.py` before it.

## 4. How to run it (on a GPU machine)

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt

python src/fake_news_classifier/train_classifier.py --model muril
python src/fake_news_classifier/train_classifier.py --model indicbert
python src/fake_news_classifier/compare_classifiers.py
```

Each run saves to `models_fake_news/<model_key>/`: the fine-tuned model +
tokenizer (standard HuggingFace format) and `test_metrics.json`. Useful
flags: `--epochs`, `--batch-size`, `--lr`, `--max-length` (defaults: 4
epochs, batch 16, lr 2e-5, max_length 64 — headlines are short).

To train on the English translations instead of Tamil script (e.g. to
compare Tamil vs English input performance), regenerate the data with:
```bash
python src/fake_news_classifier/preprocess.py --text-column english
```
(this overwrites `data/fake_news/*.csv` — copy the Tamil-based splits
elsewhere first if you want to keep both).

## 5. Design choices

- **Binary classification** (`REAL`/`FAKE`), matching what the dataset
  actually labels — not the full `REAL/FAKE/MISLEADING/UNVERIFIED` scheme
  your LLM pipeline outputs. If you want a like-for-like comparison
  later, you'd need to either collapse the LLM's 4-way verdicts down to
  2 (`MISLEADING`/`UNVERIFIED` → treat as which?) or find/build a
  4-class-labeled dataset — worth deciding deliberately rather than
  silently conflating the two schemes.
- **Metrics**: Accuracy, and both **binary** (positive class = `FAKE`)
  and **macro-averaged** Precision/Recall/F1 — binary numbers tell you
  specifically how good it is at catching fake headlines; macro numbers
  balance both classes equally, useful since the classes aren't perfectly
  balanced (56% FAKE / 44% REAL after cleaning).
- **Same `MODEL_REGISTRY`, same pattern as `train_lid_tagger.py`** — one
  script parameterized by `--model` rather than duplicated per model, so
  any performance difference you report is attributable to the model,
  not an inconsistent pipeline.

## 6. Suggested next step: comparing against the LLM pipeline

Not built yet (bigger task, needs your `SERPER_API_KEY` + costs API
quota per headline) — but once both classifiers are trained, the natural
next step is: run `test.csv`'s headlines through
`src/verification/pipeline.py` too, and compare its verdicts (collapsed
to REAL/FAKE, or kept as 4-way) against the same ground truth. That
produces the actual 3-way comparison (MuRIL vs IndicBERT vs LLM+evidence)
your title promises. This needs a batch runner with checkpointing (same
pattern as `run_annotation.py`) since it's ~500 test headlines × web
search + LLM calls — happy to build this once you're ready, and once
you've decided how to reconcile the 2-class vs 4-class label mismatch
above.
