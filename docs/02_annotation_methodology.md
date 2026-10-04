# Annotation Methodology — LLM-Assisted Token-Level Language ID

## 1. Background and rationale

The original project plan was to fine-tune **IndicBERT** and **MuRIL** —
transformer models pretrained on Indian languages — as token-classification
models that label every token of a Tamil-English comment as Tamil or
English. This is the standard approach for token-level Language
Identification (LID) in code-mixed text.

**Why we didn't fine-tune IndicBERT/MuRIL directly (and what we did
instead):**

Fine-tuning a token-classification model needs two things we don't
currently have:

1. **A labeled Tamil-English LID dataset** to fine-tune on (e.g. an
   existing corpus from LinCE or DravidianLangTech, or manually annotated
   data of our own).
2. **GPU compute and time** to actually run the fine-tuning.

Rather than blocking the whole project on acquiring both of those, we use
**Gemini (Google's LLM, via the free-tier API)** as an automatic
*silver-label annotator*: it performs the same token-level TA/EN/MIX/NE/O
tagging task directly via prompting, with no training step required. This
is a documented and defensible workaround in low-resource NLP, generally
called **LLM-assisted / silver-label annotation** — it is not a substitute
for a properly trained tagger, but it:

- unblocks the rest of the pipeline (aggression detection, evaluation)
  immediately without GPU access,
- produces a **labeled dataset as a side effect**, which is exactly what
  would be needed to fine-tune MuRIL/IndicBERT properly later (this closes
  the loop back to the original plan — see Section 6).

This tradeoff (and the option to fine-tune on LinCE/DravidianLangTech data
instead) should be confirmed with your supervisor as the final documented
design decision, per the project brief.

## 2. Pipeline overview

```
data/raw/youtube_comments.csv
        │  (src/preprocess.py)
        ▼
data/processed/cleaned_comments.csv        [Step 1: clean]
        │  (src/run_annotation.py -> src/gemini_annotator.py)
        ▼
data/annotated/annotated_comments.jsonl    [Step 2: tokenize + LID via Gemini]
data/annotated/annotated_comments.csv      (flat, human-readable export)
        │  (src/review_sample.py)
        ▼
data/annotated/review_sample.xlsx          [Step 3: human quality check]
        │  (src/convert_to_lid_format.py)
        ▼
data/lid_dataset/{train,dev,test}.conll    [Step 4: fine-tuning-ready format]
```

## 3. Step 1 — Preprocessing (`src/preprocess.py`)

Cleans the raw comment CSV while preserving Tamil-English code-mixed
content (no lowercasing, no stripping of Tamil Unicode):

- Drops exact duplicate `comment_id`s.
- Drops duplicate comment text (spam / copy-paste comments).
- Strips URLs, collapses whitespace, collapses excessive repeated
  characters (`"soooo"` → `"soo"`) to reduce noise.
- Optionally strips emojis (`--keep-emoji false`; kept by default).
- Drops empty, punctuation-only, or single-character comments, and
  comments with no Latin or Tamil characters at all.

**Result for this run:** 1000 raw comments → **862 cleaned comments**
(138 dropped: duplicates + 49 low-quality/empty rows).

## 4. Step 2 — Automatic token-level language annotation (`src/run_annotation.py` + `src/gemini_annotator.py`)

For each cleaned comment, the pipeline sends a structured prompt
(`prompts/annotation_prompt.txt`) to the Gemini API asking it to:

1. **Tokenize** the comment (split on whitespace/punctuation, keeping
   emojis and hashtags as their own tokens).
2. **Label every token** with exactly one of:

   | Label | Meaning |
   |-------|---------|
   | `TA`  | Tamil word — Tamil script **or** romanized/transliterated (e.g. `semma`, `da`, `irukku`) |
   | `EN`  | English word |
   | `MIX` | Single token blending Tamil + English morphology (e.g. `loveah`, `kekkalam`) |
   | `NE`  | Named entity — person/place/brand/title — language-neutral |
   | `O`   | Numbers, emojis, punctuation, hashtags, @mentions, URLs, symbols |

   Example: `"Bro semma movie da"` → `Bro/EN semma/TA movie/EN da/TA`

3. Gemini returns strict JSON (`{"tokens": [{"token": ..., "label": ...}, ...]}`)
   which is parsed and validated (any token with a missing/invalid label
   is dropped rather than silently trusted).

**Engineering details:**

- **Model:** configurable via `.env` (`GEMINI_MODEL`). `gemini-2.0-flash`
  was tried first but returned `429 ... limit: 0` (no free-tier quota left
  for that model on this key); `gemini-flash-lite-latest` was confirmed
  working and is what this run actually used.
- **Rate limiting:** `REQUESTS_PER_MINUTE` in `.env` throttles calls to stay
  under the free-tier quota (default: 12/min, conservative).
- **Resumable / checkpointed:** every annotated comment is appended to
  `data/annotated/annotated_comments.jsonl` immediately after it's
  produced. If the run is interrupted (quota exhaustion, crash, Ctrl+C),
  re-running `run_annotation.py` skips `comment_id`s already present in the
  checkpoint file — safe to run in multiple sessions over multiple days.
- **Retries:** transient API failures are retried with exponential backoff
  (up to 3 attempts) before a comment is marked `annotation_failed: true`
  and excluded from downstream steps.
- **Outputs:**
  - `annotated_comments.jsonl` — one JSON record per comment (the
    authoritative checkpoint/source of truth).
  - `annotated_comments.csv` — flat, spreadsheet-friendly export
    (`token/LABEL token/LABEL ...` per comment) for quick viewing.

## 5. Step 3 — Human quality review (`src/review_sample.py`)

Because Gemini's labels are **silver labels, not gold**, a random sample
(default 50, seeded for reproducibility) of successfully annotated
comments is exported to `data/annotated/review_sample.xlsx` with empty
`is_correct (Y/N)`, `corrected_annotation`, and `notes` columns for a human
reviewer (ideally a Tamil speaker) to fill in.

This produces a **measured agreement/accuracy rate** that should be
reported alongside the dataset — e.g. *"Gemini's token-level LID labels
agreed with human review on X% of Y sampled tokens/comments"* — rather than
assuming the LLM's labels are correct by default. This is the key
methodological safeguard that makes an LLM-annotated dataset defensible in
an academic report.

## 6. Step 4 — CoNLL export for future fine-tuning (`src/convert_to_lid_format.py`)

Converts the validated annotations into a standard **CoNLL-style**
token-tagging format (one `token<TAB>label` per line, blank line between
sentences) and splits into `train` (80%) / `dev` (10%) / `test` (10%) sets
under `data/lid_dataset/`.

This format is a **direct drop-in** for HuggingFace
`AutoModelForTokenClassification` fine-tuning — meaning this
Gemini-annotated corpus becomes the training data for actually fine-tuning
MuRIL/IndicBERT as a proper token-level LID tagger later, once GPU access
is available. This is how the LLM-assisted shortcut still arrives at the
original project goal, just in a different order (annotate first with an
LLM → train the lightweight tagger second, instead of needing a
pre-existing labeled dataset to train the tagger first).

## 7. What comes after language annotation (next phase, not yet run)

Once the LID-annotated corpus exists, the next phase (per the project
brief) is:

1. Use the language-annotation output (code-switching patterns per
   comment) as an input feature / preprocessing signal.
2. Classify each **comment as a whole** into aggression/harm categories
   (e.g. Normal, Offensive, Aggressive, Hate Speech, Toxic) using
   transformer models and LLMs.
3. Compare model performance (IndicBERT, MuRIL, selected LLMs) using
   Accuracy, Precision, Recall, and F1-score to identify the best-performing
   approach for Tamil-English code-mixed aggression detection.

This is intentionally out of scope for the current run and will be scoped
separately once you've reviewed the annotated LID output.

## 8. Honesty / limitations to state explicitly in your report

- Labels are **LLM-generated silver labels**, not human gold labels — cite
  the review-sample agreement rate once available.
- A single LLM (Gemini) was used as the sole annotator; no
  inter-annotator agreement between multiple models/humans was computed
  beyond the spot-check sample. If time allows, running a second LLM (or a
  second human reviewer) on the same sample would strengthen the validity
  claim.
- The dataset originates from a single YouTube video's comment section
  (see `01_data_collection.md`), so language-mixing patterns may not
  generalize across content genres/audiences.
