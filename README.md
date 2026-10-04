# Tamil-English Code-Mixed NLP Project

**Current focus (active development): Evidence-Aware Fake News Detection
for Tamil-English Code-Mixed Social Media using IndicBERT, MuRIL and
LLMs.** Two complementary pieces:

- `src/verification/` — the evidence-aware LLM pipeline (claim → search →
  evidence → verdict). See `docs/04_fake_news_verification_pipeline.md`.
- `src/fake_news_classifier/` — fine-tuned MuRIL/IndicBERT baseline
  classifiers (headline → REAL/FAKE) trained on
  `data/raw/Tamil-News-Headlines.csv`, to compare against the LLM
  pipeline. See `docs/05_fake_news_baseline_classifiers.md`.

**Both are now trained/working** — see
`docs/06_technical_summary_for_presentation.md` for a full write-up:
architecture, results (IndicBERTv2 99.0% / MuRIL 98.6% test accuracy),
a live end-to-end example of the LLM pipeline, known limitations, and
next steps.

The sections below describe the **original LID annotation pipeline**,
which still exists and works, but is no longer the primary track of the
project.

---

## LID annotation pipeline (earlier phase)

Automatic token-level language annotation of Tamil-English code-mixed
YouTube comments, using the Gemini API as a **silver-label generator**
because we don't currently have a GPU or an existing Tamil-English LID
dataset to fine-tune IndicBERT / MuRIL from scratch.

**The output of this pipeline is itself a labeled dataset** — once you have
enough Gemini-annotated + human-reviewed data, it becomes the training set
for actually fine-tuning MuRIL/IndicBERT as a proper tagger later. That's
the "probably use this dataset to fine-tune MuRIL" step your professor
mentioned.

## Why this approach (for your report / to explain to your professor)

- Original plan: fine-tune IndicBERT/MuRIL directly for token-level LID.
- Blocker: fine-tuning needs (a) a labeled LID dataset and (b) GPU time —
  we have neither right now.
- Workaround: use an LLM (Gemini, free-tier API) as an annotator to
  auto-generate token-level TA/EN/MIX/NE/O labels. This is a documented,
  common workaround in low-resource NLP ("LLM-assisted / silver-label
  annotation") — not a replacement for a trained tagger, just a way to
  bootstrap one.
- Quality control: `review_sample.py` exports a random subset for a human
  (ideally a Tamil speaker) to check, so you can report an estimated
  accuracy/agreement figure alongside the dataset rather than assuming
  Gemini is always right.
- Payoff: once this dataset exists, `convert_to_lid_format.py` turns it
  into a standard CoNLL-style token-tagging format that's a direct drop-in
  for a HuggingFace token-classification fine-tune of MuRIL/IndicBERT —
  the original goal, just sequenced differently.

## Project structure

```
tamil_english_annotation/
├── .env.example              # copy to .env, add your Gemini API key
├── requirements.txt
├── prompts/
│   └── annotation_prompt.txt # the exact prompt sent to Gemini (editable)
├── data/
│   ├── raw/                  # put your input CSV here
│   ├── processed/            # cleaned comments
│   ├── annotated/            # Gemini output (checkpointed JSONL + CSV + review sample)
│   └── lid_dataset/          # final train/dev/test.conll for fine-tuning
└── src/
    ├── config.py             # paths, label set, rate-limit settings
    ├── utils.py               # text cleaning helpers
    ├── preprocess.py          # Step 1
    ├── gemini_annotator.py    # Gemini API wrapper (prompting, parsing, retries)
    ├── run_annotation.py      # Step 2 (resumable/checkpointed)
    ├── convert_to_lid_format.py # Step 3
    └── review_sample.py       # Step 4 (quality check)
```

## Label set

| Label | Meaning |
|---|---|
| `TA`  | Tamil word (Tamil script or romanized, e.g. "semma", "da") |
| `EN`  | English word |
| `MIX` | Single token blending Tamil+English morphology (e.g. "loveah") |
| `NE`  | Named entity (person/place/brand/title) — language-neutral |
| `O`   | Numbers, emojis, punctuation, hashtags, mentions, URLs |

Loosely follows the LinCE / DravidianLangTech LID tagset conventions so
the output is directly comparable/reusable with existing Tamil-English
code-mixing research.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your Gemini API key
```

Get a free Gemini API key at https://aistudio.google.com/apikey — check
your current free-tier rate limit there and set `REQUESTS_PER_MINUTE` in
`.env` accordingly (leave it conservative; the default of 12/min is safe
for most free tiers as of early 2026, but confirm before a long run).

> Note: `gemini-2.0-flash`'s free tier has since dropped to a 0 quota for
> some API keys/projects (you'll see a `429 ... limit: 0` error). If that
> happens, run `python -c "import google.generativeai as g; g.configure(api_key='YOUR_KEY'); [print(m.name) for m in g.list_models() if 'generateContent' in m.supported_generation_methods]"`
> to see what your key currently has access to, and set `GEMINI_MODEL` in
> `.env` to a model that still has free quota — `gemini-flash-lite-latest`
> and `gemini-flash-latest` were confirmed working as of this run.

Put your raw CSV at `data/raw/youtube_comments.csv` with columns:
`video_id, comment_id, text, author, like_count, published_at`

## Run the pipeline

```bash
# 1. Clean the raw comments (dedup, strip URLs, drop empty/low-quality)
python src/preprocess.py

# 2. Annotate with Gemini (resumable — safe to Ctrl+C and re-run;
#    already-annotated comment_ids are skipped)
python src/run_annotation.py --limit 20   # test on a small batch first
python src/run_annotation.py              # then run the full ~1000

# 3. Build a human-checkable random sample
python src/review_sample.py --n 50
# -> open data/annotated/review_sample.xlsx, have someone fill in
#    is_correct(Y/N) and corrected_annotation for each row

# 4. Flatten to a simple per-token CSV (comment_id, token, label)
python src/tokens_to_csv.py
# -> data/annotated/tokens_flat.csv

# 5. Convert to a CoNLL-style LID dataset for future fine-tuning
python src/convert_to_lid_format.py
```

## Outputs to show your professor

- `data/processed/cleaned_comments.csv` — cleaned input data
- `data/annotated/annotated_comments.csv` — flat, readable annotation export
- `data/annotated/annotated_comments.jsonl` — full structured annotations (the checkpoint file)
- `data/annotated/review_sample.xlsx` — human-reviewed accuracy check
- `data/annotated/tokens_flat.csv` — one row per token (`comment_id, video_id, token_index, token, label`)
- `data/lid_dataset/{train,dev,test}.conll` — ready-to-use fine-tuning dataset

## Fine-tuning MuRIL / IndicBERT (next stage, code ready, not run here)

Now that the annotated LID dataset exists, `src/train_lid_tagger.py` +
`src/compare_lid_models.py` fine-tune MuRIL and IndicBERT as proper
token-classification taggers on it and compare them — this is written and
ready but **not executed** in this environment (needs GPU + the extra
deps in `requirements-train.txt`). See
`docs/03_finetuning_muril_indicbert.md` for full details and run
instructions.

## Notes / limitations to disclose

- Gemini labels are **silver labels**, not gold — hence the mandatory
  review-sample step. Report the measured agreement rate honestly.
- Free-tier rate limits mean ~1000 comments will take a while
  (`run_annotation.py` prints an ETA before starting); the checkpointing
  means you can run it in chunks across sessions.
- If a comment fails annotation after retries (`annotation_failed: true`
  in the JSONL), it's excluded from the CoNLL export automatically —
  check the count in the logs.
- Next step after this: fine-tune MuRIL/IndicBERT as a token-classification
  model on `data/lid_dataset/` (standard HuggingFace `Trainer` +
  `AutoModelForTokenClassification`) once you have GPU access — this
  dataset is what makes that step possible without manual annotation.
