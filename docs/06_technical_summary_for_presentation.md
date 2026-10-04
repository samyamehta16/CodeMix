# Technical Summary — Evidence-Aware Fake News Detection for Tamil-English Code-Mixed Social Media using IndicBERT, MuRIL and LLMs

*Prepared as a presentation/reference document — covers what has been built, why, how it works, and the results so far.*

---

## 1. Problem statement

Fake news / misinformation on Tamil social media is hard to detect automatically for two compounding reasons:

1. **Code-mixing**: comments and posts mix Tamil script, romanized Tamil, and English in the same sentence (e.g. *"RBI 5000 rupees note release pannitanga nu news la varudhu"*). Most off-the-shelf NLP tools and fake-news datasets assume monolingual text.
2. **Fact-checking needs evidence, not just style.** A claim is fake or real because of what's actually true in the world *right now*, not because of some fixed lexical pattern in a training set. A classifier trained once on old headlines can't know about a rumor that started yesterday.

This project attacks the problem **two ways in parallel**, so they can be directly compared:

| | Approach | Core idea |
|---|---|---|
| **A** | Fine-tuned baseline classifiers (MuRIL, IndicBERT) | Learn statistical patterns of fake vs. real headlines from a labeled dataset |
| **B** | Evidence-aware LLM pipeline | Don't rely on memorized patterns — extract the claim, search the live web, retrieve real evidence, and let an LLM judge the claim *against that evidence* |

Approach A is the classic supervised-learning baseline. Approach B is the actual novel contribution: it can in principle catch a brand-new rumor Approach A has never seen, because it looks things up instead of memorizing.

---

## 2. Approach A — Fine-tuned baseline classifiers

### 2.1 Dataset

`data/raw/Tamil-News-Headlines.csv` — 5,226 headlines with columns `News` (Tamil script), `English Version`, `Authenticity` (0/1), plus topic label, author, date.

- `Authenticity=0` → **REAL** (e.g. real 2022 tech/general news)
- `Authenticity=1` → **FAKE** (Tamil translations of known fact-checked false claims — COVID-19 misinformation, political quote claims, from ~2018-19)

After cleaning (Tamil-script normalization via the shared `clean_comment()` utility, dropping 48 duplicate headlines, dropping too-short rows): **5,171 headlines** → split **80/10/10 stratified** → `train.csv` (4,136) / `dev.csv` (517) / `test.csv` (518).

**Important, disclosable limitation**: REAL and FAKE rows come from visibly different sources/topics/eras (2022 tech news vs. 2018-19 political/health claims) rather than the same claims fact-checked both ways. A classifier can partly learn *"2022 tech-news style" vs "2018 political-rumor style"* rather than "fake-ness" in general — this is a known criticism of this style of fake-news dataset and should be stated explicitly rather than presenting the accuracy numbers below as a pure, generalizable measure of fake-news-detection ability. It is also **pure Tamil-script text**, not the romanized Tamil-English code-mixed text this project is ultimately about — a real generalization test (e.g. running these classifiers on actual YouTube comments) has not been done yet.

### 2.2 Models & training

- **MuRIL** (`google/muril-base-cased`) — Google's BERT-base model pretrained on 17 Indian languages + English (transliterated forms included), so it should natively understand Tanglish-style text.
- **IndicBERT v2** (`ai4bharat/IndicBERTv2-MLM-only`) — AI4Bharat's BERT-base model pretrained specifically on Indic languages.
  (Note: the original `ai4bharat/indic-bert`, the smaller ALBERT-based v1, is now a **gated** HuggingFace repo requiring manual approval — v2 was used instead since it's open and is AI4Bharat's current recommended release.)

Both fine-tuned identically for a fair comparison: `AutoModelForSequenceClassification`, 2 labels (REAL/FAKE), 4 epochs, batch size 16, learning rate 2e-5, max sequence length 64 sub-tokens (headlines are short), best checkpoint selected by macro-F1 on the dev set, final evaluation on the untouched test set. Trained **on CPU** (no GPU available) via `src/fake_news_classifier/train_classifier.py --model {muril|indicbert-v2}`.

### 2.3 Results (test set, 518 headlines)

| Model | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) | F1 (macro) |
|---|---|---|---|---|---|
| **IndicBERT v2** | **99.03%** | 99.65% | 98.62% | 99.13% | **99.02%** |
| **MuRIL** | 98.65% | 98.96% | 98.62% | 98.79% | 98.63% |

Both models score extremely high — consistent with the dataset limitation above: since REAL and FAKE come from visibly different sources/eras, the model has strong surface cues to latch onto, so this should be presented as *"near-ceiling performance on this particular dataset's style gap"*, not as *"99% real-world fake-news detection accuracy."*

### 2.4 Code

| File | Role |
|---|---|
| `src/fake_news_classifier/preprocess.py` | Cleans the raw CSV, maps `Authenticity` → REAL/FAKE, dedupes, stratified 80/10/10 split |
| `src/fake_news_classifier/train_classifier.py` | Fine-tunes MuRIL or IndicBERT (`--model` flag selects which); reports Accuracy/Precision/Recall/F1 on the held-out test set |
| `src/fake_news_classifier/compare_classifiers.py` | Loads both models' `test_metrics.json` and produces the side-by-side table above (`models_fake_news/classifier_comparison.csv`) |

---

## 3. Approach B — Evidence-aware LLM verification pipeline

This is the core novel piece. Instead of a single model guessing REAL/FAKE from text alone, the claim is checked against real, freshly-retrieved web evidence, and an LLM makes the final call **only from that evidence** — never from its own memorized "knowledge."

### 3.1 Pipeline, end to end

```
Tamil-English comment
        │  clean_comment()
        ▼
[1] claim_extraction.py    ── LLM: does this contain a factual claim? normalize it to one clean English sentence.
        │
        ▼
[2] query_generation.py    ── LLM: claim -> 3 diverse web search queries
        │
        ▼
[3] web_search.py          ── SearchProvider.search(query) -> [SearchResult, ...]   (title, url, snippet, domain, date)
        │
        ▼
[4] webpage_extraction.py  ── fetch each URL, extract clean article text -> Evidence (or a specific failure reason)
        │
        ▼
[5] evidence_ranking.py    ── score by relevance + source authority + freshness -> RankedEvidence (E1 = best)
        │
        ▼
[6] llm_verifier.py        ── LLM: CLAIM vs. top evidence -> SUPPORTS/CONTRADICTS/INSUFFICIENT -> REAL/FAKE/MISLEADING/UNVERIFIED
        ▼
   Verdict (strict JSON, grounded, must cite evidence_ids)
```

`src/verification/pipeline.py` wires all six stages together; `run_demo.py` is the CLI that runs one comment through the whole thing and prints every stage's output.

### 3.2 What each module does — and deliberately does *not* do

| # | File | Does | Does NOT |
|---|---|---|---|
| 1 | `claim_extraction.py` | Detects whether a comment contains a checkable factual claim; rewrites it as one clean English sentence | Judge whether the claim is true |
| 2 | `query_generation.py` | Turns the claim into 3+ diverse search-engine queries | Search the web itself |
| 3 | `web_search.py` | Calls a search API (Serper.dev → Google Search results), returns raw results | Rank, read, or judge the results |
| 4 | `webpage_extraction.py` | Downloads a URL, extracts the readable article text (`trafilatura`, fallback `BeautifulSoup`) | Decide if the page supports/contradicts anything |
| 5 | `evidence_ranking.py` | Scores/sorts evidence by relevance + source authority + freshness | Decide truth — a low-ranked page is still shown to the LLM, just later |
| 6 | `llm_verifier.py` | The **only** module allowed to output a truth verdict, and only by comparing the claim against the evidence text it was actually given | Search the web, or invent evidence not in its input |

This strict separation ("IR must never decide truth") is the main design principle: modules 3-5 are swappable independently (different search API, different ranking algorithm) without ever touching how the final verdict is produced, and module 6 explicitly returns `UNVERIFIED` (not `FAKE`) when evidence is missing — "we couldn't find proof" is never conflated with "it's false."

### 3.3 How the retrieval (IR) part actually works

- **Query generation instead of literal claim search** — search engines are keyword-oriented, so the LLM turns one claim into several short, differently-phrased queries (e.g. *"RBI 5000 rupee note launch rumors"*, *"fake news RBI issuing 5000 currency"*) rather than feeding the full sentence in.
- **`SearchProvider` is an abstract interface** with one concrete implementation, `SerperSearchProvider` (wraps [Serper.dev](https://serper.dev), which returns Google Search results). Adding another provider (Bing, Tavily, Google CSE) later means writing one more class + one `elif` in `get_search_provider()` — nothing else changes.
- **Never fabricates on failure** — if the search API key is missing or the call errors/times out/rate-limits, it raises a specific error that the pipeline surfaces as *"Search unavailable: \<reason\>"*, rather than silently returning an empty result that could be misread as "the web has no evidence."
- **Full-text extraction, not just snippets** — search results include short snippets, but each URL is actually fetched and its main article text extracted, since snippets are usually too short for the LLM to reason over properly.
- **Ranking combines three signals**: TF-IDF cosine relevance (BM25 also supported) between claim and page text, a curated authority score (RBI/PIB/ECI/WHO/major fact-checkers/reputable news = 0.75-1.0, unknown domains = neutral 0.4, not zero), and a freshness decay curve on publish date (missing date = neutral 0.5). Combined as a weighted sum (`0.5·relevance + 0.3·authority + 0.2·freshness`, tunable), then re-labeled `E1..En` (`E1` = best) — those are the exact IDs the LLM is required to cite.
- **Disk caching** (`cache.py`) — both the search calls and every LLM call are cached by exact query/prompt, so re-running the same claim during testing/demoing costs zero extra API quota.

### 3.4 Live example (actually run for this document)

```
$ python src/verification/run_demo.py --text "RBI 5000 rupees note release pannitanga nu news la varudhu"
```

| Stage | Output |
|---|---|
| Cleaned text | `RBI 5000 rupees note release pannitanga nu news la varudhu` |
| Normalized claim | *"The news is reporting that the RBI has released the 5000 rupees note."* |
| Generated queries | `RBI 5000 rupee note launch rumors`; `Reserve Bank of India new 5000 banknote`; `fake news RBI issuing 5000 currency` |
| Search results | 9 URLs retrieved (rbi.org.in, etnownews.com, instagram.com, facebook.com, youtube.com, ...) |
| Extraction | 9/9 pages extracted successfully |
| Top ranked evidence | **E1** rbi.org.in (authority 1.0) · **E2** etnownews.com fact-check article (relevance 0.62) |
| **Final verdict** | **FAKE**, confidence **1.0**, relation **CONTRADICTS**, citing **E2, E5** |
| Explanation (LLM, grounded in evidence) | *"The claim that the RBI is releasing a 5000 rupees note is false. According to the Press Information Bureau (PIB) Fact Check Unit, the viral social media message... is fake, and no such announcement has been made by the RBI or the Ministry of Finance."* |

This demonstrates the whole point of Approach B: the verdict is traceable to two specific, named, retrieved sources — not a black-box guess.

### 3.5 Code

| File | Role |
|---|---|
| `src/verification/schemas.py` | Dataclasses passed between every stage (`ExtractedClaim`, `SearchResult`, `Evidence`, `RankedEvidence`, `Verdict`) |
| `src/verification/llm_client.py` | Generic Gemini wrapper: retries with backoff, strict JSON parsing, caching |
| `src/verification/claim_extraction.py` | Stage 1 |
| `src/verification/query_generation.py` | Stage 2 |
| `src/verification/web_search.py` | Stage 3 — `SearchProvider` interface + `SerperSearchProvider` |
| `src/verification/webpage_extraction.py` | Stage 4 |
| `src/verification/evidence_ranking.py` | Stage 5 |
| `src/verification/llm_verifier.py` | Stage 6 |
| `src/verification/pipeline.py` | Orchestrator — runs all 6 stages, stops early with a clear reason if any stage can't proceed |
| `src/verification/run_demo.py` | CLI — runs one comment through the pipeline and prints every stage |
| `src/verification/cache.py` | Disk-backed JSON cache used by `llm_client.py` and `web_search.py` |

---

## 4. Where the two approaches currently stand — comparison caveat

Approach A outputs a **2-class** verdict (REAL/FAKE); Approach B outputs a **4-class** verdict (REAL/FAKE/MISLEADING/UNVERIFIED) and needs a live web search API call + LLM call per claim (cost and latency, unlike A's instant inference). A rigorous head-to-head on the *same* test set hasn't been run yet — it requires either collapsing B's 4 classes down to 2, or a batch runner that pushes all 518 test headlines through the full pipeline (not yet built; see §6).

|  | Approach A (MuRIL / IndicBERT) | Approach B (LLM + evidence) |
|---|---|---|
| Input | Headline text only | Comment/claim text |
| Knowledge source | Frozen at training time | Live web search, every call |
| Classes | REAL / FAKE | REAL / FAKE / MISLEADING / UNVERIFIED |
| Explainability | None (just a label) | Full explanation + cited sources |
| Cost per prediction | ~free after training | 1 LLM call (claim) + 1 LLM call (queries) + N search API calls + N page fetches + 1 LLM call (verdict) |
| Can catch a brand-new rumor? | No (limited to training-time patterns) | Yes, in principle |

---

## 5. Tech stack

| Component | Library / service |
|---|---|
| Baseline classifiers | `transformers`, `torch`, `datasets`, `scikit-learn` (metrics) — MuRIL / IndicBERTv2 checkpoints from HuggingFace |
| Claim extraction, query generation, verdict reasoning | Google **Gemini API** (`google-generativeai`), model `gemini-flash-lite-latest` |
| Web search | **Serper.dev** REST API (Google Search results) |
| Webpage text extraction | `trafilatura`, fallback `BeautifulSoup4` |
| Evidence ranking | TF-IDF (`scikit-learn`) or BM25 (`rank-bm25`) + hand-curated authority list + date-based freshness decay |
| Environment / config | `python-dotenv` (`.env`), `data/cache/` disk cache |

---

## 6. Honest limitations to state up front

1. **Baseline classifier dataset has a source/era confound** (§2.1) — 99% accuracy reflects separability of two dataset styles, not a validated real-world fake-news detection rate.
2. **Baseline classifiers trained on pure Tamil-script text**, not the romanized/code-mixed text the project is really about — cross-domain generalization untested.
3. **No GPU was available** — both classifiers were fine-tuned on CPU (a few hours each); this was viable specifically because MuRIL/IndicBERT are base-sized models and the dataset is small (~4-5k rows), not something that would scale to a much larger dataset without a GPU.
4. **Approach A and B haven't been benchmarked against each other yet** on the same test set (§4) — that's the immediate next step, not a finished result.
5. **The LLM pipeline has only been run on one demo example end-to-end**, not the full dataset — a batch runner with rate-limiting/checkpointing (like the earlier LID-annotation runner) is needed before that's possible.
6. **Ranking weights (0.5/0.3/0.2) and the trusted-domain authority list are reasonable starting points, not empirically tuned.**
7. `google-generativeai` (used for the Gemini calls) is now a **deprecated package** upstream (Google is migrating everyone to `google-genai`) — still functional today, but worth migrating before long-term/graded submission if time allows.

---

## 7. Suggested next steps

1. Decide how to reconcile Approach A's 2-class output with Approach B's 4-class output (collapse B to 2, or vice versa) for a fair side-by-side comparison.
2. Build `run_batch.py` for `src/verification/` (same checkpointing/rate-limit pattern as the earlier annotation runner) to push all 518 test headlines through the LLM pipeline.
3. Run the trained MuRIL/IndicBERT classifiers on a handful of real Tanglish YouTube comments (not headlines) to sanity-check whether they generalize beyond the training domain at all.
4. If time allows, migrate `src/verification/llm_client.py` off the deprecated `google-generativeai` package.

---

## 8. How to reproduce everything above

```bash
# one-time setup (isolated venv avoids polluting any other Python project on this machine)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-train.txt

# Approach A — baseline classifiers (already run; results in models_fake_news/)
.venv\Scripts\python src\fake_news_classifier\preprocess.py
.venv\Scripts\python src\fake_news_classifier\train_classifier.py --model indicbert-v2
.venv\Scripts\python src\fake_news_classifier\train_classifier.py --model muril
.venv\Scripts\python src\fake_news_classifier\compare_classifiers.py

# Approach B — evidence-aware LLM pipeline (needs GEMINI_API_KEY + SERPER_API_KEY in .env)
.venv\Scripts\python src\verification\run_demo.py --text "your Tamil-English comment here"
```
