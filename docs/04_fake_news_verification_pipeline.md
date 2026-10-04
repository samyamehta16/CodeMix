# Evidence-Aware Fake News Detection — Core Modules

## 1. Project pivot

This project has moved from *"detect aggression in Tamil-English code-mixed
comments"* to:

> **Evidence-Aware Fake News Detection for Tamil-English Code-Mixed Social
> Media using IndicBERT, MuRIL and LLMs.**

The LID annotation pipeline (`01_data_collection.md`, `02_...`, `03_...`)
and its code (`src/preprocess.py`, `src/run_annotation.py`, etc.) still
exist and still work — they may still feed into this project later (e.g.
code-mixing patterns as a signal), but the active pipeline is now the one
described here, under `src/verification/`.

**Status:** the 6 core modules + orchestrator + test script below are
built and working end-to-end (tested module-by-module and as a full
pipeline). Your **full comment dataset has not been run through it yet** —
per your instructions, this was built and tested on a single example
claim only.

## 2. The pipeline, end to end

```
Tamil-English comment
        │  clean_comment() [existing utils.py]
        ▼
[1] claim_extraction.py    ── LLM: does this contain a factual claim? normalize it.
        │
        ▼
[2] query_generation.py    ── LLM: claim -> 3 diverse web search queries
        │
        ▼
[3] web_search.py          ── SearchProvider.search(query) -> [SearchResult, ...]
        │                      (title, url, snippet, domain, date)
        ▼
[4] webpage_extraction.py  ── fetch each URL, extract clean article text
        │                      -> Evidence (or a specific failure reason)
        ▼
[5] evidence_ranking.py    ── score by relevance + source authority + freshness
        │                      -> RankedEvidence (E1 = best)
        ▼
[6] llm_verifier.py        ── LLM: CLAIM vs top evidence -> SUPPORTS/CONTRADICTS/INSUFFICIENT
        │                      -> REAL / FAKE / MISLEADING / UNVERIFIED
        ▼
   Verdict (strict JSON, grounded, cites evidence_ids)
```

`pipeline.py` wires these together and **stops immediately with a clear
reason** the moment a stage can't proceed (no Gemini key, no claim
detected, no search provider configured, no search results) rather than
guessing past a broken step. `run_demo.py` is the CLI that prints every
stage for one input comment.

## 3. What each module does (and does NOT do)

| # | File | Does | Does NOT |
|---|---|---|---|
| 1 | `claim_extraction.py` | Detects if a comment contains a checkable claim; rewrites it as one clean English sentence | Judge whether the claim is true |
| 2 | `query_generation.py` | Turns the claim into 3+ diverse search-engine queries | Search the web itself |
| 3 | `web_search.py` | Calls a search API, returns raw results (title/url/snippet/domain/date) | Rank, read, or judge the results |
| 4 | `webpage_extraction.py` | Downloads a URL, extracts the readable article text | Decide if that page supports/contradicts anything |
| 5 | `evidence_ranking.py` | Scores/sorts evidence by relevance, authority, freshness | Decide truth — a low-authority page can still be shown to the LLM, just ranked lower |
| 6 | `llm_verifier.py` | The **only** module that outputs a truth verdict, and only by comparing claim vs. the evidence text it was given | Search the web or invent evidence not in its input |

This separation is the point of the whole "IR must not decide truth"
requirement: modules 3–5 are swappable/replaceable independently (change
search provider, change ranking algorithm) without ever touching how the
final verdict is produced, and module 6 can never claim something is fake
just because retrieval came up empty (that path explicitly returns
`UNVERIFIED`, see `pipeline.py`'s `STAGE_NO_RESULTS` /
`STAGE_SEARCH_UNAVAILABLE` and `llm_verifier.verify_claim`'s
no-evidence branch).

## 4. How the Information Retrieval (IR) part actually works

1. **Query generation, not literal claim search.** Search engines are
   keyword-oriented; a full sentence claim often retrieves worse than a
   few short, differently-phrased queries. That's why module 2 exists
   instead of feeding the claim straight into module 3.
2. **`SearchProvider` is an abstract interface** (`web_search.py`):
   ```python
   class SearchProvider(ABC):
       def search(self, query: str, num_results: int = 5) -> List[SearchResult]: ...
   ```
   Right now exactly one concrete implementation exists —
   `SerperSearchProvider` (wraps [Serper.dev](https://serper.dev)'s REST
   API, which itself returns Google Search results). `get_search_provider()`
   is the single factory function that decides which one to instantiate,
   based on `SEARCH_PROVIDER` in `.env`. To add Tavily/Bing/Google
   CSE/Brave later: write one more class implementing `.search()`, add one
   `elif` branch in `get_search_provider()` — nothing else in the codebase
   changes.
3. **Never fabricate on failure.** If `SERPER_API_KEY` is missing, or the
   API errors/times out/rate-limits, `search()` raises
   `SearchProviderError` with a specific reason. The pipeline catches this
   and reports `"Search unavailable: <reason>"` — it never silently
   returns an empty list that could be mistaken for "the web has no
   evidence" (which would bias the LLM toward `UNVERIFIED` for the wrong
   reason).
4. **Extraction, not just snippets.** Search results include a short
   `snippet`, but module 4 actually fetches each URL and extracts the
   full article text with `trafilatura` (falls back to BeautifulSoup) —
   snippets are often too short/out-of-context for the LLM to reason over
   properly.
5. **Ranking combines three signals** (`evidence_ranking.py`):
   - **Relevance** — TF-IDF cosine similarity between the claim and each
     page's extracted text (BM25 also supported via `method="bm25"`).
   - **Authority** — a curated dict of trusted domains (RBI, PIB, ECI,
     WHO, major fact-checkers, reputable news outlets) scored 0.75–1.0;
     unknown domains default to a neutral 0.4 (not zero — an unfamiliar
     domain isn't automatically untrustworthy, just unverified).
   - **Freshness** — a recency decay curve on the published date, with a
     neutral 0.5 for missing/unparseable dates (so "no date found" isn't
     punished as "definitely old").
   - These combine as a weighted sum (`0.5·relevance + 0.3·authority +
     0.2·freshness` by default, tunable via `rank_evidence(weights=...)`),
     and the sorted result gets re-labeled `E1..En` (`E1` = best evidence)
     — those are the exact IDs the LLM must cite back.
6. **Caching** (`cache.py`): both `SerperSearchProvider.search()` and
   `LLMClient.generate_json()` check a local disk cache (keyed by the
   exact query / exact prompt) before calling out, and write to it after
   a successful call. Re-running the same claim during testing costs zero
   extra API quota. Delete `data/cache/` to force a refresh.

## 5. How to test it right now

```bash
pip install -r requirements.txt
# .env needs GEMINI_API_KEY (already set) and SEARCH_PROVIDER=serper + SERPER_API_KEY

python src/verification/run_demo.py --text "RBI 5000 rupees note release pannitanga nu news la varudhu"
```

This prints, in order: cleaned text → extracted/normalized claim →
generated search queries → retrieved URLs → per-page extraction
success/failure → ranked evidence with scores → final verdict JSON. If
any stage can't run (e.g. `SERPER_API_KEY` not set yet), it stops there
and tells you exactly why instead of continuing with fabricated data —
that's what you saw when we tested this without a search key configured.

Every module is also independently runnable for isolated testing/debugging:

```bash
python src/verification/claim_extraction.py "your comment here"
python src/verification/query_generation.py "your normalized claim"
python src/verification/web_search.py "your search query"
python src/verification/webpage_extraction.py https://example.com/article
python src/verification/evidence_ranking.py     # uses a built-in dummy example
python src/verification/llm_verifier.py         # uses a built-in dummy example
```

## 6. Search provider: Serper.dev

You chose **Serper.dev** as the first (and currently only) wired-up
provider:

- Sign up at [serper.dev](https://serper.dev) → free tier is 2,500
  one-time queries, no credit card required.
- Copy your API key into `.env`:
  ```
  SEARCH_PROVIDER=serper
  SERPER_API_KEY=your_key_here
  ```
- No code changes needed — `web_search.py`'s factory picks it up
  automatically once the key is present.

## 7. Bug fixed while testing this

While testing module 1 on the exact example from your spec, `"RBI 5000
rupees note..."` came back normalized as `"...500 rupees note..."` — the
existing text-cleaning helper (`utils.clean_comment`, used everywhere
including the earlier LID pipeline) was collapsing 3+ repeated characters
(`"soooo"` → `"soo"`) **without excluding digits**, so `"5000"` (three
repeated `0`s) got collapsed to `"500"`. Fixed the regex to only collapse
repeated *letters*, never digits. This matters a lot for a fact-checking
system where the exact number in a claim is often the entire point.
(Checked impact on the existing dataset: only 4 of ~1000 raw comments —
all near-duplicates of one Tamil comment mentioning "First 1000, First
500, First 100" — were ever affected, so the existing LID dataset wasn't
meaningfully compromised, but it's fixed for everything going forward.)

## 8. Known gaps / next decisions (not yet built)

- **Not yet processing your dataset.** You said you'll provide the fake
  news-relevant CSV separately — nothing here has been pointed at
  `data/raw/youtube_comments.csv` for claim extraction yet.
- **IndicBERT/MuRIL aren't in this pipeline yet** — per your instructions,
  this version uses Gemini for claim extraction/query generation/final
  reasoning only. The existing `src/train_lid_tagger.py` (LID tagging) is
  unrelated to this; if/when you want IndicBERT/MuRIL involved in *this*
  pipeline (e.g. as an embedding-based relevance scorer in
  `evidence_ranking.py`, or as a classification baseline to compare
  against the LLM verdict), that's a follow-up module, not yet started.
- **No batch/dataset runner yet** — `run_demo.py` handles one comment at a
  time by design (as requested). A `run_batch.py` over a full CSV would
  be a quick follow-up once you're ready to process real data, and should
  probably add: a per-row rate limiter, resumable checkpointing (same
  pattern as `run_annotation.py`), and cost/quota estimation up front.
- **Ranking weights (0.5/0.3/0.2) and the trusted-domain list are starting
  points**, not tuned — worth revisiting once you see real claims.
