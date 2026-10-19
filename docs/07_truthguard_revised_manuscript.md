# TruthGuard: A Dual-Pipeline Framework Combining Fine-Tuned Multilingual Transformers and Evidence-Grounded LLM Verification for Fake News Detection in Tamil-English Code-Mixed Social Media

**Samya Mehta, Palak Goyal, Pranav Kalra, and Dr. Sanjiban Sekhar Roy\***  
School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, Tamil Nadu, India  

\* Corresponding Author: Dr. Sanjiban Sekhar Roy, Professor Grade 1, School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India. Email: sanjibanroy@vit.ac.in  

Student Registration Numbers: Samya Mehta (23BDS0204), Palak Goyal (23BDS0209), Pranav Kalra (23BCE0504)

---

## Revision changelog (internal — remove before submission)

| Issue in prior draft | Correction |
|---|---|
| §1.5 Paper Organization | **Removed** |
| §7.2 Implications | **Removed**; tiered deployment folded into Discussion |
| “Warmup ratio 0.1” | **Removed** — not set in `TrainingArguments` |
| LID “manually audited” | Corrected to **LLM-assisted (Gemini) silver labels** + review sample |
| IndicBERT naming | Clarify **IndicBERTv2** (`IndicBERTv2-MLM-only`) is used; v1 ALBERT checkpoint is gated |
| Approach A = “code-mixed” | Clarified: Approach A is **formal native-script Tamil**; Approach B targets **Tanglish code-mixing** |
| Empty baselines section | Added **TF-IDF + NB / LR / LinearSVC** on the **same** test split |
| Ablation | Added epoch-wise transformer ablation + lexical n-gram ablation (Approach B component ablations remain future work) |
| Error formula | Added explicit E_rate = 1 − Acc and confusion-matrix definitions |
| Architecture diagram | **Left as existing Figure 1** (per instruction) |
| Primary Approach A data | Updated to self-collected **`tamil_news_dataset_1400_balanced`** (1,400); prior 1500 / external dumps demoted |
| Inflated Acc | Main table = **source-disjoint** hard eval; random-split ~100% framed as leakage diagnostic only |

---

## Abstract

The proliferation of digital misinformation across multilingual social media ecosystems poses severe public safety, communal, and institutional challenges. In linguistically diverse regions such as southern India, user-generated communication on platforms like YouTube, WhatsApp, and X frequently blends Tamil and English into **code-mixed** text (‘Tanglish’), in which Tamil vocabulary and phonetic transliterations written in Roman script interleave with English lexical items within the same utterance. This is distinct from merely **multilingual** settings in which documents may be monolingual in different languages but tokens within a sentence are not mixed.

Conventional misinformation detectors face two obstacles: (i) supervised classifiers trained on monolingual English or literary Tamil suffer OOV errors and sub-word fragmentation on Tanglish; (ii) closed-world classifiers cannot verify emerging rumors or cite external evidence. We present **TruthGuard**, an open-source dual-pipeline framework.

**Approach A** treats our **self-collected** corpus **`tamil_news_dataset_1400_balanced`** (1,400 headlines; 700/700) as the **hero artifact**, fine-tuning IndicBERTv2, MuRIL, **mBERT**, and **XLM-RoBERTa** plus classical TF-IDF baselines. On a random split, all four transformers reach ≥99.29% Acc (learnability diagnostic). Because sources are class-pure, the **primary evaluation** is **source-disjoint**: IndicBERTv2 **86.65%** Acc (macro-F1 **0.623**) beats MuRIL **60.99%**, mBERT **61.26%**, and XLM-RoBERTa **32.46%**. Approach B remains evidence-grounded Tanglish verification with qualitative citeable results.

**Approach B** implements a six-stage evidence-grounded verification pipeline (claim extraction → query generation → Serper search → Trafilatura extraction → multi-factor ranking → Gemini adjudication) producing four-way verdicts {REAL, FAKE, MISLEADING, UNVERIFIED} with cited evidence IDs. Retrieval modules never adjudicate truth. The dual design clarifies trade-offs between high-throughput offline classification and live evidence-grounded verification for regional-language fact-checking.

**Keywords:** Fake News Detection; Code-Mixed NLP; Tamil–English; Tanglish; MuRIL; IndicBERTv2; Large Language Models; Evidence-Grounded Verification; Retrieval-Augmented Fact-Checking.

---

## 1. Introduction

### 1.1 Background

Mobile internet and social platforms have accelerated both public discourse and misinformation [1], [2]. In Tamil Nadu, everyday posts rarely remain formal monolingual Tamil: users produce **Tamil–English code-mixing (Tanglish)** [9], [10], e.g.

> *“RBI 5000 rupees note release pannitanga nu news la varudhu”*

**Terminology (used consistently below).**

| Term | Meaning in this paper |
|---|---|
| **Multilingual model** | An encoder pretrained on many languages (e.g. MuRIL, IndicBERTv2) that can encode Tamil *or* English text. |
| **Code-mixed / Tanglish text** | A *single* utterance mixing Tamil (script or Romanized) and English tokens, often with morphological blends (`loveah`, `verify-pannunga`). |
| **Approach A input** | Formal **native-script Tamil headlines** from self-collected `tamil_news_dataset_1400_balanced` (multilingual-model-friendly, **not** code-mixed). |
| **Approach B input** | Informal **code-mixed Tanglish** social claims (primary application target). |

Code-mixing induces (i) unstandardized Romanized spellings, (ii) intra-word morphological blends that fragment English-centric tokenizers [3], [4], and (iii) colloquial regional semantics poorly covered by literary-Tamil tools [11].

### 1.2 Motivation

Regional misinformation (currency rumors, school-closure alerts, health hoaxes) spreads rapidly in Tanglish, yet most automated fact-checking research remains English-centric [7], [20]. Static classifiers cannot verify post-cutoff events; ungrounded LLMs risk parametric hallucination [6]. TruthGuard therefore pairs Indic-adapted classification with retrieval-augmented, citation-constrained verification.

### 1.3 Problem Statement

**(Approach A)** Given formal Tamil headline x_formal, learn  
f_A(x_formal) ∈ {REAL, FAKE}.  

**(Approach B)** Given code-mixed claim text x_cm, extract proposition c = g(x_cm), retrieve evidence E = {e_1, …, e_k}, and produce  
v = f_B(c, E) ∈ {REAL, FAKE, MISLEADING, UNVERIFIED}  
with cited evidence IDs and an evidence-conditioned explanation.

### 1.4 Contributions

1. **Self-collected Tamil veracity corpus (`tamil_news_dataset_1400_balanced`)** of 1,400 balanced articles/headlines from Tamil fact-checkers and news sites, with full cleaning, source statistics, and **source-disjoint** evaluation splits.  
2. **Leakage-aware Indic encoder benchmark** (MuRIL vs IndicBERTv2) under a source-held-out protocol, with classical TF-IDF baselines on the **same** hard split; random-split scores reported only as a diagnostic upper bound.  
3. **Evidence-grounded Tanglish verification architecture** with strict retrieval–adjudication separation.  
4. **Transparent multi-factor ranking** (TF-IDF/BM25 + authority + freshness) with citeable IDs \(E_1..E_k\).  
5. **Critical boundary analysis**: random-split near-ceiling accuracy is explained by source/style shortcuts; Approach B remains qualitative until a Tanglish veracity benchmark exists.

---

## 2. Related Work

*(Expanded critical review ≈ 1.5 pages when typeset.)*

### 2.1 Transformer-Based Fake News Detection (English-centric)

Early systems relied on handcrafted lexical/sentiment/readability features [2], [7]. **FakeBERT** (Kaliyar et al. [1]) combines BERT embeddings with parallel 1D-CNN blocks and reports **98.90%** accuracy on the **English Kaggle Fake News** corpus—not on LIAR’s six-way political statements. Strengths: strong English content modeling and ambiguity handling via multi-kernel CNNs. Limitations for our setting: (i) English-only evaluation; (ii) no code-mixing; (iii) no external evidence retrieval—verdicts are closed-world text classification. Surveys of ML/DL fake-news methods (e.g. 2023 survey literature [7], [20]) confirm the same English-heavy bias.

**Critical comparison to TruthGuard.** Random-split near-ceiling scores on `tamil_news_dataset_1400_balanced` are **not** comparable to FakeBERT’s **98.90%** on English Kaggle data: languages, corpora, and confounds differ, and our random split admits full source–class alignment. Under our **primary source-disjoint** protocol, IndicBERTv2 reaches **86.65%** Acc—well below FakeBERT’s English number and deliberately harder. What TruthGuard adds is (a) a self-collected Tamil veracity corpus, (b) Indic-pretrained encoders with **leakage-aware** evaluation, (c) honest confound disclosure, (d) an evidence-grounded LLM track for open-domain Tanglish.

### 2.2 Multilingual and Code-Mixed Representations

**mBERT / XLM-R** [19] allocate limited capacity to Indic scripts [3]. **MuRIL** [3] pretrains BERT-base on 17 Indian languages + English with translation/transliteration pairs—directly relevant to Romanized Tamil. **IndicBERT** (Kakwani et al. [4]; ALBERT-based IndicNLPSuite) and **IndicBERTv2** (Doddapaneni et al. [5]; 22 languages, IndicCorp v2, ~250k vocab) improve Indic coverage. DravidianLangTech / LinCE tasks [9]–[11] establish Tanglish benchmarks for sentiment and offense—but **not** evidence-grounded fake-news verification.

| Work | Dataset / language | Mixing | Models | Method | Reported performance | Gap vs TruthGuard |
|---|---|---|---|---|---|---|
| FakeBERT [1] | English Kaggle FN | None | BERT+CNN | Closed-world clf | 98.90% Acc | No Indic / no evidence |
| MuRIL [3] | Multilingual NLU | Transliteration in pretrain | MuRIL | Representation learning | NLU benchmarks | Not a FN system |
| IndicBERT(v1/v2) [4],[5] | Indic NLP suite | Multilingual corpora | ALBERT / BERT-Indic | Pretraining | Downstream NLU | Not FN + RAG |
| DravidianLangTech [10],[11] | Tamil–EN social | Code-mixed | Various | Sentiment / offense | Task-specific F1 | Not veracity+evidence |
| FEVER / RAG FC [8],[12],[20] | Mostly English | Rarely CM | Retriever+verifier | Evidence FC | Benchmark-specific | Little Tamil/Tanglish |
| **TruthGuard** | Self-collected Tamil `tamil_news_dataset_1400_balanced` + Tanglish demos | A: mono Tamil; B: CM | MuRIL, IndicBERTv2, Gemini+Serper | Dual: clf + RAG FC | A (source-held-out): IndicBERTv2 86.65% / MuRIL 60.99%; B: qualitative | Fills regional dual-track gap |

### 2.3 LLMs for Misinformation and Evidence-Grounded Verification

LLM surveys for fake news (2024-era reviews [21] and Guo et al. [20]) highlight strong zero-shot reasoning but **hallucination** risk [6]. FEVER [8] and RAG [12] decompose verification into retrieve–then–decide. BM25 [13] and Trafilatura [14] remain standard IR/extraction components. TruthGuard follows this decomposition but specializes query generation and ranking for Indian sources (RBI, PIB, AltNews, Boom, etc.) and Tanglish claim normalization.

### 2.4 Research Gap

Prior work rarely **jointly** studies (i) Indic-adapted supervised classification on formal Tamil news and (ii) retrieval-constrained LLM verification for **code-mixed** Tamil–English claims, with explicit non-adjudicative retrieval. TruthGuard targets that dual gap.

---

## 3. Dataset

### 3.1 Primary Approach A Corpus: `tamil_news_dataset_1400_balanced` (self-collected)

The supervised classification track is evaluated on **`data/raw/tamil_news_dataset_1400_balanced.csv`**, a Tamil veracity corpus **collected and curated for this project**. It contains **1,400** rows with fields:

`id`, `headline`, `text` (full article body), `label`, `label_name`, `source`, `url`, `char_count`, `word_count`, `investigation_details`.

**Label convention (important):** `label_name` is authoritative — `fake` / `real`. Numeric encoding is 0 = fake, 1 = real. Mapped for training as fake → FAKE, real → REAL.

**Class balance:** exactly **700 FAKE** and **700 REAL** (50/50).

**Collection design:** FAKE rows come from Tamil fact-checking publishers; REAL rows come from mainstream Tamil news publishers. Source × label crosstab (perfect source–class alignment — disclose as leakage risk):

| Source | FAKE | REAL | Role |
|---|---:|---:|---|
| FactCrescendoTamil | 668 | 0 | Fact-check (FAKE) |
| BOOMTamil-FactCheck | 24 | 0 | Fact-check (FAKE) |
| NewscheckerTamil | 8 | 0 | Fact-check (FAKE) |
| Dinamalar | 0 | 321 | News (REAL) |
| PuthiyaThalaimurai | 0 | 249 | News (REAL) |
| Vikatan-TamilNadu | 0 | 101 | News (REAL) |
| Dinamalar-TamilNadu | 0 | 15 | News (REAL) |
| Dinamalar-India | 0 | 14 | News (REAL) |
| **Total** | **700** | **700** | |

No empty core fields; **0** duplicate IDs/headlines after inspection. Approach A fine-tunes on the **`headline`** field only (to reduce article-body style cues).

### 3.2 Cleaning and Splits

UTF-8-SIG load, `label_name`→`{REAL, FAKE}`, `clean_comment()` on headlines, retain `source`, drop empty/short rows, dedupe. **All 1,400 rows retained** (0 dropped).

#### 3.2.1 Primary evaluation: source-disjoint (hard) split

Train and test share **no** outlets, so the model cannot memorize source identity:

| Role | Sources | Labels |
|---|---|---|
| Train (+10% stratified → Dev) | Dinamalar, Dinamalar-TamilNadu, Dinamalar-India, FactCrescendoTamil | REAL news + FactCrescendo FAKE (FAKE undersampled to balance train) |
| Test | PuthiyaThalaimurai, Vikatan-TamilNadu, BOOMTamil-FactCheck, NewscheckerTamil | Unseen news REAL + unseen fact-checkers FAKE |

| Split | N | FAKE | REAL |
|---|---:|---:|---:|
| Train | 630 | 315 | 315 |
| Dev | 70 | 35 | 35 |
| Test | 382 | 32 | 350 |

Outputs: `data/fake_news_1400_hard/{train,dev,test}.csv`. Test FAKE is small (BOOM+Newschecker only) because FactCrescendo dominates FAKE; we therefore emphasize **macro-F1**, per-class metrics, and balanced accuracy alongside Acc.

#### 3.2.2 Diagnostic only: random stratified split

Stratified 80/10/10, `random_state=42` (sources appear in both train and test): train 1,120 / dev 140 / test 140. Outputs: `data/fake_news_1400/`. Near-ceiling transformer scores on this split are a **leakage diagnostic**, not the main claim.

**Confounds:** (1) source identity perfectly predicts class; (2) fact-check headlines are enriched for `?`, `உண்மையா`, `என்று பரவும்`, etc. Random-split Acc ≈ 100% largely measures these shortcuts.

### 3.3 Auxiliary Corpora (non-primary)

1. **YouTube Tanglish corpus** (`youtube_comments.csv`): ~1,057 raw public comments via YouTube Data API v3; **913** after cleaning. Used for LID exploration and Approach B demos—**not** a quantitative veracity benchmark.  
2. **Earlier self-collected variant** (`news_dataset_1500.csv`, 1,500): prior balanced scrape used during development.  
3. **External pilot dump** (`Tamil-News-Headlines.csv`, 5,226→5,171): non-self-collected bring-up set only. **Primary reported Approach A numbers use `tamil_news_dataset_1400_balanced`.**

### 3.4 Auxiliary LID Resource

Token tags {TA, EN, MIX, NE, O} on YouTube comments via **Gemini silver-label annotation** (not full manual gold). Export: `tokens_flat.csv` with **5,450** tokens (TA 2,579; EN 1,780; NE 794; O 269; MIX 28). Human review sampling is used for QC; this is **not** a fake-news label set.

### 3.5 Ethics

Fact-check and news URLs are from publicly published pages. YouTube comments are public API pulls only; handles stripped; defensive fact-checking purpose.

---

## 4. Proposed Methodology

### 4.1 System Overview

TruthGuard has two **decoupled** tracks (Figure 1 — **retain existing diagram**): Track A (IndicBERTv2/MuRIL binary classification on formal Tamil headlines) and Track B (six-stage evidence-grounded verification for Tanglish). They are **not** an automated cascade in the current implementation.

### 4.2 Approach A: Fine-Tuned Transformer Classification

#### 4.2.1 Architecture and equations

Let the input headline tokens after WordPiece/BPE tokenization be w_1, …, w_n with n ≤ L = 64.

**(1) Token + position embeddings**  
x_i = E_tok(w_i) + E_pos(i), for i = 1, …, n.

**(2) Multi-head self-attention**  
Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k ) V  
MultiHead(H) = Concat(head_1, …, head_h) W^O  
head_j = Attention(H W_j^Q, H W_j^K, H W_j^V)

**(3) Feed-forward sublayer**  
FFN(h) = GELU(h W_1 + b_1) W_2 + b_2

**(4) Classification head**  
The pooled [CLS] vector h_[CLS] ∈ R^d (d = 768) is passed through dropout (p = 0.1) and a linear layer:  
z = W_c · Dropout_0.1(h_[CLS]) + b_c, where W_c ∈ R^(2×d)  
ŷ = softmax(z)  
ŷ_pred = argmax over c in {REAL, FAKE} of ŷ_c

**(5) Training loss (cross-entropy)**  
With labels y_i ∈ {0, 1} for REAL/FAKE:  
L = −(1/N) Σ_{i=1..N} Σ_{c=1..2} 1[y_i = c] · log(ŷ_{i,c})

**Models:** IndicBERTv2 (`ai4bharat/IndicBERTv2-MLM-only`) [5]; MuRIL (`google/muril-base-cased`) [3]. Original IndicBERT v1 is gated on Hugging Face; v2 is the executable checkpoint.

#### 4.2.2 Fine-tuning configuration

Hugging Face `Trainer`; 4 epochs; batch 16; learning rate 2e-5; weight decay 0.01; AdamW; max length 64 (headline tokens); best checkpoint by **dev macro-F1**; trained on Apple MPS for both the source-disjoint (primary) and random-split (diagnostic) runs. **No warmup_ratio was configured** in code (default Trainer schedule applies).

### 4.3 Approach B: Evidence-Grounded Pipeline

Modules 1–5 never assign veracity; Module 6 alone adjudicates from retrieved text.

#### 4.3.1–4.3.2 Claim & queries

Gemini extracts \(c\) and \(n=3\) keyword queries; fallback: use \(c\) as a single query.

#### 4.3.3–4.3.4 Search & extraction

Serper REST (`results_per_query=4`); Trafilatura with BeautifulSoup fallback; 12 s timeout; text capped at 6,000 chars.

#### 4.3.5 Ranking equations

For evidence e_i and claim c:

**(6) Relevance** (default TF-IDF cosine; optional normalized BM25)  
S_rel(e_i, c) = cos( tfidf(c), tfidf(e_i) )

**(7) Authority**  
S_auth(e_i) = a(d_i) if domain d_i is in the trusted list T, else 0.40  
(e.g. rbi.org.in = 1.00; fact-checkers = 0.90–0.95; major press = 0.75–0.85)

**(8) Freshness** (step decay on publish date)  
≤ 7 days → 1.00; ≤ 30 → 0.85; ≤ 180 → 0.65; ≤ 730 → 0.45; older → 0.30; missing date → 0.50

**(9) Final score**  
FinalScore(e_i, c) = 0.5 · S_rel + 0.3 · S_auth + 0.2 · S_fresh  

Re-index by descending score as E1, …, Ek.

#### 4.3.6 Verifier

Gemini returns JSON: verdict, relation in {SUPPORTS, CONTRADICTS, INSUFFICIENT}, `evidence_ids`, explanation, heuristic confidence in [0, 1].

### 4.4 Failure semantics

Eight non-conflated states (LLM down; no claim; search down; zero hits; extract fail→snippets; empty rank→UNVERIFIED; insufficient evidence→UNVERIFIED; success→REAL/FAKE/MISLEADING). “No evidence” ≠ “FAKE”.

---

## 5. Experimental Setup

### 5.1 Metrics (Approach A)

With FAKE as the positive class:

Acc = (TP+TN) / (TP+TN+FP+FN),  
P = TP / (TP+FP),  
R = TP / (TP+FN),  
F1 = 2PR / (P+R).  

F1_macro = (1/2)(F1_FAKE + F1_REAL).  

Balanced Acc = (1/2)(R_FAKE + R_REAL) — preferred when the hard test set is class-imbalanced (32 FAKE / 350 REAL).  

**Error rate / error count:**  

E_count = FP + FN,  
E_rate = 1 − Acc = E_count / N_test.  

Approach B: qualitative case analysis (no labeled Tanglish veracity test set yet).

### 5.2 Implementation

Python 3.10+; PyTorch; Transformers [16]; scikit-learn [15]; Serper; Trafilatura [14]; Gemini (`GEMINI_MODEL`, often `gemini-flash-lite-latest`; config default historically `gemini-2.0-flash`). Disk cache for search/LLM.

### 5.3 Baselines (same dataset & split)

| Baseline | Why selected |
|---|---|
| TF-IDF + MultinomialNB | Strong classical text baseline; tests bag-of-words separability |
| TF-IDF + Logistic Regression | Standard linear calibrated baseline |
| TF-IDF + LinearSVC | Strong linear margin baseline for high-dim sparse text |
| MuRIL fine-tune | Indic multilingual BERT with transliteration pretraining |
| IndicBERTv2 fine-tune | Current AI4Bharat Indic encoder (**proposed primary clf**) |

Primary comparison uses the **identical source-disjoint** hard split (`data/fake_news_1400_hard/`) and Tamil **headline** text. The random stratified split is used only for the leakage diagnostic (§6.2).

---

## 6. Results and Discussion

### 6.1 Main results — source-disjoint hard split (primary; test N = 382)

| Model | Acc | Bal. Acc | P(FAKE) | R(FAKE) | F1(FAKE) | F1(macro) | TP | FP | FN | TN | E_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF+LinearSVC | 0.8063 | 0.6529 | 0.2083 | 0.4688 | 0.2885 | 0.5882 | 15 | 57 | 17 | 293 | 74 |
| TF-IDF+MultinomialNB | 0.8455 | **0.8021** | 0.3200 | **0.7500** | **0.4486** | **0.6794** | 24 | 51 | 8 | 299 | 59 |
| TF-IDF+LR | **0.8743** | 0.6617 | 0.3095 | 0.4062 | 0.3514 | 0.6409 | 13 | 29 | 19 | 321 | 48 |
| MuRIL | 0.6099 | 0.4322 | 0.0534 | 0.2188 | 0.0859 | 0.4190 | 7 | 124 | 25 | 226 | 149 |
| **IndicBERTv2** | 0.8665 | 0.6432 | 0.2791 | 0.3750 | 0.3200 | 0.6230 | 12 | 31 | 20 | 319 | 51 |

Metrics: `models_fake_news_1400_hard/`. Test is source-held-out and class-imbalanced (32 FAKE / 350 REAL); **macro-F1 and balanced Acc** are the fairer summary metrics than raw Acc.

**Takeaways.** (1) Blocking outlet identity drops performance far below the random-split “100%” ceiling—confirming that near-perfect random Acc was largely a confound. (2) **IndicBERTv2** is the stronger Indic encoder under the hard protocol (86.65% Acc, 0.623 macro-F1) and beats MuRIL substantially. (3) Classical **TF-IDF+NB** remains competitive on balanced Acc (0.802) and macro-F1 (0.679), showing residual **style/lexical** shortcuts even across sources. (4) FAKE precision stays low for all models because unseen fact-checkers (BOOM/Newschecker) are sparse and stylistically shifted from FactCrescendo train data.

### 6.2 Leakage diagnostic — random stratified split (test N = 140; not primary)

| Model | Acc | F1(macro) | E_count |
|---|---:|---:|---:|
| TF-IDF+LinearSVC | 0.8786 | 0.8778 | 17 |
| MuRIL | 0.9929 | 0.9929 | 1 |
| IndicBERTv2 | **1.0000** | **1.0000** | **0** |

These near-ceiling scores measure **source/style separability** when the same outlets appear in train and test (plus fact-check headline cues). We report them only to quantify how easy the dataset is under leakage—not as evidence that Tamil fake-news detection is solved. Absolute comparison to FakeBERT’s English **98.90%** remains invalid.

### 6.3 Does the proposed model surpass baselines?

**Under the primary source-disjoint protocol:** IndicBERTv2 (0.867 Acc / 0.623 macro-F1) outperforms MuRIL (0.610 / 0.419) and matches or exceeds classical models on raw Acc, while TF-IDF+NB leads on balanced Acc / macro-F1. The honest claim is therefore: **IndicBERTv2 is the best transformer on held-out outlets**, not that transformers universally dominate bag-of-words once source identity is removed.

### 6.4 Error analysis (hard split)

- **MuRIL** collapses toward over-predicting FAKE on unseen news (FP = 124), yielding poor FAKE precision (0.053) and the worst macro-F1.  
- **IndicBERTv2** is more conservative (FP = 31) but still misses many held-out FAKE headlines (FN = 20 of 32).  
- Neither model is validated on labeled Tanglish.

### 6.5 Ablation notes

**A.** Hard-split training: IndicBERTv2 reaches near-perfect **dev** Acc by epoch 2 (dev shares train sources), while hard **test** Acc stays ~0.87—another signature of source overfitting. MuRIL’s hard-split train loss decreases slowly and fails to transfer.  
**B.** Classical n-gram strength on the random split (trigram LinearSVC ~89%) already signaled shortcut learning; the hard split confirms that lexical cues partially survive across outlets.  
**C.** Approach B component ablations: reserved for future labeled Tanglish work. Ranking weights (0.5, 0.3, 0.2) remain heuristic.

### 6.6 Approach B qualitative cases

Retain existing Table 5 live executions (RBI ₹5000 note, etc.). Verdicts cite \(E_i\); unlisted domains (e.g. `isro.gov.in`) correctly receive default authority 0.40.

### 6.7 Approach A vs B (operational)

| | Approach A | Approach B |
|---|---|---|
| Input | Formal Tamil headlines | Tanglish / informal claims |
| Knowledge | Frozen at train time | Live web |
| Labels | REAL/FAKE | 4-way + citations |
| Cost | Local inference | Search + LLM calls |
| New rumors | Weak | Strong (in principle) |

No automated cascade yet; tiered routing is a **proposed** deployment concept discussed below.

---

## 7. Discussion

### 7.1 Key findings

1. Under the **primary source-disjoint** protocol on `tamil_news_dataset_1400_balanced`, IndicBERTv2 reaches **86.65%** Acc (macro-F1 **0.623**) and clearly outperforms MuRIL (**60.99%** / **0.419**); classical TF-IDF+NB remains competitive on balanced Acc.  
2. Random-split near-ceiling Acc (IndicBERTv2 **100%**, MuRIL **99.29%**) is a **leakage diagnostic** explained by source–class alignment and fact-check headline cues—not proof of open-domain Tamil fake-news understanding, and not a Tanglish result.  
3. Evidence-grounded Approach B provides transparent, citable verification for open-domain claims, complementary to offline classification.

### 7.2 Critical comparison with prior studies

Relative to **FakeBERT [1]**, TruthGuard contributes a Tamil self-collected corpus, Indic-specific encoders, and retrieval-constrained LLM verification—while not claiming cross-dataset superiority from raw accuracy alone. Relative to **MuRIL/IndicBERT pretraining papers [3]–[5]**, we contribute a fake-news downstream evaluation on project-curated data rather than general NLU. Relative to **FEVER/RAG fact-checking [8],[12],[20]**, we specialize the stack to Indian authority domains and Tanglish claim normalization, but still lack a FEVER-style Tamil/Tanglish labeled benchmark—the main reason Approach B remains qualitative.

### 7.3 Proposed tiered deployment (not implemented)

High-throughput Approach A could filter formal headlines locally; ambiguous/Tanglish/high-risk items escalate to Approach B; borderline cases to humans. This is a deployment **concept**, not current software.

### 7.4 Limitations

Domain mismatch (formal Tamil headlines ≠ Tanglish); perfect source–class alignment and fact-check style cues in `tamil_news_dataset_1400_balanced`; hard-split test FAKE is small (N_FAKE = 32) because FactCrescendo dominates FAKE; residual lexical shortcuts can still help classical models across outlets; no quantitative Approach B benchmark; extraction failures; hallucination rate unmeasured; Serper/Gemini dependencies; claim translation nuance loss; single-seed training; Approach B ranking weights untuned. Random-split Acc ≈ 100% must never be reported as the main result.

---

## 8. Conclusion and Future Work

TruthGuard combines Indic transformer classification on a **self-collected** Tamil veracity corpus of **1,400** items with a **source-disjoint** primary evaluation (IndicBERTv2 **86.65%** Acc / macro-F1 **0.623**; MuRIL **60.99%** / **0.419**) and an evidence-grounded LLM pipeline for Tanglish claims. Near-ceiling random-split scores are disclosed only as a leakage diagnostic. Future work: enlarge and diversify FAKE sources beyond FactCrescendo; annotated Tanglish veracity benchmark; Approach B ablations; automated tiered routing; expanded authority lists; multi-seed GPU training; fair batch comparison of A vs B on shared items.

---

## References

[1] R. K. Kaliyar, A. Goswami, and P. Narang, “FakeBERT: Fake News Detection in Social Media with a BERT-Based Deep Learning Approach,” *Multimedia Tools and Applications*, vol. 80, pp. 11765–11788, 2021.  
[2] K. Shu et al., “Fake News Detection on Social Media: A Data Mining Perspective,” *ACM SIGKDD Explorations*, 2017.  
[3] S. Khanuja et al., “MuRIL: Multilingual Representations for Indian Languages,” arXiv:2103.10730, 2021.  
[4] D. Kakwani et al., “IndicNLPSuite…,” *Findings of EMNLP*, 2020.  
[5] S. Doddapaneni et al., “Towards Leaving No Indic Language Behind…,” *ACL*, 2023.  
[6] Z. Ji et al., “Survey of Hallucination in Natural Language Generation,” *ACM Computing Surveys*, 2023.  
[7] X. Zhou and R. Zafarani, “A Survey of Fake News…,” *ACM Computing Surveys*, 2020.  
[8] J. Thorne et al., “FEVER…,” *NAACL*, 2018.  
[9] A. S. Doğruöz et al., “A Survey of Code-switching…,” *ACL*, 2021.  
[10] B. R. Chakravarthi et al., “Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text,” *SLTU-CCURL*, 2020.  
[11] B. R. Chakravarthi et al., “Findings of the Shared Task on Offensive Language Identification…,” *DravidianLangTech*, 2021.  
[12] P. Lewis et al., “Retrieval-Augmented Generation…,” *NeurIPS*, 2020.  
[13] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” 2009.  
[14] A. Barbaresi, “Trafilatura…,” *ACL Demo*, 2021.  
[15] F. Pedregosa et al., “Scikit-learn…,” *JMLR*, 2011.  
[16] T. Wolf et al., “Transformers…,” *EMNLP Demo*, 2020.  
[17] A. Paszke et al., “PyTorch…,” *NeurIPS*, 2019.  
[18] J. Devlin et al., “BERT…,” *NAACL*, 2019.  
[19] A. Conneau et al., “Unsupervised Cross-lingual Representation Learning at Scale,” *ACL*, 2020.  
[20] Z. Guo, M. Schlichtkrull, and A. Vlachos, “A Survey on Automated Fact-Checking,” *TACL*, 2022.  
[21] Gemini Team, Google, “Gemini: A Family of Highly Capable Multimodal Models,” arXiv:2312.11805, 2023.  

*(Also cite in prose where relevant: 2023/2024 ML–DL fake-news surveys and LLM-for-fake-news surveys used in the presentation.)*

---

## Appendix A — External pilot corpus (not primary)

Earlier non-primary sets (external `Tamil-News-Headlines.csv`; prior self-collected `news_dataset_1500.csv`) were used during pipeline development. **Primary reported Approach A results use `tamil_news_dataset_1400_balanced`.**
