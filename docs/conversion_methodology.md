# Conversion Methodology and Romanisation Specification

**Document Version:** 2.0-RC  
**Author:** NLP Data Engineer & Research Reproducibility Auditor  
**Scope:** Tamil News Fake News Benchmark (1,400 Records)  
**Associated Scripts:** [`scripts/reproducible_conversion.py`](file:///c:/Users/palak/OneDrive/Desktop/CodeMix/scripts/reproducible_conversion.py), [`scripts/tanglish_engine.py`](file:///c:/Users/palak/OneDrive/Desktop/CodeMix/scripts/tanglish_engine.py)

---

## 1. System Environment and Dependencies

All transformations are strictly deterministic, executed locally without external cloud APIs, non-deterministic language models, or proprietary translation endpoints.

- **Python Runtime**: Python 3.11+
- **Core Dependencies**:
  - `pandas` ($\ge$ 2.0.0): Structured tabular data operations.
  - `numpy` ($\ge$ 1.24.0): Numerical indexing.
  - Standard Library: `re` (regex state machine), `unicodedata` (Unicode NFC normalization), `hashlib` (SHA-256 validation), `argparse`, `json`.
- **Execution Command**:
  ```bash
  python scripts/reproducible_conversion.py --raw data/raw/tamil_news_dataset_1400_balanced.csv --output output/tamil_news_candidate_v2_sample100.csv --rows 100
  ```

---

## 2. Core Conversion Policy & Architectural Principles

To ensure scientific reproducibility and eliminate bias, four strict design boundaries are established:

```
[Raw Tamil Input] 
       │
       ▼
1. Pre-Normalization (Unicode NFC, NBSP removal, quotes, dashes)
       │
       ▼
2. Structural Parsing (Extract '|' English translations to 'headline_en')
       │
       ▼
3. Canonical Lexicon Lookup (High-confidence loanwords & standard acronyms)
       │
       ▼
4. Phonological Transliteration (Intervocalic voicing, gemination, retroflex zh)
       │
       ▼
5. Post-Processing (Hyphenated suffix rules, typography, capitalization)
       │
       ▼
[Unified Output Schema (17 Columns, Label-Blind, Preserves Original Tamil)]
```

### Principle A: Complete Label-Blindness
The Romanisation engine must **never read, pass, or condition upon the `label` or `label_name` columns**. This mathematically guarantees that no linguistic cues, vocabulary shifts, or stylistic variations can be artificially introduced between real and fake articles.

### Principle B: Strict Separation of Cleanup, Transliteration, and Translation
1. **Mechanical Cleanup**: Normalizing whitespace (`\xa0` $\to$ `' '`), standardizing dashes (`--` $\to$ ` - `), collapsing doubled quotes, and stripping zero-width spaces.
2. **Phonetic Transliteration**: Converting native Tamil morphemes into Roman script using phonological state-machine rules (e.g., `அரசு` $\to$ `arasu`, `செய்திகள்` $\to$ `seithigal`).
3. **Translation Prevention**: Native Tamil words must **never** be translated into English unless they represent loanwords that originated in English.
   - *Prohibited*: `தமிழக` $\to$ `Tamil Nadu` (semantic substitution).
   - *Permitted*: `கூகுள் லென்ஸ்` $\to$ `Google Lens` (restoration of English loanword).
   - *Permitted*: `சிசிடிவி` $\to$ `CCTV` (restoration of English acronym).

---

## 3. Explicit Transformation Rules

### 3.1 Acronyms and Institutional Names
Acronyms that originated in English and are widely known across Indian media should be rendered in standard uppercase Latin characters rather than clumsy phonetic strings:

| Tamil Original | v1 Baseline (Defective) | Proposed Standard Policy | Classification |
| :--- | :--- | :--- | :--- |
| **திமுக / தி.மு.க** | `DMK` | `DMK` | Recognized Indian Political Acronym |
| **பாஜக / பா.ஜ.க** | `BJP` | `BJP` | Recognized Indian Political Acronym |
| **தவெக / த.வெ.க** | `TVK` | `TVK` | Recognized Indian Political Acronym |
| **அதிமுக / அ.தி.மு.க** | `ADMK` | `ADMK` | Recognized Indian Political Acronym |
| **ஆர்எஸ்எஸ் / ஆர்.எஸ்.எஸ்** | `aareses` *(defective)* | `RSS` | Recognized National Entity Acronym |
| **ஏடிஎம் / ஏ.டி.எம்** | `ediem` *(defective)* | `ATM` | Universal Financial Acronym |
| **சி.ஏ.ஏ / சிஏஏ** | `si.e.e` *(defective)* | `CAA` | Universal Legal / Legislative Acronym |
| **சிசிடிவி / சி.சி.டி.வி** | `sisidivi` *(defective)* | `CCTV` | Technical Acronym |
| **சிஎஸ்கே / சி.எஸ்.கே** | `siesge` *(defective)* | `CSK` | Sports Franchise Acronym |
| **பிஐபி** | `PIB` | `PIB` | Press Information Bureau |
| **சிஎன்என்** | `CNN` | `CNN` | Media Network |

### 3.2 Geographic and Political Entities
- **State vs. Adjective**:
  - `தமிழ்நாடு` $\to$ `Tamil Nadu` (Canonical state name).
  - `தமிழக` $\to$ `Thamizhaga` (Adjective / regional term preserved; **do not substitute** `Tamil Nadu`).
  - `தமிழக அரசு` $\to$ `Thamizhaga arasu` (Government of Tamil Nadu; transliterated accurately).
- **Prefix Preservation**:
  - `புதுடில்லி / புதுடெல்லி` $\to$ `Pudhudilli` or `New Delhi` (Preserve `Pudhu` [New]; **do not collapse to bare `Delhi`**).
  - `டெல்லி / தில்லி` $\to$ `Delhi`.

### 3.3 Currency References
- `ரூபாய்` $\to$ `roobai` (Standard Romanised Tamil for rupee).
- `ரூ.` $\to$ `Rs.` (Standard news abbreviation for currency; eliminates ambiguous `Roo.`).

### 3.4 Hybrid Noun-Suffix Compounds
When an English or Latin noun takes a Tamil case suffix, apply hyphenated boundary transliteration:
$$\text{LatinStem} + \text{TamilSuffix} \implies \text{LatinStem}-\text{TransliteratedSuffix}$$
- `Reuters` + `இன்` $\implies$ `Reuters-in`
- `threads.com` + `இல்` $\implies$ `threads.com-il`
- `MP` + `ஆ` $\implies$ `MP-aa`
- `Facebook` + `இல்` $\implies$ `Facebook-il`

### 3.5 Pipe Delimiter (` | `) Resolution
- **Appended English Translations**: When headlines feature ` | ` followed by an English translation or summary (prevalent in BOOM Fact Check articles), the English text must be extracted into the dedicated `headline_en` column. It must **not** be appended to the Romanised Tamil headline, as that leaks bilingual translation shortcuts to text classifiers.
- **Category & Columnist Tags**: Short tags (e.g. `Photo Album`, `Interview`, `கழுகார்`) must be stored in `format_tag` to maintain headline purity.

---

## 4. Candidate Dataset Schema

The candidate dataset retains complete provenance across ground truth and transformed columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique 16-character hexadecimal record identifier (stable across all versions). |
| `headline_ta` | String | Ground-truth Tamil headline (stripped of appended English translations). |
| `headline_romanised` | String | Standardized Romanised Tanglish headline. |
| `headline_en` | String | Preserved English translation (extracted from pipe-separated sources, or empty). |
| `text_ta` | String | Ground-truth Tamil article body text. |
| `text_romanised` | String | Standardized Romanised Tanglish article body text. |
| `investigation_details_ta` | String | Ground-truth fact-check investigation notes (Tamil, null for real news). |
| `investigation_details_romanised` | String | Standardized Romanised fact-check investigation notes. |
| `label` | Integer | Binary label (0 = fake, 1 = real). |
| `label_name` | String | Text label (`fake` or `real`). |
| `source` | String | Original publisher / fact-checking organization. |
| `url` | String | Original canonical article URL. |
| `split` | String | Deterministic dataset split (`train`, `dev`, `test`, or `unassigned`). |
| `char_count` | Integer | Character length of `text_romanised`. |
| `word_count` | Integer | Whitespace word count of `text_romanised`. |
| `format_tag` | String | Extracted media/format tags (`Photo Album`, `Interview`, etc.). |
| `is_flagged_exclusion` | Boolean | True if record appears on [`proposed_exclusions.csv`](file:///c:/Users/palak/OneDrive/Desktop/CodeMix/proposed_exclusions.csv). |

---

## 5. Audit Logging and Validation Protocol

Every execution of [`scripts/reproducible_conversion.py`](file:///c:/Users/palak/OneDrive/Desktop/CodeMix/scripts/reproducible_conversion.py) enforces:
1. **Pre-execution Checksum**: Records raw input SHA-256 before any data read.
2. **Post-execution Checksum**: Re-verifies raw input SHA-256 after writing output to confirm zero file mutation.
3. **Residual Tamil Scan**: Asserts 0 occurrences of Unicode code points `\u0B80`–`\u0BFF` in all converted Romanised columns.
4. **ID Order Alignment**: Asserts 1-to-1 sequence matching against the original dataset.
5. **Null Cardinality Check**: Asserts null count preservation across corresponding ground-truth and Romanised columns.
