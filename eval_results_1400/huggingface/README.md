---
pretty_name: Tamil News Veracity 1400 (TruthGuard)
language:
- ta
- en
license: other
task_categories:
- text-classification
size_categories:
- 1K<n<10K
---

# Tamil News Veracity 1400 (TruthGuard)

Self-collected balanced Tamil headline veracity corpus used in the TruthGuard project.

## Dataset summary

| Item | Value |
|---|---|
| Rows | 1400 |
| Classes | 700 `fake` / 700 `real` |
| Text unit | Headline only (article body **not** included in this release) |
| Language | Tamil (native script) |

## Columns

- `id`: original integer id
- `headline`: Tamil headline text
- `label`: 0 = fake, 1 = real
- `label_name`: `fake` / `real`
- `source`: publishing outlet
- `url`: source URL when available
- `char_count`, `word_count`
- `random_split`: `train` / `dev` / `test` (stratified 80/10/10, seed 42)
- `hard_split`: `train` / `dev` / `test` / `unused` (source-disjoint protocol; unused = held out of hard train due to FAKE undersampling or unassigned)

## Important confound (read before training)

Every source is class-pure (fact-checkers → FAKE; mainstream news → REAL). Random-split near-ceiling accuracy largely reflects **source/style separability**. Prefer the **hard** (source-disjoint) split for primary claims.

Hard test sources: PuthiyaThalaimurai, Vikatan-TamilNadu (REAL); BOOMTamil-FactCheck, NewscheckerTamil (FAKE).

## Recommended citation

TruthGuard: Dual-pipeline Tamil/Tanglish fake-news detection (VIT SCOPE). Please cite the accompanying paper when available.

## Ethical notes

Public web pages only. No social-media user handles. Downstream use for defensive fact-checking research.

## License

`other` — verify redistribution rights for outlet content before public Hugging Face upload; prefer CC-BY-4.0 if cleared, else research-only.
