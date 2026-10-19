# TruthGuard 1400 Evaluation Suite Results
FAKE = positive class. Existing seed-42 splits unchanged.
## 1. Style-only logistic baseline
| Model | Acc | Bal.Acc | P_F | R_F | F1_F | Macro-F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Style-only LR (random) | 0.8929 | 0.8929 | 0.8481 | 0.9571 | 0.8993 | 0.8924 | 12 | 3 |
| Style-only LR (hard) | 0.5759 | 0.4137 | 0.0486 | 0.2188 | 0.0795 | 0.402 | 137 | 25 |
## 2. Seed-42 models (reference; not overwritten)
### hard split
| Model | Acc | Bal.Acc | P_F | R_F | F1_F | Macro-F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indicbert-v2 | 0.8665 | 0.6432 | 0.2791 | 0.375 | 0.32 | 0.623 | 31 | 20 |
| muril | 0.6099 | 0.4322 | 0.0534 | 0.2188 | 0.0859 | 0.419 | 124 | 25 |
| mbert | 0.6126 | 0.4337 | 0.0538 | 0.2188 | 0.0864 | 0.4203 | 123 | 25 |
| xlm-roberta | 0.3246 | 0.2907 | 0.0331 | 0.25 | 0.0584 | 0.2659 | 234 | 24 |
| TF-IDF + MultinomialNB | 0.8455 | 0.8021 | 0.32 | 0.75 | 0.4486 | 0.6794 | 51 | 8 |
| TF-IDF + LogisticRegression | 0.8743 | 0.6617 | 0.3095 | 0.4062 | 0.3514 | 0.6409 | 29 | 19 |
| TF-IDF + LinearSVC | 0.8063 | 0.6529 | 0.2083 | 0.4688 | 0.2885 | 0.5882 | 57 | 17 |
### random split
| Model | Acc | Bal.Acc | P_F | R_F | F1_F | Macro-F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indicbert-v2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| muril | 0.9929 | 0.9929 | 0.9859 | 1.0 | 0.9929 | 0.9929 | 1 | 0 |
| mbert | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| xlm-roberta | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| TF-IDF + MultinomialNB | 0.8643 | 0.8643 | 0.8806 | 0.8429 | 0.8613 | 0.8642 | 8 | 11 |
| TF-IDF + LogisticRegression | 0.8429 | 0.8429 | 0.9444 | 0.7286 | 0.8226 | 0.8408 | 3 | 19 |
| TF-IDF + LinearSVC | 0.8786 | 0.8786 | 0.9492 | 0.8 | 0.8682 | 0.8778 | 3 | 14 |
## 3. Suspicious-result checks
```json
{
  "checks": [],
  "muril_vs_mbert": {
    "n": 382,
    "n_predictions_differ": 109,
    "n_predictions_same": 273,
    "agreement_rate": 0.7146596858638743,
    "muril_sha256": "50c5b17fa3712eb6dec7204c24695a66b11623b057b5062ed8167829cf43c26a",
    "mbert_sha256": "c3bfa7b8b4489e3a96ad15954e312efa3d36079d4b22e154485ec68c817a8c00",
    "identical_files": false,
    "muril_path": "/Users/samyamehta/Documents/Codes/College/Project1/CodeMix/eval_results_1400/predictions/muril_hard_seed42_test.csv",
    "mbert_path": "/Users/samyamehta/Documents/Codes/College/Project1/CodeMix/eval_results_1400/predictions/mbert_hard_seed42_test.csv",
    "conclusion": "Distinct models with overlapping preds"
  },
  "xlm_roberta_hard": {
    "n_real": 350,
    "share_real_predicted_fake": 0.6685714285714286,
    "n_real_pred_fake": 234,
    "overall_pred_counts": {
      "FAKE": 242,
      "REAL": 140
    }
  },
  "xlm_roberta_train_log": {
    "log_path": "data/fake_news_1400_hard/_train_xlmr_log.txt",
    "logged_train_loss_points": [
      {
        "loss": 0.6446,
        "epoch": 0.5
      },
      {
        "loss": 0.5072,
        "epoch": 1.0
      },
      {
        "loss": 0.3839,
        "epoch": 1.5
      },
      {
        "loss": 0.2524,
        "epoch": 2.0
      },
      {
        "loss": 0.2163,
        "epoch": 2.5
      },
      {
        "loss": 0.08848,
        "epoch": 3.0
      },
      {
        "loss": 0.05253,
        "epoch": 3.5
      },
      {
        "loss": 0.07473,
        "epoch": 4.0
      }
    ],
    "logged_eval_points": [
      {
        "eval_loss": 0.3397,
        "eval_accuracy": 1.0,
        "epoch": 1.0
      },
      {
        "eval_loss": 0.1586,
        "eval_accuracy": 1.0,
        "epoch": 2.0
      },
      {
        "eval_loss": 0.005471,
        "eval_accuracy": 1.0,
        "epoch": 3.0
      },
      {
        "eval_loss": 0.002199,
        "eval_accuracy": 1.0,
        "epoch": 4.0
      }
    ]
  }
}
```
## 3b. XLM-RoBERTa extra run (lr=1e-5, warmup=0.1, epochs=6)
_Not finished yet._
## 4. Multi-seed mean ± std (seeds 1–5)
_Pending multiseed completion._
## 5. Hard-split 95% bootstrap CIs
| Model |
| --- |
| indicbert-v2 |
| muril |
| mbert |
| xlm-roberta |
| TF-IDF_plus_MultinomialNB |
| TF-IDF_plus_LogisticRegression |
| TF-IDF_plus_LinearSVC |
| style_lr_hard |
## 6. Error analysis (hard)
```json
{
  "indicbert-v2": {
    "model": "indicbert-v2",
    "fake_source_recall": {
      "BOOMTamil-FactCheck": {
        "n": 24,
        "recall": 0.20833333333333334,
        "TP": 5,
        "FN": 19
      },
      "NewscheckerTamil": {
        "n": 8,
        "recall": 0.875,
        "TP": 7,
        "FN": 1
      }
    },
    "n_FN_total": 20,
    "n_FP_total": 31,
    "FN_marker_rates_all": {
      "has_question": 0.0,
      "has_paravum": 0.0,
      "has_unmai": 0.0
    }
  },
  "TF-IDF_NB": {
    "model": "TF-IDF_NB",
    "fake_source_recall": {
      "BOOMTamil-FactCheck": {
        "n": 24,
        "recall": 0.7916666666666666,
        "TP": 19,
        "FN": 5
      },
      "NewscheckerTamil": {
        "n": 8,
        "recall": 0.625,
        "TP": 5,
        "FN": 3
      }
    },
    "n_FN_total": 8,
    "n_FP_total": 51,
    "FN_marker_rates_all": {
      "has_question": 0.375,
      "has_paravum": 0.0,
      "has_unmai": 0.0
    }
  }
}
```
False +/- CSVs under `eval_results_1400/error_analysis/`.
## 7. Data / tokenizer stats
|  |
|  |
|  |
|  |
|  |
|  |
Near-duplicates: `{
  "threshold": 0.9,
  "n_pairs": 2,
  "method": "difflib.SequenceMatcher.ratio"
}`
Dates: `{
  "date_columns_found": [],
  "message": "No publication date column in tamil_news_dataset_1400_balanced.csv; cannot report date range per class.",
  "columns": [
    "id",
    "headline",
    "text",
    "label",
    "label_name",
    "source",
    "url",
    "char_count",
    "word_count",
    "investigation_details"
  ]
}`
## 8. Romanisation OOD (no retrain)
| Model | Acc | P_F | R_F | F1_F | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| indicbert-v2 | 0.3927 | 0.069 | 0.5 | 0.1212 | 216 | 16 |
| muril | 0.8874 | 0.0 | 0.0 | 0.0 | 11 | 32 |
| mbert | 0.3429 | 0.03 | 0.2188 | 0.0528 | 226 | 25 |
| xlm-roberta | 0.7147 | 0.0575 | 0.1562 | 0.084 | 82 | 27 |
| indicbert-v2 | 0.7 | 0.8684 | 0.4714 | 0.6111 | 5 | 37 |
| muril | 0.8143 | 0.8056 | 0.8286 | 0.8169 | 14 | 12 |
| mbert | 0.9143 | 0.9677 | 0.8571 | 0.9091 | 2 | 10 |
| xlm-roberta | 0.8786 | 0.8442 | 0.9286 | 0.8844 | 12 | 5 |
## 9. Hugging Face export
Files in `eval_results_1400/huggingface/` (CSV/Parquet + README.md).
## Surprises / things that looked wrong
- XLM-RoBERTa predicts FAKE for 66.9% of REAL hard-test headlines.
