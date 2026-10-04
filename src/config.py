"""
Central config. Loads settings from .env so no secrets live in code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# project root = one level up from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "12"))
SECONDS_BETWEEN_CALLS = 60.0 / max(REQUESTS_PER_MINUTE, 1)

# Data paths
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ANNOTATED_DIR = ROOT_DIR / "data" / "annotated"
LID_DATASET_DIR = ROOT_DIR / "data" / "lid_dataset"

RAW_CSV = RAW_DIR / "youtube_comments.csv"          # your input file
CLEANED_CSV = PROCESSED_DIR / "cleaned_comments.csv"
ANNOTATED_JSONL = ANNOTATED_DIR / "annotated_comments.jsonl"   # checkpoint file
ANNOTATED_CSV = ANNOTATED_DIR / "annotated_comments.csv"       # flat export

# Label set used for token-level language annotation.
# Kept small and linguistically defensible for a code-mixed Tamil-English
# social-media LID task (loosely follows the LinCE / DravidianLangTech
# tagset so it's directly reusable for MuRIL/IndicBERT fine-tuning later).
LABELS = ["TA", "EN", "MIX", "NE", "O"]
LABEL_DESCRIPTIONS = {
    "TA": "Tamil word, in Tamil script OR romanized/transliterated Tamil (e.g. 'semma', 'da', 'super ah irukku' -> 'irukku')",
    "EN": "English word",
    "MIX": "A single token that blends Tamil and English morphology (e.g. 'loveah', 'kekkalam' when English-root+Tamil-suffix)",
    "NE": "Named entity: person, place, brand, movie/song title, channel name - language-neutral",
    "O": "Other: numbers, emojis, punctuation, hashtags, @mentions, URLs, pure symbols",
}

# HuggingFace checkpoints used by src/train_lid_tagger.py to fine-tune a
# token-classification LID tagger on data/lid_dataset/*.conll.
# (Not run automatically - needs GPU + `pip install -r requirements-train.txt`.)
MODEL_REGISTRY = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",
}
TRAINED_MODELS_DIR = ROOT_DIR / "models"

# --- Evidence-Aware Fake News Detection pipeline (src/verification/) ---
# Web search provider used for evidence retrieval. Only "serper" is
# actually implemented right now (see src/verification/web_search.py) -
# the interface is provider-agnostic so more can be added later.
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
CACHE_DIR = ROOT_DIR / "data" / "cache"

# --- Fake news BASELINE classifiers (src/fake_news_classifier/) ---
# Fine-tuned MuRIL/IndicBERT text classifiers (headline -> REAL/FAKE),
# trained on a labeled Tamil fake-news headlines dataset, to compare
# against the evidence-aware LLM pipeline in src/verification/.
FAKE_NEWS_RAW_CSV = RAW_DIR / "Tamil-News-Headlines.csv"
FAKE_NEWS_DATA_DIR = ROOT_DIR / "data" / "fake_news"
FAKE_NEWS_LABELS = ["REAL", "FAKE"]  # Authenticity 0 -> REAL, 1 -> FAKE
FAKE_NEWS_MODELS_DIR = ROOT_DIR / "models_fake_news"
