"""
preprocess.py
-------------
Text cleaning and preprocessing utilities for primesense.
All notebooks and the Flask app import from here — one source of truth.
"""

import re
import yaml
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first use
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Individual cleaning steps ──────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove URLs, HTML tags, and special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"<.*?>", "", text)                 # HTML tags
    text = re.sub(r"[^a-z\s]", "", text)              # Non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()          # Extra whitespace
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def lemmatize_text(text: str) -> str:
    """Lemmatize each token."""
    return " ".join(LEMMATIZER.lemmatize(w) for w in text.split())


def full_preprocess(text: str) -> str:
    """Full pipeline: clean → remove stopwords → lemmatize."""
    return lemmatize_text(remove_stopwords(clean_text(text)))


# ── Sentiment labelling ───────────────────────────────────────

def assign_sentiment(rating: float) -> str:
    """
    Map a star rating to a sentiment label using config.yaml rules.
    Returns 'positive', 'neutral', or 'negative'.
    """
    cfg = CONFIG["sentiment"]
    if rating in cfg["positive_ratings"]:
        return "positive"
    elif rating in cfg["neutral_ratings"]:
        return "neutral"
    elif rating in cfg["negative_ratings"]:
        return "negative"
    return "unknown"


# ── DataFrame-level helpers ───────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame,
                          text_col: str = "text",
                          rating_col: str = "rating") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.

    Args:
        df:         Input DataFrame with review text and ratings.
        text_col:   Name of the column containing review text.
        rating_col: Name of the column containing star ratings.

    Returns:
        DataFrame with added 'cleaned_text' and 'sentiment' columns.
    """
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(full_preprocess)
    df["sentiment"] = df[rating_col].apply(assign_sentiment)
    df = df[df["cleaned_text"].str.strip() != ""]   # Drop empty reviews
    return df
