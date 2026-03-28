"""
train.py
--------
Model training and saving utilities for primesense.
Trains Naive Bayes, SVM, and Random Forest pipelines
using parameters defined in config.yaml.
"""

import yaml
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocess import preprocess_dataframe

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

PP  = CONFIG["preprocessing"]
CFG = CONFIG["models"]
DAT = CONFIG["data"]


# ── TF-IDF Vectorizer (shared across baseline models) ─────────

def build_vectorizer() -> TfidfVectorizer:
    """Build a TF-IDF vectorizer from config parameters."""
    return TfidfVectorizer(
        max_features=PP["max_features"],
        min_df=PP["min_df"],
        max_df=PP["max_df"],
        ngram_range=tuple(PP["ngram_range"])
    )


# ── Pipeline builders ─────────────────────────────────────────

def build_nb_pipeline() -> Pipeline:
    """Naive Bayes pipeline."""
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf",   MultinomialNB(alpha=CFG["naive_bayes"]["alpha"]))
    ])


def build_svm_pipeline() -> Pipeline:
    """Linear SVM pipeline."""
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf",   LinearSVC(
                      C=CFG["svm"]["C"],
                      max_iter=2000
                  ))
    ])


def build_rf_pipeline() -> Pipeline:
    """Random Forest pipeline."""
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf",   RandomForestClassifier(
                      n_estimators=CFG["random_forest"]["n_estimators"],
                      random_state=CFG["random_forest"]["random_state"],
                      n_jobs=-1
                  ))
    ])


# ── Train & evaluate ──────────────────────────────────────────

def train_and_evaluate(pipeline: Pipeline,
                        X_train, X_test,
                        y_train, y_test,
                        model_name: str) -> Pipeline:
    """
    Fit a pipeline and print a classification report.

    Args:
        pipeline:   Sklearn Pipeline to train.
        X_train:    Training texts.
        X_test:     Test texts.
        y_train:    Training labels.
        y_test:     Test labels.
        model_name: Display name for logging.

    Returns:
        Fitted pipeline.
    """
    print(f"\n🚀 Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n📊 {model_name} — Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=CONFIG["sentiment"]["labels"]))
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    """Save a fitted pipeline to disk using joblib."""
    joblib.dump(pipeline, path)
    print(f"✅ Model saved → {path}")


# ── Main training entry point ─────────────────────────────────

def run_training(model: str = "all") -> None:
    """
    Load processed data, train models, and save them.

    Args:
        model: One of 'nb', 'svm', 'rf', or 'all'.
    """
    print("📂 Loading data...")
    df = pd.read_csv(CONFIG["data"]["processed"])
    df = preprocess_dataframe(df, text_col="text", rating_col="rating")

    X = df["cleaned_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DAT["test_size"],
        random_state=DAT["random_state"],
        stratify=y
    )

    if model in ("nb", "all"):
        pipe = build_nb_pipeline()
        pipe = train_and_evaluate(pipe, X_train, X_test,
                                   y_train, y_test, "Naive Bayes")
        save_model(pipe, CFG["naive_bayes"]["saved_path"])

    if model in ("svm", "all"):
        pipe = build_svm_pipeline()
        pipe = train_and_evaluate(pipe, X_train, X_test,
                                   y_train, y_test, "Linear SVM")
        save_model(pipe, CFG["svm"]["saved_path"])

    if model in ("rf", "all"):
        pipe = build_rf_pipeline()
        pipe = train_and_evaluate(pipe, X_train, X_test,
                                   y_train, y_test, "Random Forest")
        save_model(pipe, CFG["random_forest"]["saved_path"])


if __name__ == "__main__":
    run_training(model="all")
