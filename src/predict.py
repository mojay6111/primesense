"""
predict.py
----------
Inference utilities for primesense.
Loads a trained pipeline from disk and predicts sentiment
on new review text. Used by the Flask app and notebooks.
"""

import yaml
import joblib
from src.preprocess import full_preprocess

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CFG = CONFIG["models"]


# ── Model loader ──────────────────────────────────────────────

def load_model(model_type: str = None):
    """
    Load a trained sklearn pipeline from disk.

    Args:
        model_type: One of 'svm', 'nb', 'rf'.
                    Defaults to config.yaml api.default_model.

    Returns:
        Loaded sklearn pipeline.
    """
    if model_type is None:
        model_type = CONFIG["api"]["default_model"]

    paths = {
        "svm": CFG["svm"]["saved_path"],
        "nb":  CFG["naive_bayes"]["saved_path"],
        "rf":  CFG["random_forest"]["saved_path"],
    }

    if model_type not in paths:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from: {list(paths.keys())}"
        )

    path = paths[model_type]
    print(f"📦 Loading model: {model_type} from {path}")
    return joblib.load(path)


# ── Prediction ────────────────────────────────────────────────

def predict_sentiment(text: str, pipeline=None, model_type: str = None) -> dict:
    """
    Predict the sentiment of a single review string.

    Args:
        text:       Raw review text from the user.
        pipeline:   A pre-loaded sklearn pipeline (optional).
                    If None, loads the default model from disk.
        model_type: Which model to load if pipeline is None.

    Returns:
        dict with keys:
            - 'sentiment': predicted label (positive/neutral/negative)
            - 'cleaned_text': preprocessed version of the input
            - 'model': which model was used
    """
    if pipeline is None:
        pipeline = load_model(model_type)

    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    cleaned = full_preprocess(text)
    prediction = pipeline.predict([cleaned])[0]

    return {
        "sentiment":    prediction,
        "cleaned_text": cleaned,
        "model":        model_type or CONFIG["api"]["default_model"]
    }


def predict_batch(texts: list, pipeline=None, model_type: str = None) -> list:
    """
    Predict sentiment for a list of review strings.

    Args:
        texts:      List of raw review strings.
        pipeline:   Pre-loaded pipeline (optional).
        model_type: Which model to load if pipeline is None.

    Returns:
        List of prediction dicts (same format as predict_sentiment).
    """
    if pipeline is None:
        pipeline = load_model(model_type)

    return [predict_sentiment(t, pipeline=pipeline,
                               model_type=model_type) for t in texts]
