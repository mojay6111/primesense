"""
predict.py
----------
Inference utilities for primesense.
"""

import yaml
import joblib
from pathlib import Path
from src.preprocess import full_preprocess, CONFIG

# Reuse CONFIG already loaded in preprocess.py
CFG = CONFIG["models"]


def load_model(model_type: str = None):
    if model_type is None:
        model_type = CONFIG["api"]["default_model"]

    # Build absolute path to model file
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    paths = {
        "svm": PROJECT_DIR / CFG["svm"]["saved_path"],
        "nb" : PROJECT_DIR / CFG["naive_bayes"]["saved_path"],
        "rf" : PROJECT_DIR / CFG["random_forest"]["saved_path"],
    }

    if model_type not in paths:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from: {list(paths.keys())}"
        )

    path = paths[model_type]
    print(f"📦 Loading model: {model_type} from {path}")
    return joblib.load(path)


def predict_sentiment(text: str, pipeline=None, model_type: str = None) -> dict:
    if pipeline is None:
        pipeline = load_model(model_type)

    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    cleaned    = full_preprocess(text)
    prediction = pipeline.predict([cleaned])[0]

    return {
        "sentiment"   : prediction,
        "cleaned_text": cleaned,
        "model"       : model_type or CONFIG["api"]["default_model"]
    }


def predict_batch(texts: list, pipeline=None, model_type: str = None) -> list:
    if pipeline is None:
        pipeline = load_model(model_type)
    return [predict_sentiment(t, pipeline=pipeline,
                               model_type=model_type) for t in texts]
