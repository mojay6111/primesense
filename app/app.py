"""
app.py
------
Flask API for primesense — Amazon Prime Video Sentiment Analysis.
Serves both a web UI and a REST API endpoint for sentiment prediction.
"""

import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
# Works whether run from app/ or project root or Render
APP_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import yaml
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.predict import load_model, predict_sentiment

# ── Load config ───────────────────────────────────────────────
CONFIG_PATH = PROJECT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

API = CONFIG["api"]

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Load model once at startup
print("📦 Loading model at startup...")
PIPELINE = load_model(API["default_model"])
print("✅ Model ready.")


# ── Routes ────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment    = None
    review_text  = None
    error        = None

    if request.method == "POST":
        review_text = request.form.get("review_text", "").strip()
        if review_text:
            try:
                result    = predict_sentiment(review_text, pipeline=PIPELINE)
                sentiment = result["sentiment"]
            except Exception as e:
                error = str(e)

    return render_template("index.html",
                           sentiment=sentiment,
                           review_text=review_text,
                           error=error)


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint.
    Expects JSON : { "text": "your review here" }
    Returns JSON : { "sentiment": "positive", "model": "svm" }
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Request must include a 'text' field."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "The 'text' field must not be empty."}), 400

    try:
        result = predict_sentiment(text, pipeline=PIPELINE)
        return jsonify({
            "sentiment"   : result["sentiment"],
            "model"       : result["model"],
            "cleaned_text": result["cleaned_text"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "project": CONFIG["project"]["name"],
        "version": CONFIG["project"]["version"],
        "model"  : API["default_model"]
    })


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host =API["host"],
        port =API["port"],
        debug=API["debug"]
    )
