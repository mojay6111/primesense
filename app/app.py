"""
app.py
------
Flask API for primesense — Amazon Prime Video Sentiment Analysis.
Serves both a web UI and a REST API endpoint for sentiment prediction.
"""

import sys
import os

# Allow imports from project root (src/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.predict import load_model, predict_sentiment

# ── Load config ───────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

API = CONFIG["api"]

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Load the default model once at startup (not on every request)
print("📦 Loading model at startup...")
PIPELINE = load_model(API["default_model"])
print("✅ Model ready.")


# ── Routes ────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    """Serve the web UI. Handles form submissions for browser use."""
    sentiment = None
    review_text = None

    if request.method == "POST":
        review_text = request.form.get("review_text", "").strip()
        if review_text:
            try:
                result = predict_sentiment(review_text, pipeline=PIPELINE)
                sentiment = result["sentiment"]
            except Exception as e:
                sentiment = f"Error: {str(e)}"

    return render_template("index.html",
                            sentiment=sentiment,
                            review_text=review_text)


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for sentiment prediction.

    Expects JSON:  { "text": "your review here" }
    Returns JSON:  { "sentiment": "positive", "model": "svm" }
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
            "sentiment":    result["sentiment"],
            "model":        result["model"],
            "cleaned_text": result["cleaned_text"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — useful for deployment monitoring."""
    return jsonify({
        "status":  "ok",
        "project": CONFIG["project"]["name"],
        "version": CONFIG["project"]["version"],
        "model":   API["default_model"]
    })


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=API["host"],
        port=API["port"],
        debug=API["debug"]
    )
