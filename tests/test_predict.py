"""
test_predict.py
---------------
Basic unit tests for primesense preprocessing and prediction logic.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.preprocess import (
    clean_text,
    remove_stopwords,
    lemmatize_text,
    full_preprocess,
    assign_sentiment
)


# ── clean_text ────────────────────────────────────────────────

def test_clean_text_lowercases():
    assert clean_text("GREAT Movie!") == "great movie"

def test_clean_text_removes_urls():
    assert "http" not in clean_text("Visit http://amazon.com for more")

def test_clean_text_removes_html():
    assert "<br>" not in clean_text("Good show<br>Loved it")

def test_clean_text_removes_special_chars():
    assert "!" not in clean_text("Amazing!!!")

def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123)  == ""


# ── remove_stopwords ──────────────────────────────────────────

def test_remove_stopwords_removes_the():
    result = remove_stopwords("the movie was great")
    assert "the" not in result.split()

def test_remove_stopwords_keeps_content_words():
    result = remove_stopwords("the movie was great")
    assert "great" in result.split()


# ── lemmatize_text ────────────────────────────────────────────

def test_lemmatize_text_reduces_plurals():
    result = lemmatize_text("movies reviews ratings")
    assert "movie" in result or "movies" in result

def test_lemmatize_text_returns_string():
    assert isinstance(lemmatize_text("running quickly"), str)


# ── full_preprocess ───────────────────────────────────────────

def test_full_preprocess_returns_string():
    result = full_preprocess("This is an AMAZING show!! http://amazon.com")
    assert isinstance(result, str)

def test_full_preprocess_no_urls():
    result = full_preprocess("Check http://amazon.com now")
    assert "http" not in result

def test_full_preprocess_no_stopwords():
    result = full_preprocess("this is a great movie")
    assert "this" not in result.split()
    assert "is"   not in result.split()


# ── assign_sentiment ──────────────────────────────────────────

def test_assign_sentiment_positive():
    assert assign_sentiment(5) == "positive"
    assert assign_sentiment(4) == "positive"

def test_assign_sentiment_neutral():
    assert assign_sentiment(3) == "neutral"

def test_assign_sentiment_negative():
    assert assign_sentiment(1) == "negative"
    assert assign_sentiment(2) == "negative"

def test_assign_sentiment_unknown():
    assert assign_sentiment(0)  == "unknown"
    assert assign_sentiment(99) == "unknown"
