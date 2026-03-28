# primesense 🎬

> NLP-powered sentiment analysis on Amazon Prime Video user reviews.
> Classifies reviews as **Positive**, **Neutral**, or **Negative** using
> classical ML models (SVM, Naive Bayes, Random Forest) and BERT.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Running the API](#running-the-api)
- [Running Tests](#running-tests)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Overview

In a competitive streaming landscape dominated by Netflix and Disney+,
understanding what users feel about content is a strategic advantage.
**primesense** processes 233K+ Amazon Prime Video reviews and builds
sentiment classifiers that can be queried via a REST API or web interface.

**Business objectives:**
- Identify recurring complaints and praise in user reviews
- Track sentiment trends over time and across content categories
- Provide a deployable API for real-time sentiment scoring

---

## Project Structure
```
primesense/
├── app/                        # Flask API & web UI
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/
│   ├── raw/                    # Original data (not tracked by git)
│   └── processed/              # Cleaned data (not tracked by git)
├── models/                     # Saved model pipelines (not tracked by git)
├── notebooks/                  # Jupyter notebooks
├── reports/
│   └── figures/                # All visualizations
├── src/                        # Core reusable modules
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning & sentiment labelling
│   ├── train.py                # Model training & evaluation
│   └── predict.py              # Inference utilities
├── tests/                      # Unit tests
│   └── test_predict.py
├── config.yaml                 # Central project configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — McAuley Lab
- **Category:** Movies & TV (filtered to Prime Video streaming only)
- **Size:** ~233K reviews after filtering
- **Features used:** review text, star rating, timestamp, product metadata

**Sentiment mapping:**

| Star Rating | Sentiment |
|---|---|
| 4 – 5 ⭐ | Positive |
| 3 ⭐ | Neutral |
| 1 – 2 ⭐ | Negative |

---

## Methodology

### Preprocessing
- Lowercasing, URL & HTML removal, special character stripping
- Stopword removal, lemmatization
- TF-IDF vectorization (50K features, unigrams + bigrams)

### Models
| Model | Accuracy |
|---|---|
| Naive Bayes (baseline) | ~82% |
| Random Forest (baseline) | ~86% |
| Linear SVM (baseline) | ~87% |
| Tuned SVM | ~88% |
| BERT (fine-tuned) | ~91% |

### Handling Class Imbalance
- Dataset is heavily skewed toward positive reviews
- Addressed via undersampling and class-weighted training

---

## Results

- **Best classical model:** Tuned Linear SVM at ~88% accuracy
- **Best overall model:** Fine-tuned BERT at ~91% accuracy
- **Key finding:** Neutral class is the hardest to classify across all models
- Visualizations available in `reports/figures/`

---

## Getting Started

### Prerequisites
- Python 3.11+
- Conda environment recommended

### Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/primesense.git
cd primesense

# Create and activate environment
conda create -n primesense python=3.11
conda activate primesense

# Install dependencies
pip install -r requirements.txt
```

---

## Running the API
```bash
cd app
python app.py
```

The app runs at `http://localhost:5000`

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/predict` | JSON sentiment prediction |
| GET | `/health` | API health check |

**Example API call:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely loved this series, best thing on Prime!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "model": "svm",
  "cleaned_text": "absolutely loved series best thing prime"
}
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Future Work

- [ ] Deploy to Render / Hugging Face Spaces
- [ ] Add BERT inference endpoint to the API
- [ ] Build a Streamlit dashboard for sentiment trend visualization
- [ ] Add aspect-based sentiment analysis (content, pricing, UI)
- [ ] Expand dataset to include other streaming platforms

---

## Contributors

| Name | GitHub |
|---|---|
| Edwin George | [@EdwinGeorge](https://github.com/) |
| Moses Musyoki | [@MosesMusyoki](https://github.com/) |
| Nelima Wanyama | [@NelimaWanyama](https://github.com/) |
| Spencer Lugalia | [@SpencerLugalia](https://github.com/) |
| Wambui Munene | [@WambuiMunene](https://github.com/) |
