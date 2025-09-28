# Sentiment Analysis — Interactive App

An interactive sentiment analysis dashboard built with **Streamlit**.  
Upload a dataset of text (tweets, reviews, etc.), train models (Naive Bayes / Logistic Regression), visualize results, and try live predictions.

**👉 Live Demo:** https://data-science-projects-wwaoc6hzwvpkbxzde3krsv.streamlit.app/

---

## 🔹 Features
- Upload your own CSV dataset (`text,label`) or use the built-in sample  
- Automatic text cleaning (lowercasing, remove URLs, mentions, hashtags, punctuation)  
- Train/test split with configurable validation size  
- TF-IDF vectorization (adjustable max_features, min_df, max_df, stopwords toggle)  
- Models: Logistic Regression, Naive Bayes  
- Metrics: Accuracy, ROC-AUC, Confusion Matrix, Classification Report  
- Live inference: enter any sentence and get predicted sentiment instantly  
- Download trained model (`model.pkl`) and vectorizer (`tfidf.pkl`)  

---

## 🔹 Project Structure
```
sentiment-analysis/
├── sentiment_app.py      # Streamlit app (entry point)
├── requirements.txt      # Dependencies
├── sentiment_sample.csv  # Small sample dataset (optional)
└── README.md
```

---

## 🔹 Run Locally

1. Clone the project:
```bash
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the app:
```bash
streamlit run sentiment_app.py
```

4. Open http://localhost:8501 in your browser.

---

## 🔹 Sample Dataset Format
Your CSV must contain:
- `text` → the review/tweet sentence  
- `label` → sentiment class (e.g., `positive`, `negative`)  

Example (`sentiment_sample.csv`):
```csv
text,label
I absolutely love this product!,positive
Worst experience ever. Totally disappointed.,negative
Pretty good overall.,positive
```

---

## 🔹 Data Source
- Built-in tiny sample (6 rows, for demo only)  
- For larger experiments, use [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  

---

## 🔹 Author
**Joseph Wang (Mt. SAC)** — CS transfer applicant (Fall 2026)
