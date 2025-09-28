# app.py — Sentiment Analysis Interactive App
import os, io, pickle, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

st.set_page_config(page_title='Sentiment Analysis', layout='wide')

# ---------- utilities ----------
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'http\S+|www\S+', ' ', s)
    s = re.sub(r'@[A-Za-z0-9_]+', ' ', s)
    s = re.sub(r'#\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tiny_sample_df():
    data = {
        'text': [
            'I absolutely love this product!',
            'Worst experience ever. Totally disappointed.',
            'It’s okay, nothing special.',
            'Amazing support and quick delivery!',
            'Terrible quality, do not buy.',
            'Pretty good overall.'
        ],
        'label': ['positive','negative','negative','positive','negative','positive']
    }
    return pd.DataFrame(data)

# Optional: use a larger built-in sample if present next to the app
def maybe_load_large_sample():
    for name in ['sentiment_sample_large.csv', os.path.join('data','sentiment_sample_large.csv')]:
        if os.path.exists(name):
            try:
                df = pd.read_csv(name)
                if {'text','label'}.issubset(df.columns):
                    st.info(f'Loaded bundled sample: {name} ({len(df)} rows)')
                    return df
            except Exception:
                pass
    return None

# ---------- sidebar: data input ----------
st.sidebar.header('Dataset')
uploader = st.sidebar.file_uploader('Upload CSV with columns: text,label (label: positive/negative)', type=['csv'])

if uploader is not None:
    df = pd.read_csv(uploader)
    st.success('Loaded uploaded CSV.')
else:
    df_large = maybe_load_large_sample()
    if df_large is not None:
        df = df_large
    else:
        st.info('Using built-in tiny sample. Upload your dataset to train on more data.')
        df = tiny_sample_df()

# validate & clean
if 'text' not in df.columns:
    st.error("CSV must contain a 'text' column.")
    st.stop()

label_col = 'label' if 'label' in df.columns else None
df['text'] = df['text'].astype(str).apply(clean_text)
df = df[df['text'].str.len() > 0].reset_index(drop=True)

st.write('Preview', df.head())

# ---------- train/test split ----------
test_size = st.sidebar.slider('Validation size', 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input('Random state', 0, 9999, 42, 1)
X = df['text']
y = df[label_col] if label_col else None

if label_col:
    # 類別分佈提示
    dist = df[label_col].value_counts().to_frame('count')
    dist['ratio'] = (dist['count'] / dist['count'].sum()).round(3)
    st.caption("Label distribution"); st.dataframe(dist)

    if df[label_col].nunique() < 2:
        st.error("Label column must contain at least two classes.")
        st.stop()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
else:
    X_train = X_valid = X
    y_train = y_valid = None

# ---------- vectorizer (robust for tiny datasets) ----------
max_features = st.sidebar.number_input('TF-IDF max_features', 1000, 200000, 20000, 1000)
min_df_ui = st.sidebar.number_input('min_df (≥N docs)', 1, 10, 1, 1)
max_df_ui = st.sidebar.slider('max_df (≤ratio of docs)', 0.5, 1.0, 1.0, 0.05)
use_stop = st.sidebar.checkbox('Use English stopwords', False)

def make_vectorizer(min_df, max_df, stop):
    return TfidfVectorizer(
        stop_words=('english' if stop else None),
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df
    )

vectorizer = make_vectorizer(min_df_ui, max_df_ui, use_stop)

try:
    X_train_vec = vectorizer.fit_transform(X_train)
except ValueError:
    vectorizer = make_vectorizer(min_df=1, max_df=1.0, stop=False)
    X_train_vec = vectorizer.fit_transform(X_train)

X_valid_vec = vectorizer.transform(X_valid)

vocab_size = len(getattr(vectorizer, "vocabulary_", {}))
st.caption(f"Vocabulary size: {vocab_size}")
if vocab_size < 20:
    st.warning("Vocabulary is small. Try min_df=1, disable stopwords, or use more data.")

# ---------- models for evaluation ----------
model_name = st.sidebar.selectbox('Model', ['Logistic Regression','Naive Bayes'])
if model_name == 'Naive Bayes':
    model = MultinomialNB()
else:
    model = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)

if label_col:
    # Train on train split
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_valid_vec)

    # Metrics
    acc = accuracy_score(y_valid, preds)
    st.subheader("Validation Metrics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric('Accuracy', f'{acc:.4f}')
    with m2:
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_valid_vec)[:, 1]
                # Map to binary if exactly 2 classes
                if y_valid.nunique() == 2:
                    positive = sorted(y_valid.unique())[-1]
                    y_bin = (y_valid == positive).astype(int)
                    auc = roc_auc_score(y_bin, probs)
                    st.metric('ROC-AUC', f'{auc:.3f}')
        except Exception:
            pass

    # Confusion Matrix (table + heatmap)
    labels = sorted(y_valid.unique())
    cm = confusion_matrix(y_valid, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.subheader('Confusion Matrix')
    st.dataframe(cm_df)
    fig = px.imshow(cm_df, text_auto=True, title=f'Confusion Matrix — {model_name}', color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_valid, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T)

# ---------- Fit a final model on ALL labeled data for LIVE inference ----------
if label_col:
    vectorizer_prod = make_vectorizer(min_df_ui, max_df_ui, use_stop)
    try:
        X_all_vec = vectorizer_prod.fit_transform(X)
    except ValueError:
        vectorizer_prod = make_vectorizer(min_df=1, max_df=1.0, stop=False)
        X_all_vec = vectorizer_prod.fit_transform(X)

    if model_name == 'Naive Bayes':
        model_prod = MultinomialNB()
    else:
        model_prod = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)

    model_prod.fit(X_all_vec, y)
else:
    vectorizer_prod = None
    model_prod = None

# ---------- live inference (use the FULL-DATA model) ----------
st.subheader('Try a sentence')
user_text = st.text_input('Enter text to predict sentiment', 'I absolutely love this product')

if user_text:
    if not label_col:
        st.info("Upload a labeled dataset to enable predictions.")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer_prod.transform([cleaned])
        if vec.nnz == 0:
            st.warning("Sentence has no overlap with vocabulary. Set min_df=1, disable stopwords, or use more data.")
        else:
            pred = model_prod.predict(vec)[0]
            st.write('Prediction:', f'**{pred}**')

# ---------- download artifacts ----------
st.subheader('Download trained artifacts')
if label_col:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('Download model.pkl', data=pickle.dumps(model), file_name='sentiment_model.pkl')
    with c2:
        st.download_button('Download tfidf.pkl', data=pickle.dumps(vectorizer), file_name='tfidf.pkl')

