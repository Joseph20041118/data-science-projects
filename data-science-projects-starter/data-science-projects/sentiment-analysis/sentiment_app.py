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

# ---------- sidebar: data input ----------
st.sidebar.header('Dataset')
uploader = st.sidebar.file_uploader('Upload CSV with columns: text,label (label: positive/negative)', type=['csv'])

if uploader is not None:
    df = pd.read_csv(uploader)
    st.success('Loaded uploaded CSV.')
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
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
else:
    X_train = X_valid = X
    y_train = y_valid = None

# ---------- vectorizer (robust for tiny datasets) ----------
max_features = st.sidebar.number_input('TF-IDF max_features', 1000, 200000, 20000, 1000)
min_df_ui = st.sidebar.number_input('min_df (keep terms seen in ≥N docs)', 1, 10, 1, 1)  # default 1
max_df_ui = st.sidebar.slider('max_df (drop overly common terms)', 0.5, 1.0, 1.0, 0.05)
use_stop = st.sidebar.checkbox('Use English stopwords', True)

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
    # Fallback for very small / sparse datasets
    vectorizer = make_vectorizer(min_df=1, max_df=1.0, stop=False)
    X_train_vec = vectorizer.fit_transform(X_train)

X_valid_vec = vectorizer.transform(X_valid)

#  need at least two classes to train/evaluate
if label_col and df[label_col].nunique() < 2:
    st.error("Label column must contain at least two classes (e.g., positive and negative).")
    st.stop()



# ---------- model ----------
model_name = st.sidebar.selectbox('Model', ['Logistic Regression','Naive Bayes'])
model = LogisticRegression(max_iter=2000, n_jobs=None) if model_name == 'Logistic Regression' else MultinomialNB()

if label_col:
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_valid_vec)
    acc = accuracy_score(y_valid, preds)
    st.metric('Accuracy', f'{acc:.4f}')

    # ROC-AUC if binary and proba available
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_valid_vec)[:, 1]
            # infer positive class alphabetically
            pos_label = sorted(y_valid.unique())[-1]
            y_bin = (y_valid == pos_label).astype(int)
            auc = roc_auc_score(y_bin, probs)
            st.metric('ROC-AUC', f'{auc:.3f}')
    except Exception:
        pass

    # Confusion Matrix
    labels = sorted(y_valid.unique())
    cm = confusion_matrix(y_valid, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.subheader('Confusion Matrix')
    st.dataframe(cm_df)
    fig = px.imshow(cm_df, text_auto=True, title=f'Confusion Matrix — {model_name}')
    st.plotly_chart(fig, use_container_width=True)

    # Classification report
    report = classification_report(y_valid, preds, output_dict=True, zero_division=0)
    st.subheader('Classification Report')
    st.dataframe(pd.DataFrame(report).T)

# ---------- live inference ----------
st.subheader('Try a sentence')
user_text = st.text_input('Enter text to predict sentiment', 'I love this!')
if user_text:
    vec = vectorizer.transform([clean_text(user_text)])
    pred = model.predict(vec)[0] if label_col else 'N/A (no labels to train)'
    st.write('Prediction:', f'**{pred}**')

# ---------- download artifacts ----------
st.subheader('Download trained artifacts')
if label_col:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('Download model.pkl', data=pickle.dumps(model), file_name='sentiment_model.pkl')
    with c2:
        st.download_button('Download tfidf.pkl', data=pickle.dumps(vectorizer), file_name='tfidf.pkl')
