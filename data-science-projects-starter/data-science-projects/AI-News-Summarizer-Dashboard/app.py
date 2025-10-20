# AI News Summarizer Dashboard
# Streamlit app for summarizing news/content from URLs, pasted text, or uploaded files.

import io
import re
from typing import Optional

import streamlit as st

# Optional dependencies handled gracefully
try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
except Exception:
    PlaintextParser = Tokenizer = LexRankSummarizer = None

# Transformers (huggingface) for abstractive summarization
from transformers import pipeline

APP_TITLE = "📰 AI News Summarizer Dashboard"
APP_DESC = (
    "Paste a URL, text, or upload a file to generate a clean summary. "
    "Supports HuggingFace transformers (abstractive) with a classical extractive fallback."
)

st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="centered")

# -----------------------------
# Utilities
# -----------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Keep it reasonably sized for models
    return text

def read_uploaded_file(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text
            file.seek(0)
            return extract_text(io.BytesIO(data))
        except Exception as e:
            st.warning(f"PDF 解析失敗：{e}")
            return ""
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.warning(f"DOCX 解析失敗：{e}")
            return ""
    # Fallback – try decoding as text
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_from_url(url: str) -> str:
    if trafilatura is None:
        st.info("未安裝 trafilatura，改用貼上文字或上傳檔案。")
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) or ""
    except Exception as e:
        st.warning(f"擷取失敗：{e}")
        return ""

def sumy_lexrank_summary(text: str, sentences: int = 5) -> str:
    if not (PlaintextParser and Tokenizer and LexRankSummarizer):
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sents = summarizer(parser.document, sentences)
        return " ".join(str(s) for s in sents)
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    # device_map="auto" lets transformers choose GPU if available; falls back to CPU.
    return pipeline(
        "summarization",
        model=model_name,
        device_map="auto",
    )

def hf_summarize(text: str, model_name: str, max_words: int = 220) -> str:
    # Transformers use tokens; set a reasonable token budget.
    # Rough token estimate ~ 1.3x words
    max_tokens = int(max_words * 1.3)
    min_tokens = max(30, int(max_tokens * 0.4))
    summarizer = load_summarizer(model_name)
    # Many models have max input token limits; chunk if needed.
    chunks = chunk_text_for_model(text, max_chunk_chars=2500)
    outputs = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=max_tokens, min_length=min_tokens, do_sample=False, truncation=True)
        outputs.append(out[0]["summary_text"])
    return " ".join(outputs)

def chunk_text_for_model(text: str, max_chunk_chars: int = 2500):
    text = text.strip()
    if len(text) <= max_chunk_chars:
        return [text]
    # Try to split on sentence boundaries
    sentences = re.split(r'(?<=[。！？.!?])\s+', text)
    current = ""
    chunks = []
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chunk_chars:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

# -----------------------------
# UI
# -----------------------------

st.title(APP_TITLE)
st.caption(APP_DESC)

with st.sidebar:
    st.subheader("設定 / Settings")
    model_choice = st.selectbox(
        "HuggingFace 模型 (Abstractive)",
        options=[
            "facebook/bart-large-cnn",
            "philschmid/bart-large-cnn-samsum",
            "sshleifer/distilbart-cnn-12-6",
            "google/pegasus-xsum",
            "t5-small",
        ],
        index=0,
        help="選擇摘要模型。若遇到錯誤，請換一個輕量模型。",
    )
    target_words = st.slider("摘要長度 (估計字數)", 80, 400, 180, 10)

tab1, tab2, tab3 = st.tabs(["🔗 從網址擷取", "📝 直接貼上文字", "📁 上傳檔案"])

with tab1:
    url = st.text_input("貼上新聞或文章網址 (URL)", value="", placeholder="https://example.com/news")
    if st.button("擷取並摘要", type="primary", use_container_width=True, key="summ_from_url"):
        if not url:
            st.warning("請輸入網址。")
        else:
            with st.spinner("擷取與摘要中…"):
                content = extract_from_url(url)
                content = clean_text(content)
                if not content or len(content) < 120:
                    st.error("擷取內容過短或失敗，請改用貼上文字或上傳檔案。")
                else:
                    try:
                        summary = hf_summarize(content, model_choice, max_words=target_words)
                    except Exception as e:
                        st.warning(f"HuggingFace 摘要失敗，改用傳統摘要。錯誤：{e}")
                        summary = sumy_lexrank_summary(content) or "（備援摘要也失敗，請嘗試不同輸入或模型。）"
                    st.success("完成！")
                    st.subheader("摘要 / Summary")
                    st.write(summary)
                    with st.expander("原始擷取內容"):
                        st.write(content[:5000])

with tab2:
    user_text = st.text_area("貼上你要摘要的內容", height=220, placeholder="在此貼上文章或筆記…")
    if st.button("產生摘要", type="primary", use_container_width=True, key="summ_from_text"):
        text = clean_text(user_text)
        if len(text) < 50:
            st.warning("內容太短，請提供更完整的文字。")
        else:
            with st.spinner("產生摘要中…"):
                try:
                    summary = hf_summarize(text, model_choice, max_words=target_words)
                except Exception as e:
                    st.warning(f"HuggingFace 摘要失敗，改用傳統摘要。錯誤：{e}")
                    summary = sumy_lexrank_summary(text) or "（備援摘要也失敗，請嘗試不同輸入或模型。）"
            st.subheader("摘要 / Summary")
            st.write(summary)

with tab3:
    upload = st.file_uploader("上傳 .txt / .md / .pdf / .docx", type=["txt", "md", "pdf", "docx"])
    if upload and st.button("讀取並摘要", type="primary", use_container_width=True, key="summ_from_file"):
        with st.spinner("讀取與摘要中…"):
            content = read_uploaded_file(upload)
            content = clean_text(content)
            if len(content) < 80:
                st.warning("檔案內容過短或解析失敗。")
            else:
                try:
                    summary = hf_summarize(content, model_choice, max_words=target_words)
                except Exception as e:
                    st.warning(f"HuggingFace 摘要失敗，改用傳統摘要。錯誤：{e}")
                    summary = sumy_lexrank_summary(content) or "（備援摘要也失敗，請嘗試不同輸入或模型。）"
                st.subheader("摘要 / Summary")
                st.write(summary)

st.markdown("---")
st.caption("Tip: 若遇到模型下載過慢或失敗，可改選擇較小的模型 (如 t5-small)，或在本地先行下載模型。")
