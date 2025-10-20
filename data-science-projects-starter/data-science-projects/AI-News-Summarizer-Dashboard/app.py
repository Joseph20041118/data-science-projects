# AI News Summarizer Dashboard (Fixed loader)
# Key changes:
#  - Avoid device_map='auto' to remove 'accelerate' requirement on CPU-only
#  - Add graceful fallback to smaller models
#  - Clearer error messages

import io
import re
import streamlit as st

# Optional deps
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

from transformers import pipeline


APP_TITLE = "📰 AI News Summarizer Dashboard"
APP_DESC = (
    "Paste a URL, text, or upload a file to generate a clean summary. "
    "Uses Hugging Face transformers (abstractive) with a classical extractive fallback."
)

st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="centered")


# -----------------------------
# Utilities
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def read_uploaded_file(file) -> str:
    name = file.name.lower()
    data = file.read()

    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text
            file.seek(0)
            return extract_text(io.BytesIO(data))
        except Exception as e:
            st.warning(f"PDF parsing failed: {e}")
            return ""

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.warning(f"DOCX parsing failed: {e}")
            return ""

    # Fallback: best effort decode
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_from_url(url: str) -> str:
    if trafilatura is None:
        st.info("trafilatura is not installed. Please paste text or upload a file instead.")
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) or ""
    except Exception as e:
        st.warning(f"Extraction failed: {e}")
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
    """
    Use CPU (device=-1) to avoid requiring `accelerate`.
    If loading fails, fall back to a lighter model.
    """
    try:
        return pipeline("summarization", model=model_name, device=-1)
    except Exception as e:
        st.warning(f"Failed to load model '{model_name}'. Error: {e}")
        # Fallback to a lighter model
        fallback = "t5-small"
        if model_name != fallback:
            st.info(f"Trying fallback model: {fallback}")
            return pipeline("summarization", model=fallback, device=-1)
        # If fallback also fails, re-raise
        raise


def chunk_text_for_model(text: str, max_chunk_chars: int = 2500):
    text = text.strip()
    if len(text) <= max_chunk_chars:
        return [text]
    # split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current, chunks = "", []
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


def hf_summarize(text: str, model_name: str, max_words: int = 220) -> str:
    # Rough token estimate ~ 1.3x words
    max_tokens = int(max_words * 1.3)
    min_tokens = max(30, int(max_tokens * 0.4))
    summarizer = load_summarizer(model_name)
    chunks = chunk_text_for_model(text, max_chunk_chars=2500)
    outputs = []
    for chunk in chunks:
        out = summarizer(
            chunk,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True,
        )
        outputs.append(out[0]["summary_text"])
    return " ".join(outputs)


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_DESC)

with st.sidebar:
    st.subheader("Settings")
    model_choice = st.selectbox(
        "Hugging Face model (Abstractive)",
        options=[
            "sshleifer/distilbart-cnn-12-6",
            "facebook/bart-large-cnn",
            "google/pegasus-xsum",
            "t5-small",
        ],
        index=0,
        help="Choose a summarization model. If it fails, try a lighter one.",
    )
    target_words = st.slider("Target summary length (approx. words)", 80, 400, 180, 10)

tab1, tab2, tab3 = st.tabs(["🔗 From URL", "📝 Paste Text", "📁 Upload File"])

with tab1:
    url = st.text_input("News/article URL", value="", placeholder="https://example.com/news")
    if st.button("Extract & Summarize", type="primary", use_container_width=True, key="summ_from_url"):
        if not url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and summarizing…"):
                content = extract_from_url(url)
                content = clean_text(content)
                if not content or len(content) < 120:
                    st.error("Extracted content is too short or failed. Try pasting text or uploading a file.")
                else:
                    try:
                        summary = hf_summarize(content, model_choice, max_words=target_words)
                    except Exception as e:
                        st.warning(
                            "Hugging Face summarization failed. Falling back to classic extractive method. "
                            f"Error: {e}"
                        )
                        summary = sumy_lexrank_summary(content) or "(Fallback summarization also failed. Try different input or model.)"
                    st.subheader("Summary")
                    st.write(summary)
                    with st.expander("Extracted Content (truncated)"):
                        st.write(content[:5000])

with tab2:
    user_text = st.text_area(
        "Paste the content you want to summarize",
        height=220,
        placeholder="Paste article or notes here…",
    )
    if st.button("Generate Summary", type="primary", use_container_width=True, key="summ_from_text"):
        text = clean_text(user_text)
        if len(text) < 50:
            st.warning("Content is too short. Please provide more text.")
        else:
            with st.spinner("Summarizing…"):
                try:
                    summary = hf_summarize(text, model_choice, max_words=target_words)
                except Exception as e:
                    st.warning(
                        "Hugging Face summarization failed. Falling back to classic extractive method. "
                        f"Error: {e}"
                    )
                    summary = sumy_lexrank_summary(text) or "(Fallback summarization also failed. Try different input or model.)"
            st.subheader("Summary")
            st.write(summary)

with tab3:
    upload = st.file_uploader("Upload .txt / .md / .pdf / .docx", type=["txt", "md", "pdf", "docx"])
    if upload and st.button("Read & Summarize", type="primary", use_container_width=True, key="summ_from_file"):
        with st.spinner("Reading and summarizing…"):
            content = read_uploaded_file(upload)
            content = clean_text(content)
            if len(content) < 80:
                st.warning("File content is too short or parsing failed.")
            else:
                try:
                    summary = hf_summarize(content, model_choice, max_words=target_words)
                except Exception as e:
                    st.warning(
                        "Hugging Face summarization failed. Falling back to classic extractive method. "
                        f"Error: {e}"
                    )
                    summary = sumy_lexrank_summary(content) or "(Fallback summarization also failed. Try different input or model.)"
                st.subheader("Summary")
                st.write(summary)

st.markdown("---")
st.caption("Tip: If model download is slow or failing, switch to a smaller model (e.g., t5-small) or pre-download locally.")
