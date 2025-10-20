# 📰 AI News Summarizer Dashboard (English)

A **Streamlit** dashboard that summarizes content from **URLs**, **pasted text**, or **uploaded files**.
It supports **Hugging Face transformers (abstractive)** with an **extractive LexRank fallback**.

## Features
- 🔗 Extract main article text from a URL using `trafilatura`.
- 📝 Paste arbitrary text and summarize it.
- 📁 Upload `.txt`, `.md`, `.pdf`, `.docx` files.
- 🧠 Multiple transformer models: `facebook/bart-large-cnn`, `distilbart-cnn-12-6`, `pegasus-xsum`, `t5-small`.
- 🪢 Fallback to `sumy` (LexRank) if the transformer pipeline fails.
- ⚙️ Chunking for long documents.
- 💻 Auto hardware detection (GPU if available, else CPU).

## Quickstart
```bash
git clone https://github.com/<YOUR_USERNAME>/AI-News-Summarizer-Dashboard-EN.git
cd AI-News-Summarizer-Dashboard-EN

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Notebook (Demo & API Usage)
Open `notebooks/demo.ipynb` to:
- Test the summarizer pipeline directly (without Streamlit).
- Compare a couple of models.
- Summarize a sample URL or your own pasted text.
- Save results to a file for later use.

## Deploy to Streamlit Cloud
1. Push this repo to GitHub.
2. On Streamlit Community Cloud, create a new app and point it to `app.py` on `main`.
3. Set Python 3.10+ and ensure `requirements.txt` is used.

## Push to GitHub
```bash
git init
git add .
git commit -m "feat: AI News Summarizer Dashboard (English) + demo notebook"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/AI-News-Summarizer-Dashboard-EN.git
git push -u origin main
```

## License
MIT © 2025