# 📰 AI News Summarizer Dashboard

An interactive **AI-powered news summarizer** built with **Streamlit**, capable of summarizing content from **URLs**, **pasted text**, or **uploaded documents**.  
Uses **Hugging Face Transformers** for abstractive summarization with an **extractive LexRank fallback**.

👉 **Live Demo:** *(deploy to Streamlit Cloud or run locally)*  
📓 **Notebook:** `notebooks/demo.ipynb` (for experimentation and offline use)

---

## 🔹 Features

- 🔗 Extract full text from any article **URL** using `trafilatura`
- 📝 Paste text directly and get a concise summary instantly
- 📁 Upload `.txt`, `.md`, `.pdf`, or `.docx` files for summarization
- 🧠 Multiple summarization models available (`bart-large-cnn`, `distilbart`, `pegasus-xsum`, `t5-small`)
- 🧩 Automatic chunking for long documents
- 🪢 **LexRank fallback** if the transformer model fails
- 💻 Auto-detects hardware (GPU if available, else CPU)
- 🧠 Includes **Jupyter notebook** demo for testing and customization

---

## 🔹 Project Structure

```
AI-News-Summarizer-Dashboard/
├── app.py              # Streamlit app (entry point)
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
├── .gitignore
└── notebooks/
    └── demo.ipynb      # Jupyter demo notebook (Hugging Face + LexRank)
```

> Long inputs are automatically split into chunks for better summarization results.

---

## 🔹 Run Locally (Windows/Mac/Linux)

1️⃣ **Clone the repo**
```bash
git clone https://github.com/<your-username>/AI-News-Summarizer-Dashboard.git
cd AI-News-Summarizer-Dashboard
```

2️⃣ **Create a virtual environment and install dependencies**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

3️⃣ **Run Streamlit app**
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔹 Notebook Usage

The demo notebook allows you to test summarization pipelines without Streamlit:

```bash
jupyter notebook notebooks/demo.ipynb
```

You can summarize text, test different models, and compare Hugging Face vs LexRank summaries.

---

## 🔹 Requirements

```
streamlit>=1.32.0
transformers>=4.41.0
torch>=2.2.0
sentencepiece>=0.2.0
trafilatura>=1.7.0
sumy>=0.11.0
pdfminer.six>=20221105
python-docx>=1.1.0
jupyter>=1.0.0
```

---

## 🔹 Deployment (Streamlit Community Cloud)

1️⃣ Push this project to GitHub.  
2️⃣ On [Streamlit Cloud](https://streamlit.io/cloud), click **“New app”**.  
3️⃣ Set:
- Repository: `<your-username>/AI-News-Summarizer-Dashboard`
- Branch: `main`
- Main file: `app.py`  
4️⃣ Click **Deploy**.

---

## 🔹 Screenshot

> (Add your screenshot here)

![Dashboard Screenshot](assets/screenshot.png)

---

## 🔹 Author

**Joseph Wang (Mt. SAC)** — Computer Science Transfer Applicant (Fall 2026)  
GitHub: `@<your-username>`  

---

## 🔹 License

MIT © 2025
