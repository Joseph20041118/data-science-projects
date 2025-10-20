# 📰 AI News Summarizer Dashboard

一個可以「從網址擷取 + 直接貼文 + 上傳檔案」的**AI 摘要儀表板**（Streamlit）。
同時支援 **HuggingFace 變壓器模型（abstractive）** 與 **LexRank（extractive 備援）**。

English version below ⤵️

---

## 功能 Features
- 🔗 **從網址擷取**：使用 `trafilatura` 抓取網頁正文。
- 📝 **直接貼上文字**：將段落、筆記或文章貼上即可摘要。
- 📁 **上傳檔案**：支援 `.txt`、`.md`、`.pdf`、`.docx`。
- 🧠 **多種摘要模型**：`facebook/bart-large-cnn`、`distilbart-cnn`、`pegasus-xsum`、`t5-small`…
- 🪢 **備援機制**：若 Transformer 摘要失敗，自動改用 `sumy` LexRank。
- ⚙️ **Chunking**：長文會自動分段摘要再合併。
- 💻 **自動硬體偵測**：若有 GPU 會自動使用，否則跑 CPU。

## 安裝與執行
```bash
# 1) 下載專案
git clone https://github.com/<YOUR_USERNAME>/AI-News-Summarizer-Dashboard.git
cd AI-News-Summarizer-Dashboard

# 2) 建立虛擬環境（可選）
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3) 安裝套件
pip install -r requirements.txt

# 4) 啟動
streamlit run app.py
```

## 使用方式
1. 左側選擇**摘要模型**與**目標字數**。
2. 選擇其中一個頁籤：
   - **🔗 從網址擷取**：貼上文章網址 → 按「擷取並摘要」。
   - **📝 直接貼上文字**：貼文 → 按「產生摘要」。
   - **📁 上傳檔案**：上傳 `.txt/.md/.pdf/.docx` → 按「讀取並摘要」。

## 常見問題
- **模型下載很慢或失敗**：請改用較小模型（如 `t5-small`），或本地先 `huggingface-cli` 下載。
- **網站擷取不到文字**：改用貼上文字或上傳檔案；不同網站結構差異大。
- **PDF/DOCX 解析怪怪的**：可先轉成純文字 `.txt` 再上傳。

## 部署到 Streamlit Cloud
1. 將此 repo 推上 GitHub（見下方指令）。
2. 連結到 Streamlit Cloud，指定 `app.py` 為入口。
3. 設定 Python 版本（可用 3.10+）與 `requirements.txt`。

## 推到 GitHub
```bash
git init
git add .
git commit -m "feat: AI News Summarizer Dashboard (Streamlit)"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/AI-News-Summarizer-Dashboard.git
git push -u origin main
```

---

# English

A **Streamlit** dashboard that summarizes content from **URLs**, **pasted text**, or **uploaded files**.
Supports **HuggingFace transformers (abstractive)** with an **extractive LexRank fallback**.

## Features
- 🔗 Extract main article text from a URL using `trafilatura`.
- 📝 Paste arbitrary text and summarize it.
- 📁 Upload `.txt`, `.md`, `.pdf`, `.docx` files.
- 🧠 Multiple transformer models: `facebook/bart-large-cnn`, `distilbart-cnn`, `pegasus-xsum`, `t5-small`, etc.
- 🪢 Fallback to `sumy` (LexRank) if the transformer pipeline fails.
- ⚙️ Chunking for long documents.
- 💻 Auto hardware detection (GPU if available, else CPU).

## Setup & Run
```bash
git clone https://github.com/<YOUR_USERNAME>/AI-News-Summarizer-Dashboard.git
cd AI-News-Summarizer-Dashboard
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
- Push to GitHub and connect the repo on Streamlit Cloud.
- Set the entrypoint to `app.py` and include `requirements.txt`.

---

MIT © 2025
