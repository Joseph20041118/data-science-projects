# 📘 ESG Event Study — Streamlit App

This project implements an **event study framework** for ESG-related corporate news.  
It integrates data pipelines (prices, returns, ESG events) with a **Streamlit web app** that visualizes average and cumulative abnormal returns (AAR, CAR).

🌐 **Live app**: [Streamlit Cloud Deployment](https://data-science-projects-kuclaejrt2adyr9j6nr8ao.streamlit.app/)
🌐 **Note Book**:

---

## 🔄 Project Workflow

1. **Tickers list**  
   - Defined in [`tickers.txt`](./tickers.txt).  
   - Example (current): `AAPL, MSFT, TSLA, AMZN, GOOGL`.

2. **Fetch prices & returns**  
   - [`fetch_prices.py`](./fetch_prices.py) uses `yfinance` to download daily prices/returns.  
   - Outputs → `data/latest/prices_latest.csv`, `returns_latest.csv`.

3. **Fetch ESG events**  
   - [`fetch_esg_news_gdelt.py`](./fetch_esg_news_gdelt.py) queries the GDELT API for ESG-related news.  
   - Requires [`company_ticker_map.csv`](./company_ticker_map.csv) mapping company names ↔ tickers.  
   - Outputs → `data/latest/esg_events_latest.csv`.

4. **Build dataset**  
   - [`make_dataset.py`](./make_dataset.py) integrates prices, returns, and events.  
   - Produces optional panel file `events_prices_panel.csv`.

5. **Run event study**  
   - Notebooks:
     - [`event_study.ipynb`](./event_study.ipynb) → Mean-adjusted model  
     - [`event_study_market_model.ipynb`](./event_study_market_model.ipynb) → Market model

6. **Streamlit App**  
   - [`app.py`](./app.py) is the user-facing app.  
   - Lets you upload CSVs or auto-load from `data/latest/`.  
   - Outputs AAR, CAR, t-statistics, and CSV downloads.

---

## 🚀 Running Locally

### 1. Clone & install
```bash
git clone <your-repo-url>
cd data-science-projects
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:
```
pandas
numpy
yfinance
requests
python-dateutil
pytz
matplotlib
scikit-learn
streamlit
```

### 2. Generate latest data
```bash
python fetch_prices.py
python fetch_esg_news_gdelt.py
python make_dataset.py
```

### 3. Launch the app
```bash
streamlit run app.py
```
Go to: [http://localhost:8501](http://localhost:8501)

## 🔹 Author
**Joseph Wang (Mt. SAC)** — CS transfer applicant (Fall 2026)
