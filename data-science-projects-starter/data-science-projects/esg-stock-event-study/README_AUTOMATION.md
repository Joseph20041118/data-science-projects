# ESG Daily Dataset Pipeline (Automated)

This folder contains a **daily-updating pipeline** to build two datasets:
- `data/latest/esg_events_latest.csv` — ESG news events (from GDELT) mapped to tickers
- `data/latest/prices_latest.csv`, `data/latest/returns_latest.csv` — latest price snapshots (from Yahoo Finance)
- `data/latest/events_prices_panel.csv` — price panel around events (τ = -5..+5)

## Files
- `tickers.txt` — list of tickers to track
- `company_ticker_map.csv` — company name ⇄ ticker mapping for ESG news matching
- `fetch_prices.py` — downloads prices and returns with `yfinance`
- `fetch_esg_news_gdelt.py` — pulls ESG-related news via **GDELT Doc API**
- `make_dataset.py` — joins and produces the final datasets

## Run locally
```bash
pip install -r requirements.txt
python fetch_prices.py
python fetch_esg_news_gdelt.py
python make_dataset.py
```

## GitHub Actions (daily schedule)
Add this workflow to `.github/workflows/daily_esg_dataset.yml` to run **every day at 09:00 UTC** and push updated CSVs:

```yaml
name: ESG Daily Dataset
on:
  schedule:
    - cron: "0 9 * * *"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Fetch prices
        run: python fetch_prices.py

      - name: Fetch ESG news (GDELT)
        run: python fetch_esg_news_gdelt.py

      - name: Build latest datasets
        run: python make_dataset.py

      - name: Commit & push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/*
          git commit -m "Auto-update ESG datasets $(date -u +'%Y-%m-%d')" || echo "No changes"
          git push
```
