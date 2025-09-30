#!/usr/bin/env python3
"""
fetch_prices.py
Fetch daily OHLCV data and compute returns for a list of tickers using yfinance.
Outputs:
- data/prices/prices_<YYYY-MM-DD>.csv (wide Adjusted Close)
- data/returns/returns_<YYYY-MM-DD>.csv (daily returns)
"""
import os, sys, datetime as dt
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
RETURNS_DIR = os.path.join(DATA_DIR, "returns")
os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(RETURNS_DIR, exist_ok=True)

def load_tickers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers

def main():
    tickers_path = os.path.join(ROOT, "tickers.txt")
    tickers = load_tickers(tickers_path)
    end = dt.date.today()
    start = end - dt.timedelta(days=2*365 + 30)

    df = yf.download(tickers, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(), auto_adjust=True)["Close"]
    # yf returns a Series for single ticker, normalize to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    today_tag = end.isoformat()
    prices_out = os.path.join(PRICES_DIR, f"prices_{today_tag}.csv")
    df.to_csv(prices_out, index=True)

    rets = df.pct_change().dropna(how="all")
    returns_out = os.path.join(RETURNS_DIR, f"returns_{today_tag}.csv")
    rets.to_csv(returns_out, index=True)

    print(f"Saved prices to {prices_out}")
    print(f"Saved returns to {returns_out}")

if __name__ == "__main__":
    main()
