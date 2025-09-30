#!/usr/bin/env python3
"""
make_dataset.py
Join latest prices/returns with latest ESG events to produce analysis-ready tables.
Outputs:
- data/latest/esg_events_latest.csv
- data/latest/prices_latest.csv
- data/latest/returns_latest.csv
- data/latest/events_prices_panel.csv  (event window index per ticker-event)
"""
import os, glob, datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
LATEST_DIR = os.path.join(DATA_DIR, "latest")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
RETURNS_DIR = os.path.join(DATA_DIR, "returns")
ESG_DIR = os.path.join(DATA_DIR, "esg_events")

os.makedirs(LATEST_DIR, exist_ok=True)

def latest_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]

def load_latest_prices_returns():
    pfile = latest_file(os.path.join(PRICES_DIR, "prices_*.csv"))
    rfile = latest_file(os.path.join(RETURNS_DIR, "returns_*.csv"))
    if pfile is None or rfile is None:
        raise FileNotFoundError("Missing prices_*.csv or returns_*.csv")
    prices = pd.read_csv(pfile, parse_dates=[0], index_col=0)
    rets = pd.read_csv(rfile, parse_dates=[0], index_col=0)
    return prices, rets

def load_latest_esg():
    efile = latest_file(os.path.join(ESG_DIR, "esg_events_*.csv"))
    if efile is None:
        # create empty structure
        cols = ["published_utc","title","url","source","lang","company","ticker","event_type","confidence"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(efile)
    # normalize date
    df["event_date"] = pd.to_datetime(df["published_utc"]).dt.date
    return df

def build_event_prices_panel(prices: pd.DataFrame, events: pd.DataFrame, window=(-5, 5)):
    """
    For each event (ticker, event_date), collect price rows within [t-5, t+5],
    and create relative day index 'tau'.
    """
    rows = []
    # Ensure price index is date
    p = prices.copy()
    p.index = pd.to_datetime(p.index).date

    for _, ev in events.iterrows():
        tic = ev["ticker"]
        if tic not in p.columns:
            continue
        t0 = ev["event_date"]
        start = t0 + dt.timedelta(days=window[0])
        end = t0 + dt.timedelta(days=window[1])
        seg = p.loc[(p.index >= start) & (p.index <= end), [tic]].copy()
        if seg.empty:
            continue
        seg["date"] = seg.index
        seg["ticker"] = tic
        seg["event_date"] = t0
        seg["tau"] = (pd.to_datetime(seg["date"]) - pd.to_datetime(t0)).dt.days
        rows.append(seg.rename(columns={tic: "price"}))

    if not rows:
        return pd.DataFrame(columns=["date","ticker","event_date","tau","price"])
    out = pd.concat(rows, ignore_index=True)
    return out[["date","ticker","event_date","tau","price"]]

def main():
    prices, rets = load_latest_prices_returns()
    events = load_latest_esg()

    # Save latest snapshots
    prices_out = os.path.join(LATEST_DIR, "prices_latest.csv")
    returns_out = os.path.join(LATEST_DIR, "returns_latest.csv")
    prices.to_csv(prices_out)
    rets.to_csv(returns_out)

    events_out = os.path.join(LATEST_DIR, "esg_events_latest.csv")
    events.to_csv(events_out, index=False)

    panel = build_event_prices_panel(prices, events, window=(-5, 5))
    panel_out = os.path.join(LATEST_DIR, "events_prices_panel.csv")
    panel.to_csv(panel_out, index=False)

    print(f"Saved latest prices → {prices_out}")
    print(f"Saved latest returns → {returns_out}")
    print(f"Saved latest ESG events → {events_out}")
    print(f"Saved panel (event windows) → {panel_out} rows={len(panel)}")

if __name__ == "__main__":
    main()
