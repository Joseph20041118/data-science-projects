#!/usr/bin/env python3
"""
make_dataset.py
Join the latest prices/returns with the latest (or most recent snapshot) ESG events 
to produce analysis-ready tables.

Outputs:
- data/latest/esg_events_latest.csv        # (auto-created if only snapshots exist)
- data/latest/prices_latest.csv
- data/latest/returns_latest.csv
- data/latest/events_prices_panel.csv      # Event window price panel (tau = [EVENT_WIN_L, EVENT_WIN_R])

Environment variable overrides (optional):
- EST_WIN_L, EST_WIN_R     # Estimation window (not sliced here, kept for downstream use)
- EVENT_WIN_L, EVENT_WIN_R # Event window (default -5, +5)

Notes:
- Prices and returns are mirrored to `data/latest/` for consistency.
- ESG events are loaded preferentially from `data/latest/esg_events_latest.csv`.
  If not found, the newest snapshot from `data/esg_events/` is used and mirrored to `latest`.
- The output `events_prices_panel.csv` contains per-event price windows, ready for
  abnormal return calculations in downstream notebooks.
"""
import os
import glob
import datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
LATEST_DIR = os.path.join(DATA_DIR, "latest")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
RETURNS_DIR = os.path.join(DATA_DIR, "returns")
ESG_DIR = os.path.join(DATA_DIR, "esg_events")

os.makedirs(LATEST_DIR, exist_ok=True)

# ---- Config (event window can be overridden by environment variables) ----
EVENT_WIN_L = int(os.getenv("EVENT_WIN_L", "-5"))
EVENT_WIN_R = int(os.getenv("EVENT_WIN_R", "5"))

def latest_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by filename and return the latest (assuming *_YYYY-MM-DD.csv naming convention)
    files.sort()
    return files[-1]

def load_latest_prices_returns():
    """Load the most recent prices/returns snapshots by filename pattern; mirror them to data/latest/*.csv."""
    pfile = latest_file(os.path.join(PRICES_DIR, "prices_*.csv"))
    rfile = latest_file(os.path.join(RETURNS_DIR, "returns_*.csv"))
    if pfile is None or rfile is None:
        raise FileNotFoundError("Missing prices_*.csv or returns_*.csv under data/prices or data/returns")
    prices = pd.read_csv(pfile, parse_dates=[0], index_col=0)
    returns = pd.read_csv(rfile, parse_dates=[0], index_col=0)

    # Mirror to latest
    prices_out = os.path.join(LATEST_DIR, "prices_latest.csv")
    returns_out = os.path.join(LATEST_DIR, "returns_latest.csv")
    prices.to_csv(prices_out)
    returns.to_csv(returns_out)
    print(f"[OK] Saved latest prices → {prices_out}")
    print(f"[OK] Saved latest returns → {returns_out}")
    return prices, returns

def load_esg_events_prefer_latest():
    """
    Prefer data/latest/esg_events_latest.csv.
    If not found, fallback to the newest data/esg_events/esg_events_YYYY-MM-DD.csv,
    and also copy it to latest for downstream consistency.
    """
    latest_path = os.path.join(LATEST_DIR, "esg_events_latest.csv")
    if os.path.exists(latest_path):
        df = pd.read_csv(latest_path)
        print(f"[OK] Using latest ESG events → {latest_path} (rows={len(df)})")
        return df

    # fallback to newest snapshot
    snap = latest_file(os.path.join(ESG_DIR, "esg_events_*.csv"))
    if snap is None:
        print("[WARN] No ESG events found (neither latest nor snapshot). Returning empty frame.")
        cols = ["published_utc","title","url","source","lang","company","ticker","event_type","confidence"]
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(snap)
    # also write to latest for downstream
    df.to_csv(latest_path, index=False)
    print(f"[OK] Fallback to snapshot → {snap} and mirrored to → {latest_path} (rows={len(df)})")
    return df

def build_event_prices_panel(prices: pd.DataFrame, events: pd.DataFrame, window=(EVENT_WIN_L, EVENT_WIN_R)):
    """
    For each event (ticker, event_date), collect price rows within [t+L, t+R],
    and create a relative day index 'tau'.
    """
    # Normalize index to dates only
    p = prices.copy()
    p.index = pd.to_datetime(p.index).date

    # Ensure events contain 'event_date'
    if "event_date" not in events.columns:
        # If only 'published_utc' is available, convert to event_date
        if "published_utc" in events.columns:
            events = events.copy()
            events["event_date"] = pd.to_datetime(events["published_utc"]).dt.date
        else:
            # If empty, return an empty panel
            return pd.DataFrame(columns=["date","ticker","event_date","tau","price"])

    rows = []
    L, R = window
    for _, ev in events.iterrows():
        tic = ev.get("ticker")
        if not tic or tic not in p.columns:
            continue
        t0 = ev["event_date"]
        try:
            t0 = pd.to_datetime(t0).date()
        except Exception:
            continue

        start = t0 + dt.timedelta(days=L)
        end = t0 + dt.timedelta(days=R)
        seg = p.loc[(p.index >= start) & (p.index <= end), [tic]].copy()
        if seg.empty:
            continue

        seg["date"] = pd.to_datetime(seg.index)
        seg["ticker"] = tic
        seg["event_date"] = pd.to_datetime(t0)
        seg["tau"] = (seg["date"].dt.normalize() - pd.to_datetime(t0)).dt.days
        rows.append(seg.rename(columns={tic: "price"})[["date","ticker","event_date","tau","price"]])

    if not rows:
        print("[WARN] No matching prices found for given ESG events. Panel will be empty.")
        return pd.DataFrame(columns=["date","ticker","event_date","tau","price"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["ticker","event_date","tau"]).reset_index(drop=True)
    return out

def main():
    # 1) Load the most recent prices/returns and mirror them to data/latest
    prices, returns = load_latest_prices_returns()

    # 2) Load ESG events: prefer latest, otherwise use the most recent snapshot and mirror it to latest
    events = load_esg_events_prefer_latest()

    # 3) Build event price panel (uses prices; abnormal returns computed later in notebooks)
    panel = build_event_prices_panel(prices, events, window=(EVENT_WIN_L, EVENT_WIN_R))

    # 4) Save to data/latest
    events_out = os.path.join(LATEST_DIR, "esg_events_latest.csv")
    if not os.path.exists(events_out):
        events.to_csv(events_out, index=False)
    panel_out = os.path.join(LATEST_DIR, "events_prices_panel.csv")
    panel.to_csv(panel_out, index=False)

    print(f"[OK] Saved latest ESG events   → {events_out} (rows={len(events)})")
    print(f"[OK] Saved event price panel  → {panel_out} (rows={len(panel)})")

if __name__ == "__main__":
    main()

