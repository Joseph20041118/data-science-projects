#!/usr/bin/env python3
"""
fetch_esg_news_gdelt.py
Query GDELT Doc API for ESG-related news by company keywords in the last 24 hours.
Outputs:
- data/esg_events/esg_events_<YYYY-MM-DD>.csv  (daily snapshot)
- data/latest/esg_events_latest.csv            (always latest)
Columns:
  ['published_utc','title','url','source','lang','company','ticker','event_type','confidence']
Notes:
- This is a lightweight heuristic matcher (keyword search). For research-grade data, consider paid ESG datasets.
"""
import os, sys, datetime as dt, time, urllib.parse, requests, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ESG_DIR = os.path.join(DATA_DIR, "esg_events")
LATEST_DIR = os.path.join(DATA_DIR, "latest")
os.makedirs(ESG_DIR, exist_ok=True)
os.makedirs(LATEST_DIR, exist_ok=True)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Basic ESG keyword sets (extend as needed)
POSITIVE_TERMS = [
    "sustainability", "carbon neutral", "net zero", "renewable", "green bond",
    "ESG initiative", "sustainability report", "recycled", "solar", "wind"
]
NEGATIVE_TERMS = [
    "environmental fine", "pollution", "oil spill", "labor strike", "lawsuit",
    "ESG downgrade", "child labor", "greenwashing", "violation", "toxic waste"
]

def load_company_map(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def classify_event(title: str) -> str:
    t = (title or "").lower()
    pos = any(term in t for term in POSITIVE_TERMS)
    neg = any(term in t for term in NEGATIVE_TERMS)
    if pos and not neg: return "positive"
    if neg and not pos: return "negative"
    if pos and neg: return "mixed"
    return "unknown"

def query_gdelt(query: str, since_minutes: int = 1440, max_records: int = 250) -> pd.DataFrame:
    span = "1d" if since_minutes >= 1440 else f"{since_minutes}min"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "timespan": span,
        "maxrecords": str(max_records),
        "sort": "datedesc"
    }
    url = GDELT_DOC_API + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    arts = js.get("articles", [])
    rows = []
    for a in arts:
        rows.append({
            "published_utc": a.get("seendate"),
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("sourceCountry"),
            "lang": a.get("language")
        })
    return pd.DataFrame(rows)

def main():
    company_map_path = os.path.join(ROOT, "company_ticker_map.csv")
    company_df = load_company_map(company_map_path)

    all_rows = []
    for _, row in company_df.iterrows():
        company = row["company"]
        ticker = row["ticker"]
        terms = "(" + " OR ".join([f'"{t}"' for t in POSITIVE_TERMS + NEGATIVE_TERMS]) + ")"
        q = f'"{company}" AND ({terms})'
        try:
            df = query_gdelt(q, since_minutes=1440, max_records=250)
            if df.empty:
                continue
            df["company"] = company
            df["ticker"] = ticker
            df["event_type"] = df["title"].apply(classify_event)
            df["confidence"] = df["title"].apply(lambda s: 0.9 if isinstance(s, str) and len(s) > 20 else 0.5)
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] query failed for {company}: {e}")

        time.sleep(0.5)  # be polite

    if not all_rows:
        print("No ESG articles found in the last day.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out = out[["published_utc","title","url","source","lang","company","ticker","event_type","confidence"]]

    today_str = dt.date.today().isoformat()
    hist_path = os.path.join(ESG_DIR, f"esg_events_{today_str}.csv")
    latest_path = os.path.join(LATEST_DIR, "esg_events_latest.csv")

    out.to_csv(hist_path, index=False, encoding="utf-8")
    out.to_csv(latest_path, index=False, encoding="utf-8")

    print(f"Saved ESG events → {hist_path} (rows={len(out)})")
    print(f"Updated latest file → {latest_path}")

if __name__ == "__main__":
    main()
