#!/usr/bin/env python3
"""
fetch_esg_news_gdelt.py  (optimized + correct GDELT field filters)
Query the GDELT Doc API for ESG-related news by company keywords.

Outputs:
- data/esg_events/esg_events_<YYYY-MM-DD>.csv   # Daily snapshot (may be empty)
- data/latest/esg_events_latest.csv             # Always the latest file

Columns:
  ['published_utc','title','url','source','lang','company','ticker','event_type','confidence']

Env:
- GDELT_TIMESPAN=7d|30d|...     (default 7d)
- GDELT_MAX_RECORDS=250
- GDELT_RETRY=3
- GDELT_BACKOFF=1.5
- MAX_QUERY_LEN=200
- LANG_FILTER=english           # use 'english' (lowercase); empty to disable
- SRC_FILTER=US                 # ISO country code; empty to disable
- USE_MOCK=1                    # demo mode if API returns nothing
"""
from __future__ import annotations

import os, time, datetime as dt, urllib.parse
from typing import List, Optional
import requests, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ESG_DIR = os.path.join(DATA_DIR, "esg_events")
LATEST_DIR = os.path.join(DATA_DIR, "latest")
os.makedirs(ESG_DIR, exist_ok=True)
os.makedirs(LATEST_DIR, exist_ok=True)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMESPAN = os.getenv("GDELT_TIMESPAN", "7d")
MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "250"))
RETRY = int(os.getenv("GDELT_RETRY", "3"))
BACKOFF_SEC = float(os.getenv("GDELT_BACKOFF", "1.5"))
MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "200"))
USE_MOCK = os.getenv("USE_MOCK", "0") == "1"

# ✅ Correct GDELT field names:
#   - sourcelang:<lang>   (e.g., sourcelang:english)
#   - sourcecountry:<CC>  (e.g., sourcecountry:US)
LANG_FILTER = os.getenv("LANG_FILTER", "english").strip()  # "" to disable
SRC_FILTER = os.getenv("SRC_FILTER", "").strip().upper()   # "" to disable

HEADERS = {"User-Agent": "Mozilla/5.0 (ESG-Event-Study/1.2; +https://github.com)"}

POSITIVE_TERMS = ["ESG", "sustainability", "net zero", "renewable", "green bond", "recycled", "solar", "wind"]
NEGATIVE_TERMS = ["pollution", "oil spill", "labor strike", "lawsuit", "ESG downgrade", "greenwashing", "violation", "toxic waste"]

def load_company_map(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def classify_event(title: str) -> str:
    t = (title or "").lower()
    pos = any(term.lower() in t for term in POSITIVE_TERMS)
    neg = any(term.lower() in t for term in NEGATIVE_TERMS)
    if pos and not neg: return "positive"
    if neg and not pos: return "negative"
    if pos and neg: return "mixed"
    return "unknown"

def _quote_if_space(s: str) -> str:
    return f"\"{s}\"" if " " in s else s

def build_keyword_clauses(terms: List[str]) -> List[str]:
    return [_quote_if_space(t) for t in terms]

def _append_filters(q: str) -> str:
    parts = [q]
    if LANG_FILTER:
        parts.append(f"sourcelang:{LANG_FILTER.lower()}")   # ✅ correct field + lowercase value
    if SRC_FILTER:
        parts.append(f"sourcecountry:{SRC_FILTER}")          # ✅ correct field + uppercase country
    return " AND ".join(parts)

def chunk_queries(company: str, terms: List[str], max_len: int) -> List[str]:
    comp = _quote_if_space(company)
    chunks, cur = [], []
    baseline_len = len(f"{comp} AND ")
    cur_len = baseline_len

    def flush():
        if not cur: return
        if len(cur) == 1:
            q = f"{comp} AND {cur[0]}"
        else:
            q = f"{comp} AND ({' OR '.join(cur)})"
        chunks.append(_append_filters(q))

    for term in terms:
        add_len = (3 if cur else 0) + len(term)
        if cur and (cur_len + add_len) > max_len:
            flush()
            cur, cur_len = [term], baseline_len + len(term)
        else:
            cur.append(term); cur_len += add_len
    flush()
    return chunks

def query_gdelt(query: str, timespan: str, max_records: int) -> pd.DataFrame:
    params = {"query": query, "mode": "ArtList", "format": "json",
              "timespan": timespan, "maxrecords": str(max_records), "sort": "datedesc"}
    url = GDELT_DOC_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            try:
                js = r.json()
            except Exception:
                snippet = r.text[:300].replace("\n", " ")
                print(f"[WARN] Non-JSON response (attempt {attempt}): {snippet}")
                raise
            arts = js.get("articles", [])
            rows = [{"published_utc": a.get("seendate"),
                     "title": a.get("title"),
                     "url": a.get("url"),
                     "source": a.get("sourceCountry"),
                     "lang": a.get("language")} for a in arts]
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[WARN] GDELT query attempt {attempt} failed: {e}")
            if attempt < RETRY:
                time.sleep(BACKOFF_SEC * attempt)
    return pd.DataFrame(columns=["published_utc","title","url","source","lang"])

def save_outputs(df: pd.DataFrame):
    today_str = dt.date.today().isoformat()
    hist_path = os.path.join(ESG_DIR, f"esg_events_{today_str}.csv")
    latest_path = os.path.join(LATEST_DIR, "esg_events_latest.csv")
    df.to_csv(hist_path, index=False, encoding="utf-8")
    df.to_csv(latest_path, index=False, encoding="utf-8")
    print(f"Saved ESG events → {hist_path} (rows={len(df)})")
    print(f"Updated latest   → {latest_path}")

def maybe_mock(company_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if os.getenv("USE_MOCK", "0") != "1":
        return None
    today = dt.date.today()
    sample = []
    for _, row in company_df.head(3).iterrows():
        company, ticker = row["company"], row["ticker"]
        sample.append({"published_utc": today.isoformat(),
                       "title": f"{company} announces new sustainability initiative",
                       "url": f"https://example.com/demo-esg-{ticker}",
                       "source": "US", "lang": "english",
                       "company": company, "ticker": ticker,
                       "event_type": "positive", "confidence": 0.9})
    return pd.DataFrame(sample)

def main():
    map_path = os.path.join(ROOT, "company_ticker_map.csv")
    company_df = load_company_map(map_path)

    term_clauses = build_keyword_clauses(POSITIVE_TERMS + NEGATIVE_TERMS)

    all_frames: List[pd.DataFrame] = []
    for _, row in company_df.iterrows():
        company, ticker = str(row["company"]), str(row["ticker"])
        queries = chunk_queries(company, term_clauses, MAX_QUERY_LEN)

        per_co: List[pd.DataFrame] = []
        for q in queries:
            df = query_gdelt(q, timespan=GDELT_TIMESPAN, max_records=MAX_RECORDS)
            if df.empty: continue
            df["company"] = company
            df["ticker"] = ticker
            df["event_type"] = df["title"].apply(classify_event)
            df["confidence"] = df["title"].apply(lambda s: 0.9 if isinstance(s, str) and len(s) > 20 else 0.5)
            per_co.append(df); time.sleep(0.30)
        if per_co:
            merged = pd.concat(per_co, ignore_index=True).drop_duplicates(subset=["url"])
            all_frames.append(merged)

    if all_frames:
        out = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["url"])
        out = out[["published_utc","title","url","source","lang","company","ticker","event_type","confidence"]]
        save_outputs(out)
    else:
        print("[INFO] No ESG articles found from GDELT within the timespan.")
        mock = maybe_mock(company_df)
        if mock is not None and len(mock) > 0:
            print("[INFO] Using MOCK events for demonstration (set USE_MOCK=0 to disable).")
            save_outputs(mock)
        else:
            cols = ["published_utc","title","url","source","lang","company","ticker","event_type","confidence"]
            save_outputs(pd.DataFrame(columns=cols))

if __name__ == "__main__":
    main()
