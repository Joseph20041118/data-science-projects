
import os
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------
# TZ/Date helpers
# --------------------------
def to_naive_datetime_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce", utc=True)
    idx = idx.tz_convert(None)
    return idx.normalize()

def to_naive_datetime_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.normalize()

def align_to_trading_day(dates: pd.Series, trading_index: pd.DatetimeIndex, method="next") -> pd.Series:
    td = pd.Index(pd.to_datetime(trading_index).normalize().unique()).sort_values()
    normalized = pd.to_datetime(dates).dt.normalize()
    out = []
    for d in normalized:
        idx = td.get_indexer([d], method="backfill" if method=="next" else "ffill")
        if idx[0] == -1:
            out.append(td[0] if method=="next" else td[-1])
        else:
            out.append(td[idx[0]])
    return pd.to_datetime(out)

# --------------------------
# Loaders
# --------------------------
def load_latest_bundle_from_disk():
    CANDIDATES = [
        "data/latest",
        "data-science-projects-starter/data-science-projects/esg-stock-event-study/data/latest",
    ]
    data_latest = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if data_latest is None:
        return None, None, None, None

    prices_path  = os.path.join(data_latest, "prices_latest.csv")
    returns_path = os.path.join(data_latest, "returns_latest.csv")
    events_path  = os.path.join(data_latest, "esg_events_latest.csv")
    panel_path   = os.path.join(data_latest, "events_prices_panel.csv")

    def _read_df(p):
        if not os.path.exists(p):
            return None
        return pd.read_csv(p)

    prices  = _read_df(prices_path)
    returns = _read_df(returns_path)
    events  = _read_df(events_path)
    panel   = _read_df(panel_path)

    return prices, returns, events, panel

def normalize_bundle(prices, returns, events, panel):
    # Index fix
    if prices is not None and prices.shape[1] > 0:
        if prices.columns[0].lower() in {"date", "datetime"}:
            prices = prices.set_index(prices.columns[0])
    if returns is not None and returns.shape[1] > 0:
        if returns.columns[0].lower() in {"date", "datetime"}:
            returns = returns.set_index(returns.columns[0])

    if prices is not None:
        prices.index  = to_naive_datetime_index(prices.index)
        prices = prices.apply(pd.to_numeric, errors="coerce")
    if returns is not None:
        returns.index = to_naive_datetime_index(returns.index)
        returns = returns.apply(pd.to_numeric, errors="coerce")

    # Events
    if events is None:
        events = pd.DataFrame(columns=["ticker","event_date"])
    if "event_date" in events.columns:
        events["event_date"] = to_naive_datetime_series(events["event_date"])
    elif "published_utc" in events.columns:
        events["event_date"] = to_naive_datetime_series(events["published_utc"])
    else:
        events["event_date"] = pd.NaT
    if "ticker" not in events.columns:
        events["ticker"] = ""
    events["ticker"] = events["ticker"].astype(str)

    # Filter to tickers present in both prices and returns
    if (prices is not None) and (returns is not None):
        tickers_in_data = set(prices.columns).intersection(returns.columns)
        if len(tickers_in_data) > 0:
            events = events[events["ticker"].isin(tickers_in_data)].copy()

    events = (events
              .dropna(subset=["event_date"])
              .drop_duplicates(subset=["ticker","event_date"], keep="first"))
    # Panel dates
    if panel is not None:
        if "date" in panel.columns:
            panel["date"] = to_naive_datetime_series(panel["date"])
        if "event_date" in panel.columns:
            panel["event_date"] = to_naive_datetime_series(panel["event_date"])

    return prices, returns, events.reset_index(drop=True), panel

# --------------------------
# Event-study (mean-adjusted & market model)
# --------------------------
def mean_adjusted_expected(ser: pd.Series, event_date: pd.Timestamp, est_win=(-120,-20), min_obs=30):
    start = event_date + pd.Timedelta(days=est_win[0])
    end   = event_date + pd.Timedelta(days=est_win[1])
    s = ser.loc[(ser.index >= start) & (ser.index <= end)]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.size < min_obs:
        return np.nan
    return float(s.mean())

def fit_alpha_beta(stock: pd.Series, market: pd.Series, event_date: pd.Timestamp, est_win=(-120,-20), min_obs=30):
    start = event_date + pd.Timedelta(days=est_win[0])
    end   = event_date + pd.Timedelta(days=est_win[1])
    s = stock.loc[(stock.index >= start) & (stock.index <= end)]
    m = market.loc[(market.index >= start) & (market.index <= end)]
    df = pd.DataFrame({"s": s, "m": m}).dropna()
    if len(df) < min_obs:
        return np.nan, np.nan
    X = np.vstack([np.ones(len(df)), df["m"].values]).T
    y = df["s"].values
    alpha, beta = (np.linalg.pinv(X) @ y).tolist()
    return float(alpha), float(beta)

def build_event_ar_table(returns_df: pd.DataFrame, events_df: pd.DataFrame,
                         evt_win=(-5,5), align_method="next",
                         model="mean", market_col=None, est_win=(-120,-20), min_obs=30):
    rows = []
    r = returns_df.copy()
    r.index = to_naive_datetime_index(r.index)

    ev = events_df.copy()
    ev["event_date"] = to_naive_datetime_series(ev["event_date"])
    ev = ev.dropna(subset=["event_date"])
    if "ticker" not in ev.columns:
        ev["ticker"] = ""
    ev["ticker"] = ev["ticker"].astype(str)
    ev = ev.drop_duplicates(subset=["ticker","event_date"]).reset_index(drop=True)

    # Align to trading day
    ev["event_date"] = align_to_trading_day(ev["event_date"], r.index, method=align_method)

    # Market series if needed
    if (model == "market") and (market_col is not None) and (market_col not in r.columns):
        market_col = None

    for _, e in ev.iterrows():
        tic = e["ticker"]
        if tic not in r.columns:
            continue
        event_date = e["event_date"]

        start = event_date + pd.Timedelta(days=evt_win[0])
        end   = event_date + pd.Timedelta(days=evt_win[1])

        if model == "mean":
            mu = mean_adjusted_expected(r[tic], event_date, est_win=est_win, min_obs=min_obs)
            if np.isnan(mu):
                continue
            seg = r.loc[(r.index >= start) & (r.index <= end), [tic]].copy().dropna()
            if seg.empty:
                continue
            seg = seg.rename(columns={tic:"ret"})
            seg["exp_ret"] = mu
        else:
            if market_col is None:
                continue
            alpha, beta = fit_alpha_beta(r[tic], r[market_col], event_date, est_win=est_win, min_obs=min_obs)
            if np.isnan(alpha) or np.isnan(beta):
                continue
            seg = r.loc[(r.index >= start) & (r.index <= end), [tic, market_col]].copy().dropna()
            if seg.empty:
                continue
            seg = seg.rename(columns={tic:"ret", market_col:"mkt"})
            seg["exp_ret"] = alpha + beta * seg["mkt"]
            seg["alpha"] = alpha
            seg["beta"] = beta

        seg["date"] = seg.index
        seg["tau"] = (seg["date"] - event_date).dt.days
        seg["ticker"] = tic
        seg["event_date"] = event_date
        seg["ar"] = seg["ret"] - seg["exp_ret"]
        keep_cols = ["date","ticker","event_date","tau","ret","exp_ret","ar"]
        if "mkt" in seg.columns: keep_cols.insert(5, "mkt")
        if "alpha" in seg.columns: keep_cols += ["alpha","beta"]
        rows.append(seg[keep_cols])

    if not rows:
        return pd.DataFrame(columns=["date","ticker","event_date","tau","ret","exp_ret","ar"])
    return pd.concat(rows, ignore_index=True).sort_values(["ticker","event_date","tau"]).reset_index(drop=True)

def aggregate_aar_car_with_t(event_ar_df: pd.DataFrame):
    if event_ar_df.empty:
        return pd.DataFrame(columns=["tau","AAR","N","sd","t"]), pd.DataFrame(columns=["tau","CAR"])
    stats = (
        event_ar_df.groupby("tau")["ar"]
        .agg(AAR="mean", sd=lambda x: x.std(ddof=1), N="count")
        .reset_index()
        .sort_values("tau")
    )
    denom = stats["sd"] / np.sqrt(stats["N"].where(stats["N"]>0, np.nan))
    stats["t"] = stats["AAR"] / denom.replace(0, np.nan)
    stats.replace([np.inf, -np.inf], np.nan, inplace=True)

    car = stats[["tau","AAR"]].copy()
    car["CAR"] = car["AAR"].cumsum()
    return stats[["tau","AAR","N","sd","t"]], car[["tau","CAR"]]

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="ESG Event Study", layout="wide")
st.title("ESG Event Study — Mean-Adjusted / Market Model")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Load data from:", ["Disk (data/latest)", "Upload files"])

    # Parameters
    st.header("Parameters")
    model = st.selectbox("Model", ["mean", "market"], index=0, help="mean = mean-adjusted, market = market model")
    est_a = st.number_input("Estimation window start (days before)", value=-120)
    est_b = st.number_input("Estimation window end (days before)", value=-20)
    evt_a = st.number_input("Event window start (days)", value=-5)
    evt_b = st.number_input("Event window end (days)", value=5)
    min_obs = st.number_input("Minimum estimation observations", value=30, min_value=5, step=1)
    align_method = st.selectbox("Align to trading day", ["next", "prev"], index=0)

    market_candidates = ["^GSPC", "SPY", "VOO", "IVV"]
    picked_market = st.selectbox("Market series (for market model)", market_candidates, index=1)

# Load data
if mode == "Disk (data/latest)":
    prices, returns, events, panel = load_latest_bundle_from_disk()
else:
    st.subheader("Upload CSVs")
    up_prices  = st.file_uploader("prices_latest.csv",  type=["csv"], key="prices")
    up_returns = st.file_uploader("returns_latest.csv", type=["csv"], key="returns")
    up_events  = st.file_uploader("esg_events_latest.csv", type=["csv"], key="events")
    up_panel   = st.file_uploader("events_prices_panel.csv (optional)", type=["csv"], key="panel")

    def _maybe_read(f):
        if f is None: return None
        return pd.read_csv(io.BytesIO(f.read()))
    prices  = _maybe_read(up_prices)
    returns = _maybe_read(up_returns)
    events  = _maybe_read(up_events)
    panel   = _maybe_read(up_panel)

prices, returns, events, panel = normalize_bundle(prices, returns, events, panel)

# Basic info
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Prices**", prices.shape if prices is not None else None)
with c2:
    st.write("**Returns**", returns.shape if returns is not None else None)
with c3:
    st.write("**Events**", events.shape if events is not None else None)

if (prices is None) or (returns is None) or (events is None) or returns.empty or events.empty:
    st.warning("Please provide valid prices / returns / events data.")
    st.stop()

# Market series fallback
mkt_col = picked_market if picked_market in returns.columns else None
if (model == "market") and (mkt_col is None):
    valid_cols = [c for c in returns.columns if returns[c].notna().mean() > 0.8]
    mkt_col = "_EW_MARKET_"
    returns[mkt_col] = returns[valid_cols].mean(axis=1)

# Build AR table
event_ar = build_event_ar_table(
    returns_df=returns,
    events_df=events,
    evt_win=(int(evt_a), int(evt_b)),
    align_method=align_method,
    model=model,
    market_col=mkt_col,
    est_win=(int(est_a), int(est_b)),
    min_obs=int(min_obs),
)

st.subheader("Event AR table (head)")
st.dataframe(event_ar.head(20))

# Aggregate
aar, car = aggregate_aar_car_with_t(event_ar)

c1, c2 = st.columns(2)
with c1:
    st.subheader("AAR by tau")
    st.dataframe(aar)
with c2:
    st.subheader("CAR by tau")
    st.dataframe(car)

# Charts
if not aar.empty:
    fig = plt.figure()
    plt.plot(aar["tau"], aar["AAR"], marker="o")
    plt.axvline(0, linestyle="--")
    plt.title("Average Abnormal Return (AAR)")
    plt.xlabel("Tau (days relative to event)")
    plt.ylabel("AAR")
    plt.grid(True)
    st.pyplot(fig)

if not car.empty:
    fig = plt.figure()
    plt.plot(car["tau"], car["CAR"], marker="o")
    plt.axvline(0, linestyle="--")
    plt.title("Cumulative Abnormal Return (CAR)")
    plt.xlabel("Tau (days relative to event)")
    plt.ylabel("CAR")
    plt.grid(True)
    st.pyplot(fig)

if not aar.empty:
    fig = plt.figure()
    plt.stem(aar["tau"], aar["t"])
    plt.axvline(0, linestyle="--")
    plt.title("AAR t-statistics by Tau")
    plt.xlabel("Tau")
    plt.ylabel("t-statistic")
    plt.grid(True)
    st.pyplot(fig)

# Downloads
st.subheader("Download results")
if not event_ar.empty:
    st.download_button("Download event_ar_table.csv", data=event_ar.to_csv(index=False).encode("utf-8"), file_name="event_ar_table.csv")
if not aar.empty:
    st.download_button("Download aar_table.csv", data=aar.to_csv(index=False).encode("utf-8"), file_name="aar_table.csv")
if not car.empty:
    st.download_button("Download car_table.csv", data=car.to_csv(index=False).encode("utf-8"), file_name="car_table.csv")
