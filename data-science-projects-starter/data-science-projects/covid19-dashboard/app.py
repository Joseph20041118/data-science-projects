# --- Robust dataset loader (works locally & on Streamlit Cloud) ---
import os, glob, io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests  # for fallback download

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

CANDIDATES = [
    "owid_clean.csv",
    "owid_sample.csv",
    os.path.join("data", "owid_clean.csv"),
    os.path.join("data", "owid_sample.csv"),
]

@st.cache_data
def load_dataset():
    # 1) Try local files (root or data/)
    for path in CANDIDATES:
        if os.path.exists(path):
            st.info(f"Loaded local dataset: `{path}`")
            return pd.read_csv(path, parse_dates=["date"])

    # 2) Last-resort: download small sample from OWID
    st.warning("Local dataset not found. Downloading a small sample from OWID (first run may take a few seconds).")
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    full = pd.read_csv(io.StringIO(r.text), parse_dates=["date"])

    # Shrink to a safe sample: 6 countries, years 2020–2021
    keep = ["United States","Taiwan","Italy","Japan","India","Brazil"]
    sample = full[full["location"].isin(keep)].copy()
    sample = sample[(sample["date"].dt.year >= 2020) & (sample["date"].dt.year <= 2021)]

    # Keep only relevant columns (smaller memory)
    cols = [
        "iso_code","continent","location","date","population",
        "new_cases","new_deaths","new_vaccinations",
        "total_cases","total_deaths",
        "new_cases_per_million","new_deaths_per_million","new_vaccinations_per_million"
    ]
    sample = sample[[c for c in cols if c in sample.columns]]
    return sample

df = load_dataset()

# Drop aggregates (world/continents)
drop_iso = {"OWID_WRL","OWID_AFR","OWID_ASI","OWID_EUR","OWID_EUN","OWID_INT","OWID_NAM","OWID_OCE","OWID_SAM"}
if "iso_code" in df.columns:
    df = df[~df["iso_code"].isin(drop_iso)].copy()



def ensure_rolling(_df: pd.DataFrame, base_col: str, window: int) -> pd.DataFrame:
    """
    Make sure a rolling-average column exists for `base_col` with window `window`.
    If 'base_col_ra{window}' already exists, reuse; otherwise compute it.
    """
    out = _df.copy()
    roll_name = f"{base_col}_ra{window}"
    if roll_name not in out.columns:
        out = out.sort_values(["location","date"])
        out[roll_name] = out.groupby("location")[base_col] \
                            .transform(lambda s: s.rolling(window, min_periods=1).mean())
    return out, roll_name

def filter_after_outbreak(frame: pd.DataFrame, cum_col="total_cases") -> pd.DataFrame:
    mask = (
        frame[cum_col].fillna(0).gt(0)
        .groupby(frame["location"]).transform(lambda s: s.cummax())
    )
    return frame[mask].copy().sort_values(["location","date"])

# ----------------
# Sidebar controls
# ----------------
st.sidebar.header("Filters")

# Country search + multiselect
all_countries = sorted(df["location"].unique().tolist())
search = st.sidebar.text_input("Search country/region")
options = [c for c in all_countries if search.lower() in c.lower()] if search else all_countries

default_set = [c for c in ["United States","Taiwan","Italy","Japan"] if c in options]
selected = st.sidebar.multiselect("Select countries/regions", options, default=default_set or options[:4])

# Metric group + per-million toggle
metric_group = st.sidebar.selectbox("Metric group", ["Cases", "Deaths", "Vaccinations"])
per_million = st.sidebar.checkbox("Per million", value=True)

base_map = {
    ("Cases", True): "new_cases_per_million",
    ("Cases", False): "new_cases",
    ("Deaths", True): "new_deaths_per_million",
    ("Deaths", False): "new_deaths",
    ("Vaccinations", True): "new_vaccinations_per_million",
    ("Vaccinations", False): "new_vaccinations",
}
base_col = base_map[(metric_group, per_million)]

# Rolling window & date range
window = int(st.sidebar.number_input("Rolling window (days)", min_value=3, max_value=28, value=7, step=1))
min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# After-outbreak toggle
after_outbreak_only = st.sidebar.checkbox("Show data only after first case", value=True)

# -------------
# Data slicing
# -------------
if not selected:
    st.warning("Please select at least one country/region.")
    st.stop()

view = df[df["location"].isin(selected)].copy()
view = view[view["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]

# Make sure rolling column exists (compute if needed)
view, roll_col = ensure_rolling(view, base_col, window)

if after_outbreak_only:
    view = filter_after_outbreak(view, "total_cases")

# ----------
# Headings
# ----------
st.title("COVID-19 Interactive Dashboard")
st.caption("Data source: Our World in Data (OWID). Use the sidebar to filter and explore.")

metric_label = {
    "new_cases": "New cases",
    "new_deaths": "New deaths",
    "new_vaccinations": "New vaccinations",
    "new_cases_per_million": "New cases per million",
    "new_deaths_per_million": "New deaths per million",
    "new_vaccinations_per_million": "New vaccinations per million",
}[base_col]
y_label = f"{metric_label} (RA{window})"

# ----------
# KPI Cards
# ----------
st.subheader("Key Stats")
if not view.empty:
    last8 = view.sort_values("date").groupby("location").tail(window+1).copy()
    last8["wow"] = last8.groupby("location")[roll_col].pct_change(window if window >= 7 else 7)
    kpis = last8.groupby("location").tail(1)[["location", roll_col, "wow"]]
    kpis = kpis.sort_values(roll_col, ascending=False).reset_index(drop=True)

    cols_per_row = 4
    for i in range(0, len(kpis), cols_per_row):
        row = kpis.iloc[i:i+cols_per_row]
        cols = st.columns(len(row))
        for slot, (_, r) in zip(cols, row.iterrows()):
            with slot:
                val = r[roll_col]
                wow = r["wow"]
                delta = None if pd.isna(wow) else f"{wow*100:.1f}% WoW"
                st.metric(label=r["location"], value=f"{val:,.2f}", delta=delta)
else:
    st.info("No data in the selected range.")

st.markdown("---")

# ------------
# Line chart
# ------------
st.subheader(f"{y_label} — {', '.join(selected)}")
line_df = view.dropna(subset=[roll_col])
if line_df.empty:
    st.warning("No non-NaN values for the selected metric/time window.")
else:
    fig = px.line(
        line_df, x="date", y=roll_col, color="location",
        labels={roll_col: y_label, "date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Latest snapshot table
# -----------------------
st.subheader("Latest snapshot (selected countries)")
if not view.empty:
    latest = view.sort_values("date").groupby("location").tail(1)
    show_cols = ["location", "date", roll_col, "population", "total_cases", "total_deaths"]
    show_cols = [c for c in show_cols if c in latest.columns]
    st.dataframe(latest[show_cols].sort_values(roll_col, ascending=False), use_container_width=True)

st.markdown("---")

# -------------
# Global map
# -------------
st.subheader(f"Global view — {y_label} (latest per country)")
latest_global = df.sort_values("date").groupby("location").tail(1)
latest_global, map_col = ensure_rolling(latest_global, base_col, window)

map_df = latest_global.dropna(subset=[map_col])
if map_df.empty:
    st.info("No data available for the map.")
else:
    fig_map = px.choropleth(
        map_df,
        locations="iso_code",
        color=map_col,
        hover_name="location",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.caption("Tip: adjust the rolling window, toggle per-million, and filter dates to explore trends.")


