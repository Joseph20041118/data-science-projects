# COVID-19 Interactive Dashboard
An interactive COVID-19 dashboard built with **Streamlit**, using the **Our World in Data (OWID)** dataset. Explore country-level trends for cases, deaths, and vaccinations with rolling averages and a global map.

**👉 Live Demo:** https://data-science-projects-q6hmdio76khz3ud2fw6u8e.streamlit.app/

## 🔹 Features
- Select multiple countries/regions to compare  
- Switch between **Cases**, **Deaths**, and **Vaccinations**  
- Toggle absolute vs. per-million values  
- Adjustable rolling window (default 7 days)  
- Custom date range filter  
- KPI cards (latest value & WoW change)  
- Global choropleth map (latest per country)  
- Robust data loading (local CSV or fallback to OWID sample)  

## 🔹 Project Structure
```
covid19-dashboard/
├── app.py              # Streamlit app (entry point)
├── requirements.txt    # Dependencies
├── owid_sample.csv     # Small sample (optional, for quick start)
└── README.md
```
> The full `owid_clean.csv` is large (over GitHub’s 100MB limit) and **not included**. The app will auto-use `owid_sample.csv` if present, otherwise it will **download a small OWID slice on first run**.

## 🔹 Run Locally (Windows/Mac/Linux)
1. Clone and enter the project:
```bash
git clone https://github.com/<your-username>/covid19-dashboard.git
cd covid19-dashboard
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Put a dataset next to `app.py`:  
   - `owid_clean.csv` (full, from Kaggle/OWID) **or**  
   - `owid_sample.csv` (small sample provided)  
4. Start the app:
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## 🔹 Data Source
- [Our World in Data — COVID-19 dataset](https://github.com/owid/covid-19-data)

## 🔹 Screenshot
*(Optional: add a screenshot image to your repo, e.g. `docs/screenshot.png`, then uncomment below)*  
<!-- ![Dashboard screenshot](docs/screenshot.png) -->

## 🔹 Author
**Joseph Wang (Mt. SAC)** — CS transfer applicant (Fall 2026)
