# COVID-19 Interactive Dashboard

This project builds an interactive dashboard to visualize **COVID-19 cases and trends** using Python.  
The goal is to practice **time-series analysis, data cleaning, and interactive visualization** with libraries like **Plotly** and **Streamlit/Dash**.

---

## 🔹 Pipeline Overview
1. **Load Data**  
   - Download daily COVID-19 case data (Johns Hopkins University / Our World in Data)  
   - Fallback: use a local CSV in `data/covid.csv`

2. **Preprocessing**  
   - Parse dates, handle missing values  
   - Aggregate by country / region  

3. **Visualization**  
   - Time-series plots of confirmed cases, deaths, vaccinations  
   - Comparisons across multiple countries  
   - Interactive filters (date range, region, metric)

4. **Dashboard App**  
   - Built with **Streamlit** (or Plotly Dash)  
   - Supports dropdown filters, sliders, and real-time plots  

5. **Export & Deployment**  
   - Save static plots into `plots/` folder  
   - Run app locally: `streamlit run app.py`

---

## 📊 Dataset
- **Preferred source:** [Our World in Data COVID-19 dataset](https://ourworldindata.org/covid-deaths)  
- **Alternative:** Johns Hopkins CSSE COVID-19 Data (time-series CSV)  
- **Fallback:** Custom `data/covid.csv`  

---

## 📈 Example Features
- Line chart of daily new cases  
- Rolling average of cases/deaths  
- Comparison: US vs Taiwan vs Italy  
- Vaccination progress plots  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/covid19-dashboard.git
   cd covid19-dashboard
