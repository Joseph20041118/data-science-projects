# 🌱 ESG Stock Event Study

Analyze the impact of **ESG-related news** on stock returns using **event study methodology (AAR/CAR)**.  

---

## 🎯 Objective
- Measure whether ESG-related events (e.g., sustainability announcements, green bond issuance, environmental fines) have a **significant impact on stock prices**.  
- Compare **abnormal returns (AR)** and **cumulative abnormal returns (CAR)** around the event window.  

---

## 📂 Dataset
- Stock price data from **Yahoo Finance (yfinance)**.  
- ESG event dates collected from news or reports (mock dataset included for demo).  

---

## 🧰 Methodology
1. **Data Preparation**  
   - Collect stock price data (event firm + market index).  
   - Align data around event dates (event windows, estimation windows).  

2. **Market Model Estimation**  
   - Estimate expected returns using OLS regression:  
     R_{i,t} = α_i + β_i R_{m,t} + ε_{i,t}  

3. **Abnormal Returns (AR)**  
   - Compute AR = Actual Return – Expected Return.  

4. **Average Abnormal Returns (AAR)** and **Cumulative Abnormal Returns (CAR)**  
   - Aggregate across events to test significance.  

5. **Statistical Testing**  
   - t-tests for AAR / CAR significance.  
   - Plot event study graphs.  

---

## 📊 Results (Example)
- ESG-related news tends to generate **short-term abnormal returns**.  
- CAR plots visualize market reactions pre- and post-event.  
- Demo dataset suggests stronger reactions to **negative ESG news**.  

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
cd esg-stock-event-study
jupyter notebook event_study.ipynb
```

---

## 📌 Future Work
- Expand dataset with real ESG event collections.  
- Compare across industries (e.g., energy vs. tech).  
- Add robustness checks (different event windows).  

---

## 👤 Author

**Joseph Wang**  
🎓 Mt. San Antonio College — CS transfer applicant (Fall 2026)  
