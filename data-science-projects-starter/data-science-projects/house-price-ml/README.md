# 🏡 House Price Prediction — Streamlit App

[Live Demo on Streamlit Cloud](https://data-science-projects-h8cgwyb9xydhtxtip9mueh.streamlit.app/)

An interactive machine learning app for **house price prediction**.  
Built with **Streamlit + scikit-learn**, this app lets you upload data, train regression models, visualize metrics, and export trained pipelines.

---

## ✨ Features

- **Auto target suggestion**  
  Detects the most likely target column (e.g., `price`) and warns if you pick a non-continuous variable.

- **Robust preprocessing**  
  - Numeric: median imputation + scaling  
  - Categorical: most frequent imputation + one-hot encoding (with options for min frequency / max categories)

- **Model options**  
  - Linear Regression  
  - Random Forest Regressor  
  - XGBoost Regressor (if available)

- **Evaluation**  
  - MAE, RMSE, R², MAPE  
  - Hold-out validation and optional 5-fold CV  
  - Diagnostic plots: Predicted vs Actual, Residuals Histogram  
  - Permutation Importance (Top 20 features)

- **Log-transform support**  
  Automatically applies `log1p/exp1m` transform to skewed targets like housing prices.

- **Export & Inference**  
  - Download trained pipeline as `.pkl`  
  - Upload features-only CSV for prediction  
  - Manual single-row input form for testing

---

## 🚀 Quickstart (Local)

```bash
# Clone repo
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run app locally
streamlit run app_v3.py
```

---

## 📂 Project Structure

```
.
├── app_v3.py              # Streamlit app (latest version)
├── sample_houses.csv      # Small synthetic dataset
├── sample_houses_large.csv# Larger synthetic dataset (2000 rows)
├── requirements.txt
└── README.md
```

---

## 🧪 Usage

1. **Upload CSV** (or use included sample dataset).  
2. **Select target column** (app auto-suggests `price`).  
3. **Configure training** (test size, CV, log transform, encoding options).  
4. **Choose model** (Linear Regression, Random Forest, XGBoost).  
5. **Train model** → view metrics, plots, and feature importances.  
6. **Download pipeline** (`trained_pipeline.pkl`).  
7. **Upload features-only CSV** or use manual input for predictions.

---

## 📌 Notes

- If your dataset is very small (<80 rows), the app automatically switches to **5-fold cross-validation** for metrics.  
- Log-transform is disabled if target contains nonpositive values.  
- Feature importances are shown via **permutation importance** for model-agnostic explanation.

---

## 🔹 Author
**Joseph Wang (Mt. SAC)** — CS transfer applicant (Fall 2026)

