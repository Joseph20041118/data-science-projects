
# House Price Prediction — Streamlit App (Sample)

This app lets you upload a tabular housing dataset, choose a numeric target (e.g., `price`), train a regression model, visualize diagnostics, and export a reusable trained pipeline.

## Features
- CSV upload with automatic type detection (numeric vs categorical).
- Choose target column from numeric fields.
- Robust preprocessing via `ColumnTransformer`:
  - Numeric: median imputation + standardization
  - Categorical: most-frequent imputation + One-Hot-Encoding (ignore unknowns)
- Model options: Linear Regression, Random Forest, (optional) XGBoost if installed.
- Metrics: MAE, RMSE, R². Falls back to 5-fold CV metrics for tiny datasets.
- Plots: Predicted vs Actual, Residuals histogram, Feature Importances (if available).
- Export: Download `trained_pipeline.pkl` (preprocessing + model).
- Inference: Upload a features-only CSV to get `prediction` outputs; or use a manual single-row form.

## Quickstart (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
- If `xgboost` fails to install on your platform, you can remove it from `requirements.txt`. The app runs fine without it.

## Sample Data
A synthetic dataset is included: `sample_houses.csv` with columns:
`bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, yr_built, zipcode, price`.

## How to Use
1. Launch the app; if you don't upload a CSV, it auto-loads `sample_houses.csv`.
2. Pick your target (e.g., `price`) in the sidebar.
3. Choose a model and start training.
4. Review metrics and plots; download the `.pkl` pipeline.
5. For inference, upload a new CSV with the same feature columns (without the target).

## Notes
- For very small datasets (< 80 rows), the app reports 5-fold cross-validation metrics and trains on all data to allow exporting a model.
- Feature importance plot appears for tree-based models (Random Forest/XGBoost).
