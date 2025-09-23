# House Price Prediction with Machine Learning

This notebook demonstrates a regression pipeline for predicting house prices using tabular data.  
The goal is to compare baseline linear and tree-based models while applying standard preprocessing, validation, and evaluation metrics.

---

## 🔹 Pipeline Overview
1. **Load Data**  
   - Preferred: custom dataset `data/house_prices.csv` with target column `price`  
   - Fallback: scikit-learn **California Housing dataset**  

2. **Exploratory Data Analysis (EDA)**  
   - Preview rows, column summary, missing value check  

3. **Dataset Split**  
   - Train/validation split  

4. **Preprocessing**  
   - Median imputation for missing values  
   - Standard scaling for numeric features  

5. **Model Training**  
   - Linear Regression  
   - Random Forest Regressor  

6. **Evaluation**  
   - Mean Absolute Error (MAE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score  
   - Predicted vs Actual plot  
   - Feature Importances (Random Forest only)  

7. **Model Persistence**  
   - Save the best-performing model as `model.pkl`  

---

## 📊 Dataset
- **Option A (Preferred):** Custom dataset `data/house_prices.csv`  
- **Option B (Fallback):** California Housing dataset (auto-loaded via scikit-learn)  

Both options include numeric features (e.g., lot area, rooms, age, location) and a target column `price`.

---

## 📈 Results
- **Random Forest** outperformed Linear Regression on validation set.  
- Achieved higher **R²** and lower **RMSE**.  

Example plots generated:  
- `plots/pred_vs_actual.png`
- <img width="552" height="542" alt="Screenshot 2025-09-23 164450" src="https://github.com/user-attachments/assets/1f8d3080-30d9-4d7d-8d6a-707b750a8cf3" />

- `plots/feature_importance.png` (Random Forest only)
- <img width="798" height="470" alt="Screenshot 2025-09-23 164504" src="https://github.com/user-attachments/assets/75a57d9e-ca93-4507-b215-d65cf226af6e" />


---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/house-price-ml.git
   cd house-price-ml
