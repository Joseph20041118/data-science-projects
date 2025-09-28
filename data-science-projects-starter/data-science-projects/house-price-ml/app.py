
import os
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional: XGBoost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 House Price Prediction — Interactive ML App")
st.caption("Upload your dataset (CSV), choose a target column, train a regression model, and export the trained pipeline.")

# -------------- Utilities --------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if is_numeric_series(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Handle case of all-numeric or all-categorical gracefully
    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    if not transformers:
        # Fallback (should not happen if there is at least one column)
        transformers.append(("passthrough", "passthrough", X.columns.tolist()))
    return ColumnTransformer(transformers)

def pick_model(name: str):
    if name == "LinearRegression":
        return LinearRegression()
    elif name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    elif name == "XGBRegressor" and HAS_XGB:
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist"
        )
    else:
        return LinearRegression()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Histogram")
    st.pyplot(fig)

def try_feature_importances(model, feature_names):
    # Tree models & XGB have feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        topn = min(20, len(importances))
        fig, ax = plt.subplots()
        ax.barh([feature_names[i] for i in order[:topn]][::-1],
                importances[order[:topn]][::-1])
        ax.set_title("Feature Importances (Top 20)")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

# -------------- Sidebar: Data Upload --------------
st.sidebar.header("1) Upload Data")
uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

if uploaded is None:
    st.info("No file uploaded yet. Using the included sample dataset (synthetic).")
    sample_path = os.path.join(os.path.dirname(__file__), "sample_houses.csv")
    df = pd.read_csv(sample_path)
else:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head(20))

if df.empty or df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns (features + target).")
    st.stop()

# -------------- Target Selection --------------
st.sidebar.header("2) Select Target")
numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
if not numeric_cols:
    st.error("No numeric columns found for a regression target.")
    st.stop()

target_col = st.sidebar.selectbox("Target column (continuous numeric):", options=numeric_cols, index=0)

# -------------- Basic EDA --------------
with st.expander("🔎 Quick EDA Summary", expanded=False):
    st.write("#### Shape:", df.shape)
    st.write("#### dtypes")
    st.write(df.dtypes)
    st.write("#### Missing values")
    st.write(df.isna().sum().sort_values(ascending=False))
    st.write("#### Target stats")
    st.write(df[target_col].describe())

# -------------- Train/Test Split Controls --------------
st.sidebar.header("3) Train/Test Split")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

# -------------- Model Selection --------------
st.sidebar.header("4) Model")
model_options = ["LinearRegression", "RandomForestRegressor"]
if HAS_XGB:
    model_options.append("XGBRegressor")
model_name = st.sidebar.selectbox("Choose a model", model_options, index=1)

# -------------- Train Button --------------
train_clicked = st.sidebar.button("🚀 Train Model", type="primary")

if train_clicked:
    with st.spinner("Training..."):
        X, y = split_features_target(df, target_col)
        preprocessor = make_preprocessor(X)
        model = pick_model(model_name)
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        # If dataset is tiny, use K-Fold CV as a safety net for metrics
        too_small = len(df) < 80
        if too_small:
            st.warning("Small dataset detected (< 80 rows). Reporting 5-fold CV scores for robustness.")
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            # Use negative MSE to compute RMSE
            neg_mse = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
            rmse_cv = np.sqrt(-neg_mse)
            mae_cv = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
            r2_cv = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
            st.write(f"**CV RMSE (mean ± sd):** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")
            st.write(f"**CV MAE (mean ± sd):** {mae_cv.mean():.3f} ± {mae_cv.std():.3f}")
            st.write(f"**CV R² (mean ± sd):** {r2_cv.mean():.3f} ± {r2_cv.std():.3f}")

            # Fit on full data to allow exporting and inference
            pipe.fit(X, y)

            # Try to extract feature names after preprocessing for importances
            try:
                feature_names = pipe.named_steps["prep"].get_feature_names_out()
            except Exception:
                feature_names = [f"f{i}" for i in range(pipe.named_steps["prep"].transform(X[:1]).shape[1])]

            # If tree-based, try importances (no y_pred plots for CV-only path)
            try_feature_importances(pipe.named_steps["model"], feature_names)

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae_val = mean_absolute_error(y_test, y_pred)
            rmse_val = rmse(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)

            st.subheader("📊 Validation Metrics")
            st.write(f"**MAE:** {mae_val:.3f}")
            st.write(f"**RMSE:** {rmse_val:.3f}")
            st.write(f"**R²:** {r2_val:.3f}")

            # Plots
            st.subheader("📈 Diagnostic Plots")
            plot_pred_vs_actual(y_test, y_pred)
            plot_residuals(y_test, y_pred)

            # Feature importances if available
            try:
                feature_names = pipe.named_steps["prep"].get_feature_names_out()
            except Exception:
                feature_names = [f"f{i}" for i in range(pipe.named_steps["prep"].transform(X_train[:1]).shape[1])]
            try_feature_importances(pipe.named_steps["model"], feature_names)

        # Save pipeline
        st.subheader("💾 Export Trained Model")
        buffer = io.BytesIO()
        pickle.dump({"pipeline": pipe, "target": target_col}, buffer)
        buffer.seek(0)
        st.download_button("Download trained_pipeline.pkl", buffer, file_name="trained_pipeline.pkl")

        st.success("Training complete. You can now use the model for inference below.")

# -------------- Inference Section --------------
st.header("🔮 Inference")
st.caption("Use the trained pipeline to predict prices for new data (upload CSV with the same columns as training features).")

infer_file = st.file_uploader("Upload inference CSV (same schema as training features)", type=["csv"], key="infer_csv")
if infer_file is not None:
    try:
        infer_df = pd.read_csv(infer_file)
        st.write("Inference data preview:")
        st.dataframe(infer_df.head(20))
    except Exception as e:
        st.error(f"Failed to read inference CSV: {e}")
        infer_df = None
else:
    infer_df = None

uploaded_model = st.file_uploader("Upload a trained pipeline (.pkl) from this app (optional; otherwise uses last trained model in session)", type=["pkl"], key="model_pkl")

# Session state to keep last trained model
if "last_pipe" not in st.session_state:
    st.session_state["last_pipe"] = None
if "last_target" not in st.session_state:
    st.session_state["last_target"] = None

if uploaded_model is not None:
    try:
        obj = pickle.loads(uploaded_model.read())
        st.session_state["last_pipe"] = obj["pipeline"]
        st.session_state["last_target"] = obj["target"]
        st.success("Loaded pipeline from uploaded .pkl.")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")

# Provide a small manual input form for quick test (optional)
with st.expander("🧪 Manual Single-Row Input (builds from first row of your dataset)"):
    if df is not None:
        X_manual = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
        defaults = X_manual.iloc[0].to_dict()
        form_values = {}
        with st.form("manual_input_form"):
            st.write("Edit values then click Predict")
            for k, v in defaults.items():
                if isinstance(v, (int, np.integer)):
                    form_values[k] = st.number_input(k, value=int(v))
                elif isinstance(v, (float, np.floating)):
                    form_values[k] = st.number_input(k, value=float(v))
                else:
                    form_values[k] = st.text_input(k, value=str(v))
            submitted = st.form_submit_button("Predict")
        if submitted:
            if st.session_state["last_pipe"] is None:
                st.warning("Please train a model (or upload a .pkl) first.")
            else:
                df_row = pd.DataFrame([form_values])
                try:
                    pred = st.session_state["last_pipe"].predict(df_row)[0]
                    st.success(f"Predicted price: {pred:,.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

if st.button("Run Inference on CSV"):
    if infer_df is None:
        st.warning("Please upload an inference CSV first.")
    elif st.session_state["last_pipe"] is None:
        st.warning("Please train a model (or upload a .pkl) first.")
    else:
        try:
            preds = st.session_state["last_pipe"].predict(infer_df)
            out = infer_df.copy()
            out["prediction"] = preds
            st.write("Predictions preview:")
            st.dataframe(out.head(20))

            # Offer download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Inference failed: {e}")
