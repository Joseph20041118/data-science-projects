
import os
import io
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional: XGBoost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="House Price Prediction (v2)", layout="wide")

st.title("🏡 House Price Prediction — Interactive ML App (v2)")
st.caption("Upload CSV → choose target → train → evaluate → export. Now with target checks, optional log-transform, and permutation importance.")

# ---------------- Utilities ----------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric if possible, preserving NaN on failures."""
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s

def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def make_preprocessor(X: pd.DataFrame, ohe_min_freq=None, ohe_max_cat=None) -> ColumnTransformer:
    num_cols = [c for c in X.columns if is_numeric_series(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if len(cat_cols) > 0:
        # set OHE config defensively across sklearn versions
        ohe_kwargs = {"handle_unknown": "ignore"}
        # Try to reduce cardinality if requested
        if ohe_min_freq is not None:
            ohe_kwargs["min_frequency"] = ohe_min_freq
        if ohe_max_cat is not None:
            ohe_kwargs["max_categories"] = ohe_max_cat
        try:
            enc = OneHotEncoder(**ohe_kwargs)
        except TypeError:
            # Fallback without new params if running on older sklearn
            enc = OneHotEncoder(handle_unknown="ignore")
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", enc)
        ]), cat_cols))

    if not transformers:
        transformers.append(("passthrough", "passthrough", X.columns.tolist()))
    return ColumnTransformer(transformers)

def pick_model(name: str):
    if name == "LinearRegression":
        return LinearRegression()
    elif name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    elif name == "XGBRegressor" and HAS_XGB:
        return XGBRegressor(
            n_estimators=600,
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
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
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

def safe_feature_names(prep, X_fit_sample: pd.DataFrame):
    try:
        return prep.get_feature_names_out()
    except Exception:
        try:
            # Transform one row to get dimensionality
            n = prep.transform(X_fit_sample[:1]).shape[1]
            return np.array([f"f{i}" for i in range(n)])
        except Exception:
            return np.array([f"f{i}" for i in range(1)])

def warn_if_bad_target(y: pd.Series):
    msgs = []
    nunique = pd.Series(y).nunique(dropna=True)
    if nunique <= 5:
        msgs.append(f"Target has very few unique values ({int(nunique)}). "
                    "If your goal is *price regression*, make sure the target is a continuous numeric variable.")
    if is_numeric_series(y):
        descr = y.describe()
        if descr.get("std", 0) == 0:
            msgs.append("Target standard deviation is 0 — all values are identical.")
        if descr.get("max", 0) - descr.get("min", 0) < 10:
            msgs.append("Target range is very small; values may be encoded labels instead of prices.")
    else:
        msgs.append("Target dtype is non-numeric. Attempt coercion or pick a numeric target.")
    return msgs

# ---------------- Sidebar: Upload ----------------
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

# ---------------- Target Selection ----------------
st.sidebar.header("2) Select Target")
# suggest numeric-like columns, but show all as fallback
numeric_like = [c for c in df.columns if is_numeric_series(df[c])]
target_col = st.sidebar.selectbox(
    "Target column (numeric for regression):",
    options=(numeric_like if numeric_like else df.columns.tolist()),
    index=0
)

# Coerce target to numeric if possible
df[target_col] = coerce_numeric(df[target_col])

with st.expander("🔎 Quick EDA Summary", expanded=False):
    st.write("#### Shape:", df.shape)
    st.write("#### dtypes")
    st.write(df.dtypes)
    st.write("#### Missing values")
    st.write(df.isna().sum().sort_values(ascending=False))
    st.write("#### Target stats")
    st.write(df[target_col].describe())

# Target sanity checks
messages = warn_if_bad_target(df[target_col])
if messages:
    for m in messages:
        st.warning(m)

# ---------------- Split / Options ----------------
st.sidebar.header("3) Train/Test & Options")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, 1)
do_cv = st.sidebar.checkbox("Also report 5-fold CV on training split", value=False)
use_log_target = st.sidebar.checkbox("Use log1p-transform on target (helps with skewed prices)", value=True)
ohe_min_freq = st.sidebar.number_input("OneHot min_frequency (0=off)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
ohe_max_cat = st.sidebar.number_input("OneHot max_categories (0=off)", min_value=0, max_value=200, value=0, step=1)

# ---------------- Model ----------------
st.sidebar.header("4) Model")
model_options = ["LinearRegression", "RandomForestRegressor"]
if HAS_XGB:
    model_options.append("XGBRegressor")
model_name = st.sidebar.selectbox("Choose a model", model_options, index=1)

# ---------------- Train ----------------
train_clicked = st.sidebar.button("🚀 Train Model", type="primary")

if train_clicked:
    with st.spinner("Training..."):
        X, y = split_features_target(df, target_col)
        preprocessor = make_preprocessor(X, ohe_min_freq if ohe_min_freq>0 else None,
                                            ohe_max_cat if ohe_max_cat>0 else None)
        base_model = pick_model(model_name)

        if use_log_target:
            # Wrap model with target transform
            model = TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=np.expm1
            )
        else:
            model = base_model

        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        too_small = len(df) < 80
        if too_small:
            st.warning("Small dataset detected (< 80 rows). Reporting 5-fold CV scores for robustness.")
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            neg_mse = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
            rmse_cv = np.sqrt(-neg_mse)
            mae_cv = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
            r2_cv = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
            st.write(f"**CV RMSE (mean ± sd):** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")
            st.write(f"**CV MAE (mean ± sd):** {mae_cv.mean():.3f} ± {mae_cv.std():.3f}")
            st.write(f"**CV R² (mean ± sd):** {r2_cv.mean():.3f} ± {r2_cv.std():.3f}")
            pipe.fit(X, y)
            feature_names = safe_feature_names(pipe.named_steps["prep"], X)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae_val = mean_absolute_error(y_test, y_pred)
            rmse_val = rmse(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)

            st.subheader("📊 Hold-out Validation Metrics")
            st.write(f"**MAE:** {mae_val:.3f}")
            st.write(f"**RMSE:** {rmse_val:.3f}")
            st.write(f"**R²:** {r2_val:.3f}")

            if do_cv:
                kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                neg_mse = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
                st.write(f"**Train 5-fold CV RMSE (mean):** {np.sqrt(-neg_mse).mean():.3f}")

            st.subheader("📈 Diagnostic Plots")
            plot_pred_vs_actual(y_test, y_pred)
            plot_residuals(y_test, y_pred)

            feature_names = safe_feature_names(pipe.named_steps["prep"], X_train)

        # Permutation importance (agnostic, optional for speed)
        with st.expander("🧠 Permutation Importance (on last fitted data) — may take time"):
            try:
                # Choose a small subsample for speed if data is large
                if too_small:
                    X_eval, y_eval = X, y
                else:
                    X_eval, y_eval = X_test, y_test
                result = permutation_importance(pipe, X_eval, y_eval, n_repeats=5, random_state=random_state, n_jobs=-1)
                importances = result.importances_mean
                order = np.argsort(importances)[::-1]
                topn = min(20, len(importances))
                fig, ax = plt.subplots()
                ax.barh([feature_names[i] for i in order[:topn]][::-1],
                        importances[order[:topn]][::-1])
                ax.set_title("Permutation Importances (Top 20)")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Permutation importance unavailable: {e}")

        # Save pipeline
        st.subheader("💾 Export Trained Model")
        meta = {
            "app_version": "v2",
            "model_name": model_name,
            "use_log_target": use_log_target,
            "target": target_col
        }
        buffer = io.BytesIO()
        pickle.dump({"pipeline": pipe, "meta": meta}, buffer)
        buffer.seek(0)
        st.download_button("Download trained_pipeline.pkl", buffer, file_name="trained_pipeline.pkl")

        # Keep in session
        st.session_state["last_pipe"] = pipe
        st.session_state["last_meta"] = meta

        st.success("Training complete. You can now use the model for inference below.")

# ---------------- Inference ----------------
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
if "last_meta" not in st.session_state:
    st.session_state["last_meta"] = None

if uploaded_model is not None:
    try:
        obj = pickle.loads(uploaded_model.read())
        st.session_state["last_pipe"] = obj.get("pipeline", None)
        st.session_state["last_meta"] = obj.get("meta", None)
        st.success("Loaded pipeline from uploaded .pkl.")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")

with st.expander("🧪 Manual Single-Row Input (builds from first row of your dataset)"):
    if infer_file is not None:
        base_df = infer_df.copy()
    else:
        base_df = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

    if base_df is not None and len(base_df) > 0:
        defaults = base_df.iloc[0].to_dict()
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
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Inference failed: {e}")
