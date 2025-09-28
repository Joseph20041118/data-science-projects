# app_v3.py — House Price Prediction (Regression) with safer target handling and smarter defaults
# Run: streamlit run app_v3.py

import os
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# ---------------- UI ----------------
st.set_page_config(page_title="House Price Prediction (v3)", layout="wide")
st.title("🏡 House Price Prediction — Interactive ML App (v3)")
st.caption("Upload CSV → pick target (auto-suggested) → train → evaluate → export. Safer target checks, optional log-transform, CV, and permutation importance.")

# ---------------- Utils ----------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def coerce_numeric_inplace(df: pd.DataFrame, cols):
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    return df

def score_target_candidates(df: pd.DataFrame):
    """
    Score numeric columns for being a good regression target.
    Heuristics:
      - name contains 'price' / 'SalePrice' → big boost
      - larger std * range preferred
      - must have > 30 unique values
    """
    candidates = []
    for c in df.columns:
        if not is_numeric_series(df[c]):
            continue
        s = df[c].dropna()
        nunique = s.nunique()
        if nunique <= 30:
            continue
        rng = (s.max() - s.min())
        std = s.std(ddof=0)
        score = float(std * rng)
        name = c.lower()
        if "price" in name or "saleprice" in name:
            score *= 50.0
        candidates.append((c, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates]

def make_preprocessor(X: pd.DataFrame, ohe_min_freq=None, ohe_max_cat=None) -> ColumnTransformer:
    num_cols = [c for c in X.columns if is_numeric_series(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ))
    if cat_cols:
        ohe_kwargs = {"handle_unknown": "ignore"}
        if ohe_min_freq and ohe_min_freq > 0:
            ohe_kwargs["min_frequency"] = ohe_min_freq
        if ohe_max_cat and ohe_max_cat > 0:
            ohe_kwargs["max_categories"] = ohe_max_cat
        try:
            enc = OneHotEncoder(**ohe_kwargs)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore")

        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", enc)
            ]),
            cat_cols
        ))

    if not transformers:
        transformers.append(("pass", "passthrough", X.columns.tolist()))
    return ColumnTransformer(transformers)

def pick_model(name: str):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    if name == "XGBRegressor" and HAS_XGB:
        return XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist"
        )
    return LinearRegression()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_safe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_pred_vs_actual(y_true, y_pred, title="Predicted vs Actual"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

def plot_residuals(y_true, y_pred, title="Residuals Histogram"):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(title)
    st.pyplot(fig)

def safe_feature_names(prep, X_fit_sample: pd.DataFrame):
    try:
        return prep.get_feature_names_out()
    except Exception:
        try:
            n = prep.transform(X_fit_sample[:1]).shape[1]
            return np.array([f"f{i}" for i in range(n)])
        except Exception:
            return np.array([])

def target_warnings(y: pd.Series):
    msgs = []
    nunique = y.nunique(dropna=True)
    if nunique <= 10:
        msgs.append(f"Target has few unique values ({int(nunique)}). For price regression, choose a continuous numeric column.")
    if is_numeric_series(y):
        descr = y.describe()
        if float(descr.get("std", 0)) == 0:
            msgs.append("Target std is 0 — all values identical.")
        r = float(descr.get("max", 0) - descr.get("min", 0))
        if r < 10:
            msgs.append("Target range is very small — this looks like a label, not price.")
    else:
        msgs.append("Target dtype is non-numeric — try converting or pick another column.")
    return msgs

# ---------------- Data Load ----------------
st.sidebar.header("1) Upload Data")
uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
if uploaded is None:
    st.info("No file uploaded yet. Using included synthetic sample.")
    sample_path = os.path.join(os.path.dirname(__file__), "sample_houses.csv")
    df = pd.read_csv(sample_path)
else:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

if df.empty or df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns (features + target).")
    st.stop()

# Try coercing all numeric-like columns
df = coerce_numeric_inplace(df, df.columns)

st.write("### Dataset Preview")
st.dataframe(df.head(20))

# ---------------- Target Selection ----------------
st.sidebar.header("2) Select Target")
suggested = score_target_candidates(df)
target_default = 0
options = df.columns.tolist()
if suggested:
    # put suggested first in options ordering
    ordered = suggested + [c for c in options if c not in suggested]
    options = ordered
    target_default = 0

target_col = st.sidebar.selectbox(
    "Target column (numeric for regression):",
    options=options,
    index=target_default
)

# show EDA
with st.expander("🔎 Quick EDA Summary", expanded=False):
    st.write("#### Shape:", df.shape)
    st.write("#### dtypes")
    st.write(df.dtypes)
    st.write("#### Missing values")
    st.write(df.isna().sum().sort_values(ascending=False))
    if target_col in df.columns:
        st.write("#### Target stats")
        st.write(df[target_col].describe())

# target validation
msgs = target_warnings(df[target_col])
if msgs:
    for m in msgs:
        st.warning(m)

# ---------------- Options ----------------
st.sidebar.header("3) Train/Test & Options")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, 1)
do_cv = st.sidebar.checkbox("Also report 5-fold CV on training split", value=False)
use_log_target = st.sidebar.checkbox("Use log1p-transform on target (helps with skew)", value=True)
ohe_min_freq = st.sidebar.number_input("OneHot min_frequency (0=off)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
ohe_max_cat = st.sidebar.number_input("OneHot max_categories (0=off)", min_value=0, max_value=200, value=0, step=1)

# disable log-transform if target has nonpositive values
if use_log_target and (df[target_col] <= 0).any():
    st.info("Target contains nonpositive values → log1p disabled automatically.")
    use_log_target = False

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
        X = df.drop(columns=[target_col])
        y = df[target_col]

        preprocessor = make_preprocessor(
            X,
            ohe_min_freq if ohe_min_freq > 0 else None,
            ohe_max_cat if ohe_max_cat > 0 else None
        )

        base_model = pick_model(model_name)
        if use_log_target:
            model = TransformedTargetRegressor(
                regressor=base_model, func=np.log1p, inverse_func=np.expm1
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

            st.subheader("📊 Cross-Validation (5-fold)")
            st.write(f"**RMSE (mean ± sd):** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")
            st.write(f"**MAE (mean ± sd):** {mae_cv.mean():.3f} ± {mae_cv.std():.3f}")
            st.write(f"**R² (mean ± sd):** {r2_cv.mean():.3f} ± {r2_cv.std():.3f}")

            pipe.fit(X, y)
            feature_names = safe_feature_names(pipe.named_steps["prep"], X)
            X_eval, y_eval = X, y
            y_pred_eval = pipe.predict(X_eval)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae_val = mean_absolute_error(y_test, y_pred)
            rmse_val = rmse(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            mape_val = mape_safe(y_test, y_pred)

            st.subheader("📊 Hold-out Validation Metrics")
            st.write(f"**MAE:** {mae_val:,.3f}")
            st.write(f"**RMSE:** {rmse_val:,.3f}")
            st.write(f"**R²:** {r2_val:,.3f}")
            if not np.isnan(mape_val):
                st.write(f"**MAPE (%):** {mape_val:,.2f}")

            if do_cv:
                kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                neg_mse = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
                st.write(f"**Train 5-fold CV RMSE (mean):** {np.sqrt(-neg_mse).mean():.3f}")

            st.subheader("📈 Diagnostic Plots")
            plot_pred_vs_actual(y_test, y_pred)
            plot_residuals(y_test, y_pred)

            feature_names = safe_feature_names(pipe.named_steps["prep"], X_train)
            X_eval, y_eval, y_pred_eval = X_test, y_test, y_pred

        # Permutation importance
        with st.expander("🧠 Permutation Importance (Top 20) — may take time"):
            try:
                result = permutation_importance(pipe, X_eval, y_eval, n_repeats=5, random_state=random_state, n_jobs=-1)
                importances = result.importances_mean
                order = np.argsort(importances)[::-1]
                topn = min(20, len(importances))
                fig, ax = plt.subplots()
                labels = feature_names[order[:topn]] if len(feature_names) else [f"f{i}" for i in range(topn)]
                ax.barh(labels[::-1], importances[order[:topn]][::-1])
                ax.set_title("Permutation Importances (Top 20)")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Permutation importance unavailable: {e}")

        # Save model
        st.subheader("💾 Export Trained Model")
        meta = {
            "app_version": "v3",
            "model_name": model_name,
            "use_log_target": use_log_target,
            "target": target_col,
            "ohe_min_freq": ohe_min_freq,
            "ohe_max_cat": ohe_max_cat
        }
        buf = io.BytesIO()
        pickle.dump({"pipeline": pipe, "meta": meta}, buf)
        buf.seek(0)
        st.download_button("Download trained_pipeline.pkl", buf, file_name="trained_pipeline.pkl")

        # Keep session
        st.session_state["last_pipe"] = pipe
        st.session_state["last_meta"] = meta

        st.success("Training complete. You can now run inference below.")

# ---------------- Inference ----------------
st.header("🔮 Inference")
st.caption("Upload a features-only CSV (same schema as training features).")

infer_file = st.file_uploader("Inference CSV", type=["csv"], key="infer_csv")
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

uploaded_model = st.file_uploader("Or upload a trained .pkl (from this app)", type=["pkl"], key="model_pkl")

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

with st.expander("🧪 Manual Single-Row Input (builds from first row)"):
    if infer_file is not None and infer_df is not None and len(infer_df) > 0:
        base_df = infer_df.copy()
    else:
        # use training df minus target as template
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

