import joblib
import pandas as pd
import streamlit as st

from config import (
    CLF_MODEL_PATH,
    CLF_SCALER_PATH,
    DATA_PATH,
    FEATURE_COLUMNS_FALLBACK,
    FURNISH_MAP,
    GRADE_LABELS,
    NEIGHBORHOODS,
    RAW_NUMERIC_COLUMNS,
    REG_MODEL_PATH,
    REG_SCALER_PATH,
)


# Load the processed training data used for medians and category defaults.
@st.cache_data
# Load the processed training data used for medians and category defaults.
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# Load model artifacts once and verify both models expect the same feature order.
@st.cache_resource
# Load model artifacts once and verify both models expect the same feature order.
def load_artifacts(
    reg_model_mtime: int,
    reg_scaler_mtime: int,
    clf_model_mtime: int,
    clf_scaler_mtime: int,
):
    reg_model = joblib.load(REG_MODEL_PATH)
    reg_scaler = joblib.load(REG_SCALER_PATH)
    clf_model = joblib.load(CLF_MODEL_PATH)
    clf_scaler = joblib.load(CLF_SCALER_PATH)

    reg_cols = list(reg_scaler.feature_names_in_) if hasattr(reg_scaler, "feature_names_in_") else []
    clf_cols = list(clf_scaler.feature_names_in_) if hasattr(clf_scaler, "feature_names_in_") else []

    if reg_cols and clf_cols and reg_cols != clf_cols:
        raise ValueError("Regression and classification scalers use different feature orders.")

    cols = reg_cols or clf_cols or FEATURE_COLUMNS_FALLBACK

    if len(cols) != len(FEATURE_COLUMNS_FALLBACK):
        raise ValueError(f"Unexpected feature count: got {len(cols)}, expected {len(FEATURE_COLUMNS_FALLBACK)}.")

    missing = [c for c in FEATURE_COLUMNS_FALLBACK if c not in cols]
    if missing:
        raise ValueError(f"Missing required features in scaler: {missing}")

    return reg_model, reg_scaler, clf_model, clf_scaler, cols


# Bundle all data, model artifacts, and defaults needed by the Streamlit pages.
def load_runtime_context():
    df = load_clean_data()
    reg_model, reg_scaler, clf_model, clf_scaler, feature_columns = load_artifacts(
        int(REG_MODEL_PATH.stat().st_mtime_ns),
        int(REG_SCALER_PATH.stat().st_mtime_ns),
        int(CLF_MODEL_PATH.stat().st_mtime_ns),
        int(CLF_SCALER_PATH.stat().st_mtime_ns),
    )

    return {
        "df": df,
        "reg_model": reg_model,
        "reg_scaler": reg_scaler,
        "clf_model": clf_model,
        "clf_scaler": clf_scaler,
        "feature_columns": feature_columns,
        "defaults": default_values(df, feature_columns),
        "default_categories": default_categories(df),
    }


# Median defaults fill fields that the prompt or CSV did not provide.
def default_values(df, feature_columns):
    return {
        col: float(pd.to_numeric(df[col], errors="coerce").median()) if col in df.columns else 0.0
        for col in feature_columns
    }


# Most-common categorical defaults keep prompt inference possible with partial input.
def default_categories(df):
    reverse_furnish = {v: k for k, v in FURNISH_MAP.items()}
    furnishing = "Semi-furnished"
    if "Furnishing_Status" in df.columns:
        mode = pd.to_numeric(df["Furnishing_Status"], errors="coerce").dropna().mode()
        if not mode.empty:
            furnishing = reverse_furnish.get(int(round(mode.iloc[0])), furnishing)

    neighborhood = "Downtown"
    encoded_cols = [f"Neighborhood_{n}" for n in NEIGHBORHOODS[1:] if f"Neighborhood_{n}" in df.columns]
    if encoded_cols:
        encoded = df[encoded_cols].astype(float)
        counts = encoded.sum()
        downtown_count = len(df) - encoded.max(axis=1).sum()
        counts.loc["Neighborhood_Downtown"] = downtown_count
        winner = counts.idxmax()
        neighborhood = winner.replace("Neighborhood_", "")

    return furnishing, neighborhood


# Convert user-facing raw inputs into the encoded feature row used during training.
def build_feature_row(numeric_inputs, furnishing, neighborhood, feature_columns):
    row = {c: 0.0 for c in feature_columns}
    for c, v in numeric_inputs.items():
        if c in row:
            row[c] = float(v)
    if "Furnishing_Status" in row:
        row["Furnishing_Status"] = float(FURNISH_MAP[furnishing])
    for n in NEIGHBORHOODS[1:]:
        key = f"Neighborhood_{n}"
        if key in row:
            row[key] = 1.0 if neighborhood == n else 0.0
    return pd.DataFrame([row], columns=feature_columns)


# Normalize uploaded raw CSV rows into the saved model's feature schema.
def raw_to_feature_frame(raw_df, feature_columns, defaults):
    out = pd.DataFrame(0.0, index=raw_df.index, columns=feature_columns)
    for col in RAW_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(defaults.get(col, 0.0))
    if "Furnishing_Status" in out.columns:
        out["Furnishing_Status"] = raw_df["Furnishing_Status"].map(FURNISH_MAP).fillna(1.0)
    for n in NEIGHBORHOODS[1:]:
        key = f"Neighborhood_{n}"
        if key in out.columns:
            out[key] = (raw_df["Neighborhood"] == n).astype(float)
    return out


# Run the regression and classification models on already-built features.
def run_predict(features, reg_model, reg_scaler, clf_model, clf_scaler):
    price = reg_model.predict(reg_scaler.transform(features))
    price = price.clip(min=0)
    grade = clf_model.predict(clf_scaler.transform(features))
    probs = clf_model.predict_proba(clf_scaler.transform(features))
    return price, grade, probs


# Convenience wrapper for one confirmed property from prompt/manual input.
def predict_single(numeric_inputs, furnishing, neighborhood, context):
    row = build_feature_row(numeric_inputs, furnishing, neighborhood, context["feature_columns"])
    prices, grades, probs = run_predict(
        row,
        context["reg_model"],
        context["reg_scaler"],
        context["clf_model"],
        context["clf_scaler"],
    )
    return {
        "price": float(prices[0]),
        "grade": int(grades[0]),
        "confidence": float(probs[0].max()),
        "probabilities": {GRADE_LABELS.get(i, str(i)): float(probs[0][i]) for i in range(probs.shape[1])},
    }
