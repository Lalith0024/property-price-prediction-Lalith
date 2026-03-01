from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st
import xgboost
import sklearn

st.set_page_config(
    page_title="Property Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — one comprehensive, conflict-free block
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Force light theme tokens even when host has dark defaults */
:root {
    --background-color: #F7F6F3;
    --secondary-background-color: #FFFFFF;
    --text-color: #1C1C1C;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: 'Outfit', sans-serif !important;
    background: #F7F6F3 !important;
    color: #1C1C1C !important;
}

/* ── Hide ALL Streamlit chrome that shouldn't show ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNavCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"],
.st-emotion-cache-zq5wmm,
.st-emotion-cache-1dp5vir,
button[kind="header"],
[aria-label="Open sidebar"],
[aria-label="Close sidebar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
}

/* ── Hide sidebar entirely (no collapse arrow, no ghost panel) ── */
[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
}

/* ── Page layout ── */
.block-container {
    max-width: 820px !important;
    padding: 2.5rem 2rem 5rem !important;
    margin: 0 auto !important;
}

/* ── Typography ── */
h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1C1C1C !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.1rem !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1C1C1C !important;
    margin-top: 2rem !important;
    margin-bottom: 0.5rem !important;
}
h3 { font-size: 0.9rem !important; font-weight: 600 !important; color: #1C1C1C !important; }
p, li, span { color: #1C1C1C !important; }

/* ── Horizontal radio nav ── */
div[data-testid="stHorizontalBlock"]:has([data-testid="stRadio"]) {
    background: #EFEFEB;
    border-radius: 12px;
    padding: 4px;
    display: inline-flex;
    gap: 2px;
}
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 2px !important;
}
[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    border: none !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}
[data-testid="stRadio"] label input { display: none !important; }
[data-testid="stWidgetLabel"]:has(+ [data-testid="stRadio"]) { display: none !important; }

/* ── Info card ── */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E4E2DC;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.5rem;
    font-size: 0.875rem;
    color: #555 !important;
    line-height: 1.5;
}
.info-card * { color: #555 !important; }
.info-card code {
    background: #F0EFEA !important;
    color: #1C1C1C !important;
    padding: 0.1em 0.35em;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p,
label {
    color: #333 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.3rem !important;
}

/* ── Number inputs ── fully white, black text, no spinner styling issues ── */
[data-testid="stNumberInput"] input,
input[type="number"] {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    border: 1.5px solid #DDDBD6 !important;
    border-radius: 9px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.55rem 0.8rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    width: 100% !important;
    -moz-appearance: textfield !important;
}
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
    display: none !important;
}
[data-testid="stNumberInput"] input:focus,
input[type="number"]:focus {
    border-color: #1C1C1C !important;
    box-shadow: 0 0 0 3px rgba(28,28,28,0.08) !important;
    outline: none !important;
    background: #FFFFFF !important;
    color: #1C1C1C !important;
}

/* Hide the +/- stepper buttons on number inputs */
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"] button {
    display: none !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div,
[data-baseweb="select"] > div:hover {
    background: #FFFFFF !important;
    border: 1.5px solid #DDDBD6 !important;
    border-radius: 9px !important;
    transition: border-color 0.15s !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #1C1C1C !important;
    box-shadow: 0 0 0 3px rgba(28,28,28,0.08) !important;
}
/* ALL text inside select trigger */
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] p,
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    color: #1C1C1C !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    background: transparent !important;
}

/* ── Dropdown menu popup ── THIS is what was dark before ── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
}
/* BaseWeb popovers render in a portal; enforce colors there as well */
body [data-baseweb="popover"] {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
}
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1px solid #DDDBD6 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
    overflow: hidden !important;
    padding: 4px !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
ul[role="listbox"] li {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    border-radius: 7px !important;
    padding: 0.5rem 0.85rem !important;
    margin: 1px 0 !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [aria-selected="true"],
ul[role="listbox"] li:hover,
body [role="option"]:hover,
body [role="option"][aria-selected="true"] {
    background: #F0EFEA !important;
    color: #1C1C1C !important;
    font-weight: 500 !important;
}
/* Any text node inside the menu */
[data-baseweb="menu"] *,
ul[role="listbox"] *,
body [data-baseweb="popover"] * {
    color: #1C1C1C !important;
}

/* ── Primary button ── */
.stButton > button,
.stDownloadButton > button {
    background: #1C1C1C !important;
    color: #F7F6F3 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #333333 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
/* Force text color inside button — every possible child */
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div,
.stDownloadButton > button *,
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #F7F6F3 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
}
[data-testid="stFileUploader"] section {
    border: 1.5px dashed #CCCAC4 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #1C1C1C !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: #1C1C1C !important;
    color: #F7F6F3 !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploaderDropzone"] button * {
    color: #F7F6F3 !important;
}

/* ── Metric result cards ── */
[data-testid="stMetric"] {
    background: #1C1C1C !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.5rem !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] * {
    color: #888 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #F7F6F3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E4E2DC !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Section label ── */
.section-label {
    display: block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #999 !important;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #E4E2DC !important;
    margin: 1.75rem 0 !important;
}

/* ── About: code & tables ── */
code {
    background: #EEECEA !important;
    color: #1C1C1C !important;
    border-radius: 5px !important;
    padding: 0.15em 0.4em !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
pre {
    background: #EEECEA !important;
    border-radius: 10px !important;
    padding: 1.1rem !important;
    overflow-x: auto !important;
}
pre code { background: transparent !important; padding: 0 !important; }
[data-testid="stCodeBlock"] {
    background: #EEECEA !important;
    border-radius: 10px !important;
    border: 1px solid #E4E2DC !important;
}
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code { background: transparent !important; color: #1C1C1C !important; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin: 0.75rem 0 1.5rem; }
th { background: #EEECEA; color: #1C1C1C !important; font-weight: 600; padding: 0.65rem 0.9rem; text-align: left; border: 1px solid #E4E2DC; }
td { padding: 0.6rem 0.9rem; border: 1px solid #E4E2DC; color: #1C1C1C !important; vertical-align: top; }
tr:nth-child(even) td { background: #F7F6F3; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
DATA_PATH       = BASE_DIR / "data" / "processed" / "real_estate_clean.csv"
REG_MODEL_PATH  = BASE_DIR / "models" / "xgb_regression_model.joblib"
REG_SCALER_PATH = BASE_DIR / "models" / "regression_scaler.joblib"
CLF_MODEL_PATH  = BASE_DIR / "models" / "xgb_classification_model.joblib"
CLF_SCALER_PATH = BASE_DIR / "models" / "classification_scaler.joblib"

FURNISH_MAP   = {"Unfurnished": 0, "Semi-furnished": 1, "Fully-furnished": 2}
NEIGHBORHOODS = ["Downtown", "IT Hub", "Industrial", "Residential", "Suburban"]
INT_COLUMNS   = {"Bedrooms", "Bathrooms", "Age_of_Property", "Floor_Number"}
GRADE_LABELS  = {0: "0 — Low", 1: "1 — Medium", 2: "2 — High"}

FEATURE_COLUMNS_FALLBACK = [
    "Total_Square_Footage","Bedrooms","Bathrooms","Age_of_Property",
    "Floor_Number","Furnishing_Status","Distance_to_City_Center_km",
    "Proximity_to_Public_Transport_km","Crime_Index","Air_Quality_Index",
    "Neighborhood_Growth_Rate_%","Price_per_SqFt","Annual_Property_Tax",
    "Estimated_Rental_Yield_%","Neighborhood_IT Hub",
    "Neighborhood_Industrial","Neighborhood_Residential","Neighborhood_Suburban",
]

RAW_NUMERIC_COLUMNS = [
    "Total_Square_Footage","Bedrooms","Bathrooms","Age_of_Property",
    "Floor_Number","Distance_to_City_Center_km","Proximity_to_Public_Transport_km",
    "Crime_Index","Air_Quality_Index","Neighborhood_Growth_Rate_%",
    "Price_per_SqFt","Annual_Property_Tax","Estimated_Rental_Yield_%",
]

LABELS = {
    "Total_Square_Footage":             "Total Square Footage",
    "Bedrooms":                         "Bedrooms",
    "Bathrooms":                        "Bathrooms",
    "Age_of_Property":                  "Age of Property (years)",
    "Floor_Number":                     "Floor Number",
    "Distance_to_City_Center_km":       "Distance to City Center (km)",
    "Proximity_to_Public_Transport_km": "Proximity to Public Transport (km)",
    "Crime_Index":                      "Crime Index",
    "Air_Quality_Index":                "Air Quality Index",
    "Neighborhood_Growth_Rate_%":       "Neighbourhood Growth Rate (%)",
    "Price_per_SqFt":                   "Price per Sq Ft",
    "Annual_Property_Tax":              "Annual Property Tax",
    "Estimated_Rental_Yield_%":         "Estimated Rental Yield (%)",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifacts(
    reg_model_mtime: int,
    reg_scaler_mtime: int,
    clf_model_mtime: int,
    clf_scaler_mtime: int,
):
    reg_model   = joblib.load(REG_MODEL_PATH)
    reg_scaler  = joblib.load(REG_SCALER_PATH)
    clf_model   = joblib.load(CLF_MODEL_PATH)
    clf_scaler  = joblib.load(CLF_SCALER_PATH)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def default_values(df, feature_columns):
    return {
        col: float(pd.to_numeric(df[col], errors="coerce").median()) if col in df.columns else 0.0
        for col in feature_columns
    }


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


def run_predict(features, reg_model, reg_scaler, clf_model, clf_scaler):
    price = reg_model.predict(reg_scaler.transform(features))
    # Price cannot be negative in this domain; clip defensive lower bound.
    price = price.clip(min=0)
    grade = clf_model.predict(clf_scaler.transform(features))
    probs = clf_model.predict_proba(clf_scaler.transform(features))
    return price, grade, probs


# ─────────────────────────────────────────────────────────────────────────────
#  Pages
# ─────────────────────────────────────────────────────────────────────────────
def page_csv(defaults, feature_columns, reg_model, reg_scaler, clf_model, clf_scaler):
    st.title("CSV Upload")
    st.markdown(
        "<div class='info-card'>Upload a CSV to get bulk predictions. "
        "Accepts encoded feature columns <em>or</em> raw columns with "
        "<code>Furnishing_Status</code> and <code>Neighborhood</code>.</div>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    if file is None:
        st.caption("Drag & drop or click Browse to select a CSV file.")
        return

    input_df = pd.read_csv(file)
    st.markdown(f"<span class='section-label'>{len(input_df)} rows · preview</span>", unsafe_allow_html=True)
    st.dataframe(input_df.head(10), use_container_width=True, height=210)

    has_encoded = all(c in input_df.columns for c in feature_columns)
    has_raw = (
        all(c in input_df.columns for c in RAW_NUMERIC_COLUMNS)
        and "Furnishing_Status" in input_df.columns
        and "Neighborhood" in input_df.columns
    )

    if not has_encoded and not has_raw:
        st.error("Column mismatch. Provide encoded feature columns or raw columns with Furnishing_Status and Neighborhood.")
        return

    if st.button("Run Predictions", use_container_width=True):
        features = input_df[feature_columns].copy() if has_encoded else raw_to_feature_frame(input_df, feature_columns, defaults)
        prices, grades, probs = run_predict(features, reg_model, reg_scaler, clf_model, clf_scaler)

        out = input_df.copy()
        out["Predicted_Price_INR"]        = prices
        out["Predicted_Investment_Grade"] = grades
        for i in range(probs.shape[1]):
            out[f"Grade_{i}_Probability"] = probs[:, i]

        st.markdown("---")
        st.markdown(f"<span class='section-label'>Results — {len(out)} rows</span>", unsafe_allow_html=True)
        st.dataframe(out, use_container_width=True, height=300)
        st.download_button(
            "⬇  Download predictions as CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="property_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def page_manual(defaults, feature_columns, reg_model, reg_scaler, clf_model, clf_scaler):
    st.title("Manual Input")
    st.markdown(
        "<div class='info-card'>Fill in the property details below. "
        "Integer fields only accept whole numbers — decimals are not allowed. "
        "Click <strong>Predict</strong> when ready.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    numeric_inputs = {}

    for idx, col in enumerate(RAW_NUMERIC_COLUMNS):
        target = c1 if idx % 2 == 0 else c2
        base   = defaults.get(col, 0.0)
        with target:
            if col in INT_COLUMNS:
                # Integer fields — no decimals, no letters (st.number_input enforces this natively)
                numeric_inputs[col] = st.number_input(
                    LABELS[col],
                    min_value=0,
                    value=int(round(base)),
                    step=1,
                    format="%d",
                    key=f"f_{col}",
                )
            else:
                # Decimal fields — 2 dp, no letters
                numeric_inputs[col] = st.number_input(
                    LABELS[col],
                    value=round(base, 2),
                    step=0.01,
                    format="%.2f",
                    key=f"f_{col}",
                )

    st.markdown("---")

    d1, d2 = st.columns(2, gap="medium")
    with d1:
        furnishing = st.selectbox("Furnishing Status", list(FURNISH_MAP.keys()), index=1)
    with d2:
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)

    st.markdown("---")

    if st.button("Predict", use_container_width=True):
        row = build_feature_row(numeric_inputs, furnishing, neighborhood, feature_columns)
        prices, grades, probs = run_predict(row, reg_model, reg_scaler, clf_model, clf_scaler)

        st.session_state["last_result"] = {
            "price":      float(prices[0]),
            "grade":      int(grades[0]),
            "confidence": float(probs[0].max()),
        }

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Price", f"₹{r['price']:,.0f}")
        m2.metric("Investment Grade", GRADE_LABELS.get(r["grade"], str(r["grade"])))
        m3.metric("Confidence", f"{r['confidence']:.1%}")


def page_about():
    st.title("About")
    st.markdown("""
<div class='info-card'>End-to-end ML pipeline predicting <strong>property market price</strong>
and <strong>investment grade</strong> using XGBoost, trained on Indian real estate data.</div>
""", unsafe_allow_html=True)
    st.markdown(f"Hosted App: [{HOSTED_APP_URL}]({HOSTED_APP_URL})")

    st.markdown("## System Architecture")
    st.markdown("""
**Training pipeline**

1. Raw CSV loaded from `data/raw/real_estate_raw.csv`
2. Preprocessing — median imputation · ordinal encoding for `Furnishing_Status` (0/1/2) · one-hot encoding for `Neighborhood` (drop-first) · IQR clipping (target columns excluded)
3. Two XGBoost models trained independently: `XGBRegressor` → price · `XGBClassifier` → investment grade
4. Models and scalers saved as `.joblib` artifacts to `models/`

**Inference pipeline**

1. Artifacts loaded at startup (cached)
2. Input — manual form or CSV upload
3. Features validated, ordinal-mapped, and one-hot transformed
4. Scaler normalises → model predicts price + grade + class probabilities
""")

    st.markdown("## Models")
    st.markdown("""
| | Regression | Classification |
|---|---|---|
| **Algorithm** | XGBRegressor | XGBClassifier |
| **Target** | Current_Market_Price | Investment_Grade (0 / 1 / 2) |
| **n_estimators** | 200 | 200 |
| **learning_rate** | 0.05 | 0.05 |
| **max_depth** | 5 | 4 |
| **Scaling** | MinMaxScaler | MinMaxScaler |
""", unsafe_allow_html=True)

    st.markdown("## Results")
    st.markdown("""
| Metric | Value |
|---|---|
| R² Score | **0.9483** |
| MAE | ₹ 550,851 |
| RMSE | ₹ 1,010,636 |
| Classification Accuracy | **97.50 %** |
| Weighted F1 | **0.97** |
""", unsafe_allow_html=True)

    st.markdown("## Data Processing")
    st.markdown("""
- **Missing values** — imputed with column median
- **Furnishing_Status** — Unfurnished → 0 · Semi-furnished → 1 · Fully-furnished → 2
- **Neighborhood** — one-hot encoded, drop-first
- **Outliers** — IQR clipping, excluding `Current_Market_Price` and `Investment_Grade`
- **Split** — 80 / 20, `random_state=42`
""")

    st.markdown("## Repository")
    st.code("""property-price-prediction/
├── app/streamlit_app.py
├── data/
│   ├── raw/real_estate_raw.csv
│   └── processed/real_estate_clean.csv
├── models/
│   ├── xgb_regression_model.joblib
│   ├── regression_scaler.joblib
│   ├── xgb_classification_model.joblib
│   └── classification_scaler.joblib
├── notebooks/training_colab.ipynb
└── README.md""", language="text")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    required = [DATA_PATH, REG_MODEL_PATH, REG_SCALER_PATH, CLF_MODEL_PATH, CLF_SCALER_PATH]
    missing  = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Missing required files:\n" + "\n".join(f"- {m}" for m in missing))
        st.stop()

    df = load_clean_data()
    reg_model, reg_scaler, clf_model, clf_scaler, feature_columns = load_artifacts(
        int(REG_MODEL_PATH.stat().st_mtime_ns),
        int(REG_SCALER_PATH.stat().st_mtime_ns),
        int(CLF_MODEL_PATH.stat().st_mtime_ns),
        int(CLF_SCALER_PATH.stat().st_mtime_ns),
    )
    defaults = default_values(df, feature_columns)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style='margin-bottom:1.5rem'>
  <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;color:#999;margin-bottom:0.2rem'>
    Real Estate ML
  </div>
  <h1 style='margin:0;padding:0'>Property Predictor</h1>
</div>
""", unsafe_allow_html=True)

    # ── Horizontal tab nav ───────────────────────────────────────────────────
    page = st.radio(
        "Navigation",
        ["CSV Upload", "Manual Input", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "CSV Upload":
        page_csv(defaults, feature_columns, reg_model, reg_scaler, clf_model, clf_scaler)
    elif page == "Manual Input":
        page_manual(defaults, feature_columns, reg_model, reg_scaler, clf_model, clf_scaler)
    else:
        page_about()

    with st.expander("Runtime diagnostics"):
        st.write(
            {
                "python": sys.version.split()[0],
                "streamlit": st.__version__,
                "xgboost": xgboost.__version__,
                "scikit-learn": sklearn.__version__,
                "feature_count": len(feature_columns),
            }
        )


if __name__ == "__main__":
    main()
