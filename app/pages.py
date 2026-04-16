import pandas as pd
import streamlit as st

from agent_flow import agent_audit_rows, api_settings_from, assemble_agent_fields
from config import FURNISH_MAP, GRADE_LABELS, INT_COLUMNS, LABELS, NEIGHBORHOODS, PROMPT_EXAMPLE, RAW_NUMERIC_COLUMNS
from ml_pipeline import build_feature_row, predict_single, raw_to_feature_frame, run_predict

SOURCE_LABELS = {
    "Groq extracted": "Prompt",
    "Rule parser fallback": "Prompt",
    "Training-data default": "Default",
    "Edited by user": "Edited",
}


# Show the common prediction output block used by prompt and manual pages.
def render_result(result):
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Price", f"Rs {result['price']:,.0f}")
    m2.metric("Investment Grade", GRADE_LABELS.get(result["grade"], str(result["grade"])))
    m3.metric("Confidence", f"{result['confidence']:.1%}")

    prob_df = pd.DataFrame(
        [{"Investment Grade": grade, "Probability": f"{prob:.1%}"} for grade, prob in result["probabilities"].items()]
    )
    st.dataframe(prob_df, width="stretch", hide_index=True, height=145)


# Show compact counts after prompt extraction.
def flow_summary(flow):
    found_count = sum(1 for value in flow["sources"].values() if value != "Training-data default")
    default_count = len(flow["sources"]) - found_count
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        f"<div class='summary-card'><div class='label'>Agent Source</div><div class='value'>{flow['agent_source']}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='summary-card'><div class='label'>Extracted Fields</div><div class='value'>{found_count}</div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='summary-card'><div class='label'>Defaulted Fields</div><div class='value'>{default_count}</div></div>",
        unsafe_allow_html=True,
    )


# Natural-language entry point: prompt -> Groq extraction -> user review -> ML.
def page_agent(context):
    st.title("Prompt Agent")
    st.markdown(
        "<div class='info-card'>Describe a property in normal text. "
        "The agent uses your Groq API key when available, fills missing values from the training data, "
        "and opens a review popup before prediction.</div>",
        unsafe_allow_html=True,
    )

    default_furnishing, default_neighborhood = context["default_categories"]
    settings = api_settings_from(st.secrets)
    if settings["api_key"]:
        st.caption(f"Groq API mode: {settings['model']}")
    else:
        st.caption("Using local fallback extractor. Add GROQ_API_KEY in .env or Streamlit secrets for Groq mode.")
    clear_stale_fallback_flow(settings)

    prompt = st.text_area(
        "Property prompt",
        key="agent_prompt_text",
        height=115,
        placeholder=PROMPT_EXAMPLE,
    )

    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        analyze = st.button("Analyze Prompt", width="stretch")
    with c2:
        reset = st.button("Start Over", width="stretch")

    if reset:
        for key in ["agent_flow", "agent_result", "agent_dialog"]:
            st.session_state.pop(key, None)
        st.rerun()

    if analyze:
        if not prompt.strip():
            st.warning("Enter a property description first.")
            return
        clear_agent_edit_keys()
        try:
            st.session_state["agent_flow"] = assemble_agent_fields(
                prompt,
                context["defaults"],
                default_furnishing,
                default_neighborhood,
                settings,
            )
        except Exception as exc:
            st.session_state.pop("agent_flow", None)
            st.session_state.pop("agent_result", None)
            st.session_state.pop("agent_dialog", None)
            st.error(f"Groq extraction failed: {exc}")
            return
        st.session_state["agent_dialog"] = "review"
        st.session_state.pop("agent_result", None)

    flow = st.session_state.get("agent_flow")
    if not flow:
        st.caption(PROMPT_EXAMPLE)
        return

    st.markdown("---")
    flow_summary(flow)

    if flow.get("agent_warning"):
        st.info("Fallback extractor active. Add GROQ_API_KEY to use the Groq parser.")

    b1, b2 = st.columns(2, gap="medium")
    with b1:
        if st.button("Review Parameters", width="stretch"):
            st.session_state["agent_dialog"] = "review"
    with b2:
        if st.button("Change Values", width="stretch"):
            st.session_state["agent_dialog"] = "edit"

    if "agent_result" in st.session_state:
        st.markdown("---")
        st.markdown("## Prediction")
        render_result(st.session_state["agent_result"])

    # Keep confirmation in a popup so the main page does not become scroll-heavy.
    @st.dialog("Review categorized parameters", width="large")
    # Keep confirmation in a popup so the main page does not become scroll-heavy.
    def review_dialog():
        flow = st.session_state["agent_flow"]
        audit = compact_audit_frame(flow)

        if flow.get("agent_warning"):
            st.caption("Fallback mode: GROQ_API_KEY is not configured.")

        st.dataframe(
            audit,
            width="stretch",
            height=420,
            hide_index=True,
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Source": st.column_config.TextColumn("Source", width="small"),
            },
        )

        defaulted = audit.loc[audit["Source"].eq("Default"), "Field"].tolist()
        if defaulted:
            with st.expander(f"{len(defaulted)} defaulted fields"):
                st.write(", ".join(defaulted))

        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            if st.button("Proceed", width="stretch"):
                st.session_state["agent_result"] = predict_single(
                    flow["numeric_inputs"],
                    flow["furnishing"],
                    flow["neighborhood"],
                    context,
                )
                st.session_state["agent_dialog"] = None
                st.rerun()
        with d2:
            if st.button("Change", width="stretch"):
                st.session_state["agent_dialog"] = "edit"
                st.rerun()
        with d3:
            if st.button("Close", width="stretch"):
                st.session_state["agent_dialog"] = None
                st.rerun()

    # Keep edits grouped by meaning instead of showing one long form.
    @st.dialog("Change categorized values", width="large")
    # Keep edits grouped by meaning instead of showing one long form.
    def edit_dialog():
        flow = st.session_state["agent_flow"]
        with st.form("agent_edit_form"):
            core, location, market = st.tabs(["Core", "Location", "Market"])
            edited_numeric = {}

            with core:
                cols = st.columns(2, gap="medium")
                for idx, col in enumerate(
                    ["Total_Square_Footage", "Bedrooms", "Bathrooms", "Age_of_Property", "Floor_Number"]
                ):
                    with cols[idx % 2]:
                        edited_numeric[col] = integer_or_decimal_input(col, flow["numeric_inputs"][col], f"agent_edit_{col}")
                edited_furnishing = st.selectbox(
                    "Furnishing Status",
                    list(FURNISH_MAP.keys()),
                    index=list(FURNISH_MAP.keys()).index(flow["furnishing"]),
                    key="agent_edit_furnishing",
                )

            with location:
                cols = st.columns(2, gap="medium")
                for idx, col in enumerate(
                    [
                        "Distance_to_City_Center_km",
                        "Proximity_to_Public_Transport_km",
                        "Crime_Index",
                        "Air_Quality_Index",
                    ]
                ):
                    with cols[idx % 2]:
                        edited_numeric[col] = integer_or_decimal_input(col, flow["numeric_inputs"][col], f"agent_edit_{col}")
                edited_neighborhood = st.selectbox(
                    "Neighborhood",
                    NEIGHBORHOODS,
                    index=NEIGHBORHOODS.index(flow["neighborhood"]),
                    key="agent_edit_neighborhood",
                )

            with market:
                cols = st.columns(2, gap="medium")
                for idx, col in enumerate(
                    [
                        "Neighborhood_Growth_Rate_%",
                        "Price_per_SqFt",
                        "Annual_Property_Tax",
                        "Estimated_Rental_Yield_%",
                    ]
                ):
                    with cols[idx % 2]:
                        edited_numeric[col] = integer_or_decimal_input(col, flow["numeric_inputs"][col], f"agent_edit_{col}")

            submitted = st.form_submit_button("Save & Predict", width="stretch")

        if submitted:
            updated_sources = mark_changed_sources(
                flow,
                edited_numeric,
                edited_furnishing,
                edited_neighborhood,
            )
            st.session_state["agent_flow"] = {
                "prompt": flow["prompt"],
                "numeric_inputs": edited_numeric,
                "furnishing": edited_furnishing,
                "neighborhood": edited_neighborhood,
                "sources": updated_sources,
                "agent_source": flow.get("agent_source", "Groq extracted"),
                "agent_warning": None,
            }
            st.session_state["agent_result"] = predict_single(
                edited_numeric,
                edited_furnishing,
                edited_neighborhood,
                context,
            )
            st.session_state["agent_dialog"] = None
            st.rerun()

    if st.session_state.get("agent_dialog") == "review":
        review_dialog()
    elif st.session_state.get("agent_dialog") == "edit":
        edit_dialog()


# Render a number input with the correct integer/decimal behavior for a field.
def integer_or_decimal_input(col, value, key):
    if col in INT_COLUMNS:
        return st.number_input(LABELS[col], min_value=0, value=int(round(value)), step=1, format="%d", key=key)
    return st.number_input(LABELS[col], value=round(float(value), 2), step=0.01, format="%.2f", key=key)


# Prepare review rows with friendly display labels and formatted values.
def compact_audit_frame(flow):
    audit = pd.DataFrame(agent_audit_rows(flow))
    audit["Value"] = audit["Value"].map(format_review_value)
    audit["Source"] = audit["Source"].map(lambda value: SOURCE_LABELS.get(value, value))
    return audit


# Mark only the fields that were actually changed in the edit popup.
def mark_changed_sources(flow, edited_numeric, edited_furnishing, edited_neighborhood):
    sources = dict(flow["sources"])
    for col in RAW_NUMERIC_COLUMNS:
        if values_differ(flow["numeric_inputs"][col], edited_numeric[col]):
            sources[col] = "Edited by user"

    if flow["furnishing"] != edited_furnishing:
        sources["Furnishing_Status"] = "Edited by user"
    if flow["neighborhood"] != edited_neighborhood:
        sources["Neighborhood"] = "Edited by user"

    return sources


# Compare numeric values without marking tiny float formatting differences as edits.
def values_differ(old_value, new_value):
    try:
        return abs(float(old_value) - float(new_value)) > 1e-9
    except (TypeError, ValueError):
        return old_value != new_value


# Format values cleanly for the review popup.
def format_review_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Clear edit-form widget state before analyzing a new prompt.
def clear_agent_edit_keys():
    for col in RAW_NUMERIC_COLUMNS:
        st.session_state.pop(f"agent_edit_{col}", None)
    st.session_state.pop("agent_edit_furnishing", None)
    st.session_state.pop("agent_edit_neighborhood", None)


# If a key is added mid-session, remove old fallback results before re-analysis.
def clear_stale_fallback_flow(settings):
    flow = st.session_state.get("agent_flow")
    if settings.get("api_key") and flow and flow.get("agent_source") == "Rule parser fallback":
        for key in ["agent_flow", "agent_result", "agent_dialog"]:
            st.session_state.pop(key, None)
        st.info("Groq key is now loaded. Re-analyze the prompt to use Groq extraction.")


# CSV page for batch predictions using either raw or already-encoded columns.
def page_csv(context):
    st.title("CSV Upload")
    st.markdown(
        "<div class='info-card'>Upload a CSV to get bulk predictions. "
        "Accepts encoded feature columns or raw columns with "
        "<code>Furnishing_Status</code> and <code>Neighborhood</code>.</div>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    if file is None:
        st.caption("Drag and drop or click Browse to select a CSV file.")
        return

    input_df = pd.read_csv(file)
    st.markdown(f"<span class='section-label'>{len(input_df)} rows preview</span>", unsafe_allow_html=True)
    st.dataframe(input_df.head(10), width="stretch", height=210)

    feature_columns = context["feature_columns"]
    has_encoded = all(c in input_df.columns for c in feature_columns)
    has_raw = (
        all(c in input_df.columns for c in RAW_NUMERIC_COLUMNS)
        and "Furnishing_Status" in input_df.columns
        and "Neighborhood" in input_df.columns
    )

    if not has_encoded and not has_raw:
        st.error("Column mismatch. Provide encoded feature columns or raw columns with Furnishing_Status and Neighborhood.")
        return

    if st.button("Run Predictions", width="stretch"):
        features = (
            input_df[feature_columns].copy()
            if has_encoded
            else raw_to_feature_frame(input_df, feature_columns, context["defaults"])
        )
        prices, grades, probs = run_predict(
            features,
            context["reg_model"],
            context["reg_scaler"],
            context["clf_model"],
            context["clf_scaler"],
        )

        out = input_df.copy()
        out["Predicted_Price_INR"] = prices
        out["Predicted_Investment_Grade"] = grades
        for i in range(probs.shape[1]):
            out[f"Grade_{i}_Probability"] = probs[:, i]

        st.markdown("---")
        st.markdown(f"<span class='section-label'>Results - {len(out)} rows</span>", unsafe_allow_html=True)
        st.dataframe(out, width="stretch", height=300)
        st.download_button(
            "Download predictions as CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="property_predictions.csv",
            mime="text/csv",
            width="stretch",
        )


# Manual page for directly entering structured model fields.
def page_manual(context):
    st.title("Manual Input")
    st.markdown(
        "<div class='info-card'>Fill in the property details below. "
        "Integer fields only accept whole numbers. Click <strong>Predict</strong> when ready.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    numeric_inputs = {}

    for idx, col in enumerate(RAW_NUMERIC_COLUMNS):
        target = c1 if idx % 2 == 0 else c2
        base = context["defaults"].get(col, 0.0)
        with target:
            numeric_inputs[col] = integer_or_decimal_input(col, base, f"f_{col}")

    st.markdown("---")

    d1, d2 = st.columns(2, gap="medium")
    with d1:
        furnishing = st.selectbox("Furnishing Status", list(FURNISH_MAP.keys()), index=1)
    with d2:
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)

    st.markdown("---")

    if st.button("Predict", width="stretch"):
        row = build_feature_row(numeric_inputs, furnishing, neighborhood, context["feature_columns"])
        prices, grades, probs = run_predict(
            row,
            context["reg_model"],
            context["reg_scaler"],
            context["clf_model"],
            context["clf_scaler"],
        )

        st.session_state["last_result"] = {
            "price": float(prices[0]),
            "grade": int(grades[0]),
            "confidence": float(probs[0].max()),
            "probabilities": {GRADE_LABELS.get(i, str(i)): float(probs[0][i]) for i in range(probs.shape[1])},
        }

    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"])


# About page summarizing the system for demo/project review.
def page_about():
    st.title("About")
    st.markdown(
        """
<div class='info-card'>End-to-end ML pipeline predicting <strong>property market price</strong>
and <strong>investment grade</strong> using XGBoost, with a Groq-powered prompt agent in front.</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## Agentic Flow")
    st.markdown(
        """
1. User describes a property in natural language.
2. The Groq extractor uses `GROQ_API_KEY` when configured.
3. Missing fields are filled from training-data defaults.
4. A review popup asks the user to proceed, change values, or close.
5. The confirmed values are converted into model features and sent to the saved XGBoost models.
"""
    )

    st.markdown("## Models")
    st.markdown(
        """
| | Regression | Classification |
|---|---|---|
| **Algorithm** | XGBRegressor | XGBClassifier |
| **Target** | Current_Market_Price | Investment_Grade (0 / 1 / 2) |
| **n_estimators** | 200 | 200 |
| **learning_rate** | 0.05 | 0.05 |
| **max_depth** | 5 | 4 |
| **Scaling** | MinMaxScaler | MinMaxScaler |
""",
        unsafe_allow_html=True,
    )

    st.markdown("## Results")
    st.markdown(
        """
| Metric | Value |
|---|---|
| R2 Score | **0.9483** |
| MAE | Rs 550,851 |
| RMSE | Rs 1,010,636 |
| Classification Accuracy | **97.50 %** |
| Weighted F1 | **0.97** |
""",
        unsafe_allow_html=True,
    )

    st.markdown("## Files")
    st.code(
        """app/
  streamlit_app.py   # app shell
  config.py          # paths, labels, feature lists
  agent_flow.py      # Groq extractor and fallback parser
  ml_pipeline.py     # model loading, defaults, prediction
  pages.py           # Streamlit pages and dialogs
  styles.py          # visual styling""",
        language="text",
    )
