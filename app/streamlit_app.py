import sys
from pathlib import Path

import sklearn
import streamlit as st
import xgboost

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import CLF_MODEL_PATH, CLF_SCALER_PATH, DATA_PATH, REG_MODEL_PATH, REG_SCALER_PATH
from pages import page_about, page_agent, page_csv, page_manual
from prediction_nodes import load_runtime_context
from styles import apply_styles


# Make imports work both in Streamlit and in Streamlit's test runner.
st.set_page_config(
    page_title="Property Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)
apply_styles()


# Main app shell: validate files, load context, route to the selected page.
def main():
    required = [DATA_PATH, REG_MODEL_PATH, REG_SCALER_PATH, CLF_MODEL_PATH, CLF_SCALER_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Missing required files:\n" + "\n".join(f"- {m}" for m in missing))
        st.stop()

    context = load_runtime_context()

    st.markdown(
        """
<div style='margin-bottom:1.2rem'>
  <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#999;margin-bottom:0.2rem'>
    Real Estate ML
  </div>
  <h1 style='margin:0;padding:0'>Property Predictor</h1>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Prompt Agent", "CSV Upload", "Manual Input", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "Prompt Agent":
        page_agent(context)
    elif page == "CSV Upload":
        page_csv(context)
    elif page == "Manual Input":
        page_manual(context)
    else:
        page_about()

    with st.expander("Runtime diagnostics"):
        st.write(
            {
                "python": sys.version.split()[0],
                "streamlit": st.__version__,
                "xgboost": xgboost.__version__,
                "scikit-learn": sklearn.__version__,
                "feature_count": len(context["feature_columns"]),
            }
        )


if __name__ == "__main__":
    main()
