import streamlit as st


def apply_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --background-color: #F7F6F3;
    --secondary-background-color: #FFFFFF;
    --text-color: #1C1C1C;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: 'Outfit', sans-serif !important;
    background: #F7F6F3 !important;
    color: #1C1C1C !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNavCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"],
button[kind="header"],
[aria-label="Open sidebar"],
[aria-label="Close sidebar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
}

[data-testid="stSidebar"] { display: none !important; width: 0 !important; }
.block-container {
    max-width: 920px !important;
    padding: 2.2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1C1C1C !important;
    letter-spacing: 0 !important;
    margin-bottom: 0.1rem !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1C1C1C !important;
    margin-top: 1.35rem !important;
    margin-bottom: 0.5rem !important;
}
h3 { font-size: 0.9rem !important; font-weight: 600 !important; color: #1C1C1C !important; }
p, li, span { color: #1C1C1C !important; }

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
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 0.45rem 0.95rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    cursor: pointer !important;
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

.info-card, .summary-card {
    background: #FFFFFF;
    border: 1px solid #E4E2DC;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: #555 !important;
    line-height: 1.5;
}
.summary-card {
    min-height: 96px;
    margin-bottom: 0.5rem;
}
.summary-card .label {
    color: #888 !important;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.summary-card .value {
    color: #1C1C1C !important;
    font-size: 1.1rem;
    font-weight: 700;
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

[data-testid="stWidgetLabel"] p,
label {
    color: #333 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stNumberInput"] input,
input[type="number"],
textarea {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    border: 1.5px solid #DDDBD6 !important;
    border-radius: 9px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.8rem !important;
    width: 100% !important;
}
[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    -moz-appearance: textfield !important;
}
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
    display: none !important;
}
[data-testid="stNumberInput"] input:focus,
input[type="number"]:focus,
textarea:focus {
    border-color: #1C1C1C !important;
    box-shadow: 0 0 0 3px rgba(28,28,28,0.08) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"] button { display: none !important; }

[data-baseweb="select"] > div,
[data-baseweb="select"] > div:hover {
    background: #FFFFFF !important;
    border: 1.5px solid #DDDBD6 !important;
    border-radius: 9px !important;
}
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
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div,
body [data-baseweb="popover"] {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    border-radius: 10px !important;
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
[data-baseweb="menu"] *,
ul[role="listbox"] *,
body [data-baseweb="popover"] * { color: #1C1C1C !important; }

.stButton > button,
.stDownloadButton > button {
    background: #1C1C1C !important;
    color: #F7F6F3 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0 !important;
    width: 100% !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover { background: #333333 !important; }
.stButton > button *,
.stDownloadButton > button * {
    color: #F7F6F3 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    color: #1C1C1C !important;
}
[data-testid="stFileUploader"] section {
    border: 1.5px dashed #CCCAC4 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #1C1C1C !important; }

[data-testid="stMetric"] {
    background: #1C1C1C !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem !important;
}
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #888 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] * {
    color: #F7F6F3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.25rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }
[data-testid="stDataFrame"] {
    border: 1px solid #E4E2DC !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; }

.section-label {
    display: block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #999 !important;
    margin-bottom: 0.5rem;
}
hr {
    border: none !important;
    border-top: 1px solid #E4E2DC !important;
    margin: 1.35rem 0 !important;
}
code {
    background: #EEECEA !important;
    color: #1C1C1C !important;
    border-radius: 5px !important;
    padding: 0.15em 0.4em !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stDialog"],
[data-testid="stDialog"] > div {
    max-width: none !important;
}
div[role="dialog"] {
    width: min(900px, calc(100vw - 2rem)) !important;
    max-width: min(900px, calc(100vw - 2rem)) !important;
    margin: auto !important;
    border-radius: 12px !important;
}
[data-testid="stDialog"] [data-testid="stVerticalBlock"],
div[role="dialog"] [data-testid="stVerticalBlock"] {
    gap: 0.65rem !important;
}
div[role="dialog"] h2,
div[role="dialog"] h3 {
    margin-top: 0 !important;
}
div[role="dialog"] [data-testid="stDataFrame"] {
    max-height: 52vh !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
