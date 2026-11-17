
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# 1. FILE PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLOUDSIM_PATH = os.path.join(
    DATA_DIR, "week8_cloudsim_simulation_results.csv"
)

OPTIMIZED_PATH = os.path.join(
    DATA_DIR, "final_combined_results.csv"
)

TEST_FEATURES_PATH = os.path.join(
    DATA_DIR, "test_features_for_dashboard.csv"
)

CPU_MODEL_PATH = os.path.join(
    MODEL_DIR, "cpu_model.pkl"
)

MEMORY_MODEL_PATH = os.path.join(
    MODEL_DIR, "memory_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR, "model_features.json"
)


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Intelligent Cloud Resource Allocation",
    page_icon=":cloud:",
    layout="wide"
)


# ==========================================================
# PREMIUM DASHBOARD THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --border: rgba(148,170,210,.16); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(ellipse at 10% 0%, rgba(48,103,190,.20), transparent 34%),
      radial-gradient(ellipse at 90% 8%, rgba(35,198,190,.10), transparent 28%),
      linear-gradient(180deg,#0b1020 0%,#0d1425 55%,#0b1020 100%);
    color:#edf3ff;
}
[data-testid="stHeader"] { background:rgba(11,16,32,.72); }
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#111a2e 0%,#0d1425 100%);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] * { color:#dce7fa; }
.block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#f4f7ff !important; }
p,label,.stCaption { color:#b4c1d6; }
.hero {
    position:relative; overflow:hidden; padding:30px 34px;
    border:1px solid rgba(117,169,255,.25); border-radius:24px;
    background:radial-gradient(circle at 88% 20%,rgba(66,217,245,.22),transparent 25%),
      linear-gradient(120deg,rgba(35,56,103,.96),rgba(18,33,62,.96) 55%,rgba(15,54,70,.94));
    box-shadow:0 18px 55px rgba(0,0,0,.24); margin-bottom:22px;
}
.hero-kicker { color:#7ee7f7; font-size:.76rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; margin-bottom:10px; }
.hero-title { font-family:'Space Grotesk',sans-serif; color:#fff; font-size:clamp(1.8rem,4vw,2.75rem); line-height:1.12; font-weight:700; letter-spacing:-.045em; margin:0 0 10px; }
.hero-subtitle { color:#c4d5ee; font-size:1rem; max-width:760px; line-height:1.6; margin-bottom:18px; }
.hero-tags { display:flex; flex-wrap:wrap; gap:8px; }
.hero-tag { padding:6px 11px; border:1px solid rgba(169,205,255,.25); background:rgba(255,255,255,.07); border-radius:999px; color:#e1f7ff; font-size:.76rem; font-weight:600; }
[data-testid="stMetric"] {
    background:linear-gradient(145deg,rgba(24,36,60,.96),rgba(17,27,47,.96));
    border:1px solid var(--border); border-radius:17px; padding:17px 19px;
    box-shadow:0 8px 24px rgba(0,0,0,.13); min-height:112px;
}
[data-testid="stMetricLabel"] p { color:#aebdd4 !important; font-size:.82rem !important; font-weight:600; }
[data-testid="stMetricValue"] { color:#f4f8ff !important; font-family:'Space Grotesk',sans-serif; font-weight:700; }
[data-testid="stTabs"] [role="tablist"] { gap:8px; border-bottom:1px solid var(--border); }
[data-testid="stTabs"] button[role="tab"] { border:1px solid transparent; border-radius:12px 12px 0 0; padding:10px 17px; color:#aebdd4; font-weight:700; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#7ee7f7; background:rgba(66,217,245,.08); border-color:rgba(66,217,245,.22); }
div[data-testid="stPlotlyChart"] { border:1px solid var(--border); border-radius:17px; padding:8px 8px 2px; background:rgba(17,27,47,.72); }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid var(--border); border-radius:14px; overflow:hidden; }
.section-eyebrow { color:#65dff2; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:2px; }
.stButton > button,.stDownloadButton > button {
    border-radius:11px; border:1px solid rgba(93,205,238,.35);
    background:linear-gradient(110deg,#2865a8,#167d91); color:white; font-weight:700;
}
.stButton > button:hover,.stDownloadButton > button:hover { border-color:#75e7f5; color:white; transform:translateY(-1px); }
hr { border-color:rgba(148,170,210,.16) !important; }
@media(max-width:700px) { .block-container{padding:1rem 1rem 2rem;} .hero{padding:23px 21px;} }

/* HIGH-CONTRAST OVERRIDES: readable alerts, controls and sidebar */
.stApp, [data-testid="stAppViewContainer"] {
    color: #18243a !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stCaptionContainer"] {
    color: #d8e2f1;
}

/* Streamlit status/alert boxes were pale with pale text in the deployed app. */
[data-testid="stAlert"],
[data-testid="stAlert"] > div,
div[data-baseweb="notification"] {
    background: #172640 !important;
    border: 1px solid #385274 !important;
    border-radius: 13px !important;
    color: #edf4ff !important;
}
[data-testid="stAlert"] *,
div[data-baseweb="notification"] * {
    color: #edf4ff !important;
}
[data-testid="stAlert"] svg { fill: #78d9ed !important; }

/* Sidebar status panels */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] > div {
    background: #183b39 !important;
    border: 1px solid #287c70 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #c9fff0 !important;
}

/* Dark, readable form controls */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #172640 !important;
    color: #f3f7ff !important;
    border: 1px solid #49617f !important;
    border-radius: 10px !important;
}
.stSelectbox div[data-baseweb="select"] *,
.stMultiSelect div[data-baseweb="select"] *,
[data-baseweb="popover"] *,
[data-baseweb="menu"] * {
    color: #edf4ff !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #172640 !important;
    border: 1px solid #49617f !important;
}
.stSlider [data-testid="stTickBar"] { background: #7890ad !important; }
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #dce7f6 !important;
}

/* Keep body copy readable on the dark canvas */
.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: #d5dfef;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #f5f8ff !important;
}
.stApp [data-testid="stMetricLabel"] p { color: #bdcbe0 !important; }
.stApp [data-testid="stMetricValue"] { color: #ffffff !important; }

/* Dataframes and tables */
[data-testid="stDataFrame"] *, [data-testid="stTable"] * {
    color: #e7eef9;
}
[data-testid="stDataFrame"] {
    background: #121e33;
}

/* Progress bars and separators */
.stProgress > div > div { background: #35c4d8 !important; }
.stProgress > div { background: #293c58 !important; }


/* FINAL VISIBILITY FIX: metric change badges and select controls */

/* Scale-up / scale-down metric deltas: strong contrast for text and arrows */
[data-testid="stMetricDelta"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    width: fit-content !important;
    padding: 5px 11px !important;
    margin-top: 7px !important;
    border-radius: 999px !important;
    background: #263b59 !important;
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] *,
[data-testid="stMetricDelta"] span,
[data-testid="stMetricDelta"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stMetricDelta"] svg {
    width: 18px !important;
    height: 18px !important;
}

/* Select boxes: selected value, dropdown chevron and disabled-looking states */
.stSelectbox [data-baseweb="select"] > div {
    background: #172640 !important;
    border: 1px solid #6683a8 !important;
    min-height: 48px !important;
}
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}
.stSelectbox [data-baseweb="select"] svg {
    fill: #dcecff !important;
    color: #dcecff !important;
    opacity: 1 !important;
}
.stSelectbox [data-baseweb="select"] [role="combobox"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #f4f7ff !important;
    background: #172640 !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: #294568 !important;
}

/* Sidebar dropdowns receive the same high-contrast treatment */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #172640 !important;
    border-color: #6683a8 !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] *,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {
    color: #ffffff !important;
    fill: #dcecff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}


/* FINAL READABILITY FIX: dark text inside light control and delta boxes */

/* Sidebar select boxes: dark selected numbers and visible chevrons */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #f4f6fa !important;
    border: 1px solid #b9c7d9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #17243a !important;
    -webkit-text-fill-color: #17243a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #263b57 !important;
    color: #263b57 !important;
    stroke: #263b57 !important;
}

/* Dropdown option menu: dark readable text on a light menu */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] {
    background: #f4f6fa !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] * {
    color: #17243a !important;
    -webkit-text-fill-color: #17243a !important;
}

/* CPU and memory SCALE UP / SCALE DOWN metric delta pills */
[data-testid="stMetricDelta"] {
    color: #17243a !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
[data-testid="stMetricDelta"] *,
[data-testid="stMetricDelta"] span,
[data-testid="stMetricDelta"] p {
    color: #17243a !important;
    -webkit-text-fill-color: #17243a !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] svg {
    fill: #17243a !important;
    color: #17243a !important;
    stroke: #17243a !important;
    opacity: 1 !important;
}

/* Keep the action labels and large SCALE UP/DOWN values bright on dark cards */
[data-testid="stMetricLabel"] p {
    color: #d5deec !important;
}
[data-testid="stMetricValue"] {
    color: #f7f9ff !important;
}

/* Make selectbox labels crisp and easy to read */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #e7edf7 !important;
    -webkit-text-fill-color: #e7edf7 !important;
}

</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#42d9f5", "#8b9cff", "#36d6a0", "#ffbd69", "#c18cff"]

st.markdown("""
<div class="hero">
    <div class="hero-kicker">Cloud intelligence • predictive analytics</div>
    <div class="hero-title">Intelligent Cloud<br>Resource Allocation</div>
    <div class="hero-subtitle">
        A machine-learning powered control room for workload forecasting,
        adaptive resource recommendations and CloudSim performance analysis.
    </div>
    <div class="hero-tags">
        <span class="hero-tag">XGBoost forecasting</span>
        <span class="hero-tag">Predictive scaling</span>
        <span class="hero-tag">Resource optimization</span>
        <span class="hero-tag">CloudSim evaluation</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD PROJECT DATA
# ==========================================================

@st.cache_data
def load_data():

    cloudsim_df = pd.read_csv(CLOUDSIM_PATH)

    optimized_df = pd.read_csv(OPTIMIZED_PATH)

    test_features_df = pd.read_csv(TEST_FEATURES_PATH)

    return cloudsim_df, optimized_df, test_features_df


if not os.path.exists(CLOUDSIM_PATH):

    st.error(
        f"CloudSim file not found: {CLOUDSIM_PATH}"
    )

    st.stop()


if not os.path.exists(OPTIMIZED_PATH):

    st.error(
        f"Optimized file not found: {OPTIMIZED_PATH}"
    )

    st.stop()


if not os.path.exists(TEST_FEATURES_PATH):

    st.error(
        f"Test feature CSV not found: {TEST_FEATURES_PATH}. "
        "Upload test_features_for_dashboard.csv into the data/ folder."
    )

    st.stop()


try:

    cloudsim_df, optimized_df, test_features_df = load_data()

except Exception as e:

    st.error(f"Error loading project data: {e}")

    st.stop()


# ==========================================================
# 4. LOAD TRAINED XGBOOST MODELS
# ==========================================================

@st.cache_resource
def load_ml_models():

    cpu_model = joblib.load(CPU_MODEL_PATH)

    memory_model = joblib.load(MEMORY_MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:

        feature_info = json.load(f)

    cpu_features = feature_info["cpu_features"]

    memory_features = feature_info["memory_features"]

    return (
        cpu_model,
        memory_model,
        cpu_features,
        memory_features
    )


model_files = [
    CPU_MODEL_PATH,
    MEMORY_MODEL_PATH,
    FEATURE_PATH
]

missing_model_files = [
    path for path in model_files
    if not os.path.exists(path)
]


if missing_model_files:

    st.error(
        "Some trained model files are missing from models/."
    )

    st.write(missing_model_files)

    st.stop()


try:

    (
        cpu_model,
        memory_model,
        cpu_features,
        memory_features
    ) = load_ml_models()

    models_loaded = True

except Exception as e:

    st.error(f"Error loading XGBoost models: {e}")

    st.stop()


# ==========================================================
# 5. BOOLEAN CONVERSION
# ==========================================================

def convert_boolean(series):

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
        .fillna(False)
    )


for col in [
    "cpu_sla_violation",
    "memory_sla_violation",
    "cloudlet_finished",
    "vm_created"
]:

    if col in cloudsim_df.columns:

        cloudsim_df[col] = convert_boolean(
            cloudsim_df[col]
        )


# ==========================================================
# 6. DATA VALIDATION
# ==========================================================

cloudsim_required = [
    "interval",
    "policy",
    "allocated_cpu_cores",
    "allocated_memory_gb",
    "actual_cpu_cores",
    "actual_memory_gb",
    "cpu_sla_violation",
    "memory_sla_violation"
]


optimized_required = [
    "interval",
    "actual_required_cpu",
    "optimized_cpu_allocation",
    "actual_required_memory",
    "optimized_memory_allocation",
    "cpu_wastage",
    "memory_wastage"
]


missing_cloudsim = [
    col for col in cloudsim_required
    if col not in cloudsim_df.columns
]


missing_optimized = [
    col for col in optimized_required
    if col not in optimized_df.columns
]


if missing_cloudsim or missing_optimized:

    st.error("Some required project data columns are missing.")

    st.write(
        "CloudSim missing columns:",
        missing_cloudsim
    )

    st.write(
        "Optimized missing columns:",
        missing_optimized
    )

    st.stop()


st.success(
    f"Project data loaded successfully: "
    f"{len(cloudsim_df)} CloudSim records and "
    f"{len(optimized_df)} optimized workload records."
)


# ==========================================================
# 7. MODEL STATUS AND FEATURE VALIDATION
# ==========================================================

st.sidebar.header("ML Model Status")

st.sidebar.success("CPU XGBoost Model Loaded")

st.sidebar.success("Memory XGBoost Model Loaded")

st.sidebar.write(
    f"CPU input features: {len(cpu_features)}"
)

st.sidebar.write(
    f"Memory input features: {len(memory_features)}"
)


feature_compatible = (
    len(cpu_features) == 25
    and len(memory_features) == 25
    and cpu_features == memory_features
)


if not feature_compatible:

    st.error(
        "Model feature configuration is incompatible. "
        "Both models must use the same 25 features in "
        "the same order."
    )

    st.stop()


# Check the dedicated test-feature CSV, not the optimization results.
# The optimization CSV contains allocation/results columns, while the
# model requires the original 25 XGBoost input features.

missing_features = [
    feature for feature in cpu_features
    if feature not in test_features_df.columns
]


# ==========================================================
# 8. GENERATE XGBOOST PREDICTIONS
# ==========================================================

prediction_df = None

prediction_available = False


if missing_features:

    st.warning(
        "The current optimized dataset does not contain "
        "all 25 original XGBoost input features."
    )

    st.write(
        "Missing model features:",
        missing_features
    )

    st.info(
        "The prediction section will remain unavailable "
        "until the original workload feature dataset is "
        "connected. Existing CloudSim and optimization "
        "results can still be viewed."
    )

else:

    try:

        prediction_df = test_features_df.sort_values(
            "interval"
        ).reset_index(drop=True).copy()

        # Confirm the feature CSV and optimization results refer to
        # the same intervals in the same order.
        optimized_intervals = (
            optimized_df["interval"].sort_values().reset_index(drop=True)
        )
        feature_intervals = prediction_df["interval"].reset_index(drop=True)

        if not feature_intervals.equals(optimized_intervals):
            st.error(
                "Interval mismatch: test_features_for_dashboard.csv "
                "and final_combined_results.csv must contain the same "
                "intervals in the same sorted order."
            )
            st.stop()

        if len(prediction_df) != 447:
            st.error(
                f"Expected 447 test intervals, found {len(prediction_df)}."
            )
            st.stop()

        # Preserve the exact training feature order

        X = prediction_df[cpu_features].copy()

        # Convert input features to numeric

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Check for invalid values

        if X.isnull().any().any():

            invalid_features = (
                X.isnull().sum()
            )

            invalid_features = invalid_features[
                invalid_features > 0
            ]

            st.warning(
                "Missing or non-numeric values were found "
                "in the model input data."
            )

            st.write(invalid_features)

        else:

            # CPU prediction

            prediction_df["predicted_cpu_util"] = (
                cpu_model.predict(X)
            )

            # Memory prediction

            prediction_df["predicted_memory_util"] = (
                memory_model.predict(X)
            )

            prediction_available = True

            st.success(
                "Both XGBoost models generated predictions successfully."
            )

    except Exception as e:

        st.error(
            f"Prediction generation failed: {e}"
        )


# ==========================================================
# 9. SIDEBAR - INTERACTIVE PREDICTIVE SCALING
# ==========================================================

st.sidebar.divider()

st.sidebar.header("Predictive Scaling Controls")

CPU_THRESHOLD = 80
MEMORY_THRESHOLD = 80

CPU_SCALE_DOWN_THRESHOLD = 50
MEMORY_SCALE_DOWN_THRESHOLD = 50

if prediction_available:

    available_intervals = (
        prediction_df["interval"].tolist()
    )

    selected_interval = st.sidebar.selectbox(
        "Select workload interval",
        available_intervals
    )

    current_cores = st.sidebar.selectbox(
        "Currently allocated CPU cores",
        [2, 4, 6, 8, 10, 12, 14, 16],
        index=3
    )

    current_memory = st.sidebar.selectbox(
        "Currently allocated memory (GB)",
        [4, 8, 12, 16, 20, 24, 28, 32],
        index=3
    )

else:

    selected_interval = None

    current_cores = 8

    current_memory = 16


# ==========================================================
# 10. PREDICTIVE SCALING DECISION
# ==========================================================

st.markdown('<div class="section-eyebrow">Live model inference</div>', unsafe_allow_html=True)
st.header("Predictive Scaling Centre")

if prediction_available:

    current_position = prediction_df.index[
        prediction_df["interval"] == selected_interval
    ][0]

    selected_row = prediction_df.iloc[
        current_position
    ]

    predicted_cpu = float(
        selected_row["predicted_cpu_util"]
    )

    predicted_memory = float(
        selected_row["predicted_memory_util"]
    )

    previous_cpu = None

    previous_memory = None

    # Get previous interval predictions

    if current_position > 0:

        previous_row = prediction_df.iloc[
            current_position - 1
        ]

        previous_cpu = float(
            previous_row["predicted_cpu_util"]
        )

        previous_memory = float(
            previous_row["predicted_memory_util"]
        )

    # CPU scale-up validation

    cpu_high = (
        predicted_cpu >= CPU_THRESHOLD
        and previous_cpu is not None
        and previous_cpu >= CPU_THRESHOLD
    )

    # Memory scale-up validation

    memory_high = (
        predicted_memory >= MEMORY_THRESHOLD
        and previous_memory is not None
        and previous_memory >= MEMORY_THRESHOLD
    )

    # CPU scale-down validation

    cpu_low = (
        predicted_cpu < CPU_SCALE_DOWN_THRESHOLD
        and previous_cpu is not None
        and previous_cpu < CPU_SCALE_DOWN_THRESHOLD
    )

    # Memory scale-down validation

    memory_low = (
        predicted_memory < MEMORY_SCALE_DOWN_THRESHOLD
        and previous_memory is not None
        and previous_memory < MEMORY_SCALE_DOWN_THRESHOLD
    )

    # Independent CPU allocation recommendation and action

    if cpu_high:
        recommended_cores = min(current_cores + 2, 16)
        cpu_action = "SCALE UP" if recommended_cores > current_cores else "AT MAXIMUM"
    elif cpu_low:
        recommended_cores = max(current_cores - 2, 2)
        cpu_action = "SCALE DOWN" if recommended_cores < current_cores else "AT MINIMUM"
    else:
        recommended_cores = current_cores
        cpu_action = "MAINTAIN"

    # Independent memory allocation recommendation and action

    if memory_high:
        recommended_memory = min(current_memory + 4, 32)
        memory_action = (
            "SCALE UP" if recommended_memory > current_memory
            else "HIGH - AT MAXIMUM"
        )
    elif memory_low:
        recommended_memory = max(current_memory - 4, 4)
        memory_action = (
            "SCALE DOWN" if recommended_memory < current_memory
            else "AT MINIMUM"
        )
    else:
        recommended_memory = current_memory
        memory_action = "MAINTAIN"

    # Overall action describes mixed decisions instead of hiding them.
    if cpu_high or memory_high:
        if cpu_low or memory_low:
            action = "MIXED SCALING"
        else:
            action = "SCALE UP"
    elif cpu_low and memory_low:
        action = "SCALE DOWN"
    else:
        action = "MAINTAIN"

    # ------------------------------------------------------
    # DISPLAY PREDICTIONS
    # ------------------------------------------------------

    st.info(
        f"Predictions for workload interval: "
        f"{selected_interval}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted CPU Utilization",
        f"{predicted_cpu:.2f}%"
    )

    c2.metric(
        "Predicted Memory Utilization",
        f"{predicted_memory:.2f}%"
    )

    c3.metric(
        "Overall Predictive Scaling",
        action
    )

    # Show the independent decisions so a high memory forecast
    # does not hide a valid CPU scale-down recommendation.
    st.subheader("Independent Resource Scaling Decisions")
    action_col1, action_col2 = st.columns(2)

    action_col1.metric(
        "CPU Scaling Action",
        cpu_action,
        delta=f"{recommended_cores - current_cores:+d} cores"
    )

    action_col2.metric(
        "Memory Scaling Action",
        memory_action,
        delta=f"{recommended_memory - current_memory:+d} GB"
    )

    if memory_high and recommended_memory == current_memory:
        st.warning(
            "Predicted memory utilization is above the scale-up threshold, "
            "but the selected memory allocation is already at the dashboard "
            "limit of 32 GB. The dashboard cannot recommend more memory "
            "without changing the configured VM capacity limit."
        )

    st.divider()

    # ------------------------------------------------------
    # PREVIOUS INTERVAL VALIDATION
    # ------------------------------------------------------

    st.subheader(
        "Consecutive Interval Validation"
    )

    if previous_cpu is not None:

        p1, p2 = st.columns(2)

        p1.metric(
            "Previous CPU Prediction",
            f"{previous_cpu:.2f}%"
        )

        p2.metric(
            "Previous Memory Prediction",
            f"{previous_memory:.2f}%"
        )

        st.caption(
            "Scale-up is checked at 80% and scale-down at 50%. "
            "Each resource must satisfy its threshold in both the "
            "current and immediately previous interval."
        )

    else:

        st.warning(
            "First workload interval: previous interval "
            "predictions are unavailable."
        )

    st.divider()

    # ------------------------------------------------------
    # RESOURCE ALLOCATION
    # ------------------------------------------------------

    st.subheader(
        "Recommended Resource Allocation"
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Current CPU",
        f"{current_cores} cores"
    )

    r2.metric(
        "Recommended CPU",
        f"{recommended_cores} cores"
    )

    r3.metric(
        "Current Memory",
        f"{current_memory} GB"
    )

    r4.metric(
        "Recommended Memory",
        f"{recommended_memory} GB"
    )

    # ------------------------------------------------------
    # UTILIZATION PROGRESS
    # ------------------------------------------------------

    st.subheader(
        "Predicted Resource Utilization"
    )

    st.progress(
        max(0.0, min(1.0, predicted_cpu / 100)),
        text="Predicted CPU utilization"
    )

    st.progress(
        max(0.0, min(1.0, predicted_memory / 100)),
        text="Predicted Memory utilization"
    )

    st.divider()

    # ------------------------------------------------------
    # PREDICTION HISTORY
    # ------------------------------------------------------

    st.subheader(
        "XGBoost Prediction History"
    )

    prediction_history = prediction_df[
        [
            "interval",
            "predicted_cpu_util",
            "predicted_memory_util"
        ]
    ]

    history_plot = prediction_history.melt(
        id_vars="interval",
        var_name="Resource",
        value_name="Predicted Utilization (%)"
    )

    fig_predictions = px.line(
        history_plot,
        x="interval",
        y="Predicted Utilization (%)",
        color="Resource",
        title="CPU and Memory Predicted Utilization"
    )

    st.plotly_chart(
        fig_predictions,
        use_container_width=True
    )

else:

    st.warning(
        "XGBoost predictions are currently unavailable. "
        "Please connect the original 25-feature workload dataset."
    )

    st.write(
        "The trained models are loaded, but prediction "
        "requires the original model input features."
    )


st.caption(
    "The interactive predictive scaling section uses "
    "trained XGBoost predictions. It recommends resource "
    "allocations but does not provision actual cloud resources "
    "or rerun CloudSim."
)


st.divider()


# ==========================================================
# 11. PROJECT OVERVIEW
# ==========================================================

st.markdown('<div class="section-eyebrow">Performance at a glance</div>', unsafe_allow_html=True)
st.header("Project Results Overview")

total_intervals = optimized_df[
    "interval"
].nunique()

total_policies = cloudsim_df[
    "policy"
].nunique()


cpu_wastage = pd.to_numeric(
    optimized_df["cpu_wastage"],
    errors="coerce"
).mean()


memory_wastage = pd.to_numeric(
    optimized_df["memory_wastage"],
    errors="coerce"
).mean()


cpu_sla_count = cloudsim_df[
    "cpu_sla_violation"
].sum()


memory_sla_count = cloudsim_df[
    "memory_sla_violation"
].sum()


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Workload Intervals",
    total_intervals
)

m2.metric(
    "CloudSim Policies",
    total_policies
)

m3.metric(
    "Average CPU Wastage",
    f"{cpu_wastage:.2f}"
)

m4.metric(
    "Average Memory Wastage",
    f"{memory_wastage:.2f} GB"
)


m5, m6 = st.columns(2)

m5.metric(
    "CloudSim CPU SLA Violations",
    int(cpu_sla_count)
)

m6.metric(
    "CloudSim Memory SLA Violations",
    int(memory_sla_count)
)


st.divider()


# ==========================================================
# 12. RESOURCE OPTIMIZATION TABS
# ==========================================================

st.markdown('<div class="section-eyebrow">Deep-dive analytics</div>', unsafe_allow_html=True)
st.header(
    "CPU and Memory Resource Optimization"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "CPU Optimization",
    "Memory Optimization",
    "CloudSim Comparison",
    "Interval Explorer"
])


# ==========================================================
# TAB 1 - CPU OPTIMIZATION
# ==========================================================

with tab1:

    st.subheader(
        "Actual CPU Requirement vs Optimized Allocation"
    )

    cpu_plot = optimized_df[
        [
            "interval",
            "actual_required_cpu",
            "optimized_cpu_allocation"
        ]
    ].melt(
        id_vars="interval",
        var_name="Allocation Type",
        value_name="CPU Cores"
    )

    fig_cpu = px.line(
        cpu_plot,
        x="interval",
        y="CPU Cores",
        color="Allocation Type",
        title="CPU Requirement vs Optimized CPU Allocation"
    )

    st.plotly_chart(
        fig_cpu,
        use_container_width=True
    )

    cpu_waste_plot = px.line(
        optimized_df,
        x="interval",
        y="cpu_wastage",
        title="CPU Resource Wastage Across Intervals"
    )

    st.plotly_chart(
        cpu_waste_plot,
        use_container_width=True
    )

    st.metric(
        "Average CPU Resource Wastage",
        f"{cpu_wastage:.2f} cores"
    )


# ==========================================================
# TAB 2 - MEMORY OPTIMIZATION
# ==========================================================

with tab2:

    st.subheader(
        "Actual Memory Requirement vs Optimized Allocation"
    )

    memory_plot = optimized_df[
        [
            "interval",
            "actual_required_memory",
            "optimized_memory_allocation"
        ]
    ].melt(
        id_vars="interval",
        var_name="Allocation Type",
        value_name="Memory (GB)"
    )

    fig_memory = px.line(
        memory_plot,
        x="interval",
        y="Memory (GB)",
        color="Allocation Type",
        title="Memory Requirement vs Optimized Allocation"
    )

    st.plotly_chart(
        fig_memory,
        use_container_width=True
    )

    memory_waste_plot = px.line(
        optimized_df,
        x="interval",
        y="memory_wastage",
        title="Memory Resource Wastage Across Intervals"
    )

    st.plotly_chart(
        memory_waste_plot,
        use_container_width=True
    )

    st.metric(
        "Average Memory Resource Wastage",
        f"{memory_wastage:.2f} GB"
    )


# ==========================================================
# TAB 3 - CLOUDSIM POLICY COMPARISON
# ==========================================================

with tab3:

    st.subheader(
        "CloudSim Policy Comparison"
    )

    # Build summary using available columns

    aggregation = {
        "intervals": (
            "interval", "nunique"
        ),

        "cpu_sla_violations": (
            "cpu_sla_violation", "sum"
        ),

        "memory_sla_violations": (
            "memory_sla_violation", "sum"
        )
    }

    if "execution_time_seconds" in cloudsim_df.columns:

        aggregation["average_execution_time"] = (
            "execution_time_seconds", "mean"
        )

    if "cloudlet_finished" in cloudsim_df.columns:

        aggregation["completed_cloudlets"] = (
            "cloudlet_finished", "sum"
        )

    if "vm_created" in cloudsim_df.columns:

        aggregation["vm_creation_count"] = (
            "vm_created", "sum"
        )

    policy_summary = cloudsim_df.groupby(
        "policy"
    ).agg(
        **aggregation
    ).reset_index()


    st.dataframe(
        policy_summary,
        use_container_width=True
    )

    # SLA comparison

    fig_sla = px.bar(
        policy_summary.melt(
            id_vars="policy",
            value_vars=[
                "cpu_sla_violations",
                "memory_sla_violations"
            ],
            var_name="SLA Type",
            value_name="Violations"
        ),
        x="policy",
        y="Violations",
        color="SLA Type",
        barmode="group",
        title="SLA Violation Comparison"
    )

    st.plotly_chart(
        fig_sla,
        use_container_width=True
    )

    # Execution time comparison

    if "average_execution_time" in policy_summary.columns:

        fig_execution = px.bar(
            policy_summary,
            x="policy",
            y="average_execution_time",
            title="Average Cloudlet Execution Time by Policy"
        )

        st.plotly_chart(
            fig_execution,
            use_container_width=True
        )

    # CPU allocation comparison

    allocation_plot = cloudsim_df.groupby(
        ["interval", "policy"]
    )[
        "allocated_cpu_cores"
    ].mean().reset_index()

    fig_allocation = px.line(
        allocation_plot,
        x="interval",
        y="allocated_cpu_cores",
        color="policy",
        title="CPU Allocation Comparison Across Policies"
    )

    st.plotly_chart(
        fig_allocation,
        use_container_width=True
    )

    # Memory allocation comparison

    memory_allocation_plot = cloudsim_df.groupby(
        ["interval", "policy"]
    )[
        "allocated_memory_gb"
    ].mean().reset_index()

    fig_memory_allocation = px.line(
        memory_allocation_plot,
        x="interval",
        y="allocated_memory_gb",
        color="policy",
        title="Memory Allocation Comparison Across Policies"
    )

    st.plotly_chart(
        fig_memory_allocation,
        use_container_width=True
    )


# ==========================================================
# TAB 4 - INTERVAL EXPLORER
# ==========================================================

with tab4:

    st.subheader(
        "Explore Workload Intervals"
    )

    min_interval = int(
        optimized_df["interval"].min()
    )

    max_interval = int(
        optimized_df["interval"].max()
    )

    selected_explorer_interval = st.slider(
        "Select workload interval",
        min_interval,
        max_interval,
        min_interval
    )

    selected_optimized = optimized_df[
        optimized_df["interval"]
        == selected_explorer_interval
    ]

    selected_cloudsim = cloudsim_df[
        cloudsim_df["interval"]
        == selected_explorer_interval
    ]

    st.write(
        "### Optimized Allocation"
    )

    st.dataframe(
        selected_optimized,
        use_container_width=True
    )

    st.write(
        "### CloudSim Policy Results"
    )

    st.dataframe(
        selected_cloudsim,
        use_container_width=True
    )

    # Show ML predictions for selected interval

    if prediction_available:

        selected_prediction = prediction_df[
            prediction_df["interval"]
            == selected_explorer_interval
        ]

        if not selected_prediction.empty:

            st.write(
                "### XGBoost Predictions"
            )

            st.dataframe(
                selected_prediction[
                    [
                        "interval",
                        "predicted_cpu_util",
                        "predicted_memory_util"
                    ]
                ],
                use_container_width=True
            )


# ==========================================================
# 13. DOWNLOAD PROJECT RESULTS
# ==========================================================

st.divider()

st.header(
    "Download Project Results"
)

d1, d2 = st.columns(2)


with d1:

    st.download_button(
        "Download CloudSim Results",
        data=cloudsim_df.to_csv(index=False),
        file_name="cloudsim_results.csv",
        mime="text/csv"
    )


with d2:

    st.download_button(
        "Download Optimized Results",
        data=optimized_df.to_csv(index=False),
        file_name="optimized_results.csv",
        mime="text/csv"
    )


# Download predictions when available

if prediction_available:

    st.download_button(
        "Download XGBoost Predictions",
        data=prediction_df.to_csv(index=False),
        file_name="xgboost_predictions.csv",
        mime="text/csv"
    )


# ==========================================================
# 14. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Intelligent Cloud Resource Allocation | "
    "Machine Learning and CloudSim Based Analysis | "
    "CPU and Memory XGBoost Predictive Scaling"
)
