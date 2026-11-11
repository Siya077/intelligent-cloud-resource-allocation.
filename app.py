
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
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Intelligent Cloud Resource Allocation")

st.caption(
    "Machine Learning Based Predictive Scaling Dashboard"
)

st.info(
    "This dashboard combines trained XGBoost models, "
    "resource optimization analysis and CloudSim "
    "policy comparison."
)


# ==========================================================
# 3. LOAD PROJECT DATA
# ==========================================================

@st.cache_data
def load_data():

    cloudsim_df = pd.read_csv(CLOUDSIM_PATH)

    optimized_df = pd.read_csv(OPTIMIZED_PATH)

    return cloudsim_df, optimized_df


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


try:

    cloudsim_df, optimized_df = load_data()

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


# Check whether the optimized dataset contains all features

missing_features = [
    feature for feature in cpu_features
    if feature not in optimized_df.columns
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

        prediction_df = optimized_df.sort_values(
            "interval"
        ).reset_index(drop=True).copy()

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

st.header("XGBoost Predictive Scaling")

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

    # Independent CPU allocation recommendation

    if cpu_high:

        recommended_cores = min(
            current_cores + 2, 16
        )

    elif cpu_low:

        recommended_cores = max(
            current_cores - 2, 2
        )

    else:

        recommended_cores = current_cores

    # Independent memory allocation recommendation

    if memory_high:

        recommended_memory = min(
            current_memory + 4, 32
        )

    elif memory_low:

        recommended_memory = max(
            current_memory - 4, 4
        )

    else:

        recommended_memory = current_memory

    # Combined scaling action

    if cpu_high or memory_high:

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
        "Predictive Scaling Action",
        action
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
            "Scaling thresholds are checked across the "
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
