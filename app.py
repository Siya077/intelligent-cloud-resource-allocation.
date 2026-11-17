
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
# 9. SIDEBAR - AUTOMATIC PREDICTIVE SCALING
# ==========================================================

st.sidebar.divider()
st.sidebar.header("Automatic Predictive Scaling")

if prediction_available:

    available_intervals = prediction_df["interval"].tolist()

    selected_interval = st.sidebar.selectbox(
        "Select workload interval",
        available_intervals
    )

    st.sidebar.caption(
        "The allocation engine replays workload intervals in order. "
        "Each interval starts from the previous interval's recommended "
        "allocation; it does not reset to the initial allocation."
    )

    st.sidebar.subheader("Initial VM Allocation")

    initial_cores = st.sidebar.selectbox(
        "Initial CPU allocation (cores)",
        [2, 4, 6, 8, 10, 12, 14, 16],
        index=3
    )

    initial_memory = st.sidebar.selectbox(
        "Initial memory allocation (GB)",
        [4, 8, 12, 16, 20, 24, 28, 32],
        index=3
    )

    st.sidebar.divider()
    st.sidebar.subheader("Automatic Scaling Policy")

    st.sidebar.caption(
        "These policy values are applied automatically to every interval."
    )

    CPU_THRESHOLD = st.sidebar.slider(
        "CPU scale-up threshold (%)",
        min_value=60,
        max_value=95,
        value=80,
        step=1
    )

    MEMORY_THRESHOLD = st.sidebar.slider(
        "Memory scale-up threshold (%)",
        min_value=60,
        max_value=95,
        value=80,
        step=1
    )

    CPU_SCALE_DOWN_THRESHOLD = st.sidebar.slider(
        "CPU scale-down threshold (%)",
        min_value=10,
        max_value=min(70, CPU_THRESHOLD - 1),
        value=min(20, CPU_THRESHOLD - 1),
        step=1
    )

    MEMORY_SCALE_DOWN_THRESHOLD = st.sidebar.slider(
        "Memory scale-down threshold (%)",
        min_value=10,
        max_value=min(70, MEMORY_THRESHOLD - 1),
        value=min(50, MEMORY_THRESHOLD - 1),
        step=1
    )

    cpu_scale_step = st.sidebar.select_slider(
        "CPU scaling step (cores)",
        options=[1, 2, 4],
        value=2
    )

    memory_scale_step = st.sidebar.select_slider(
        "Memory scaling step (GB)",
        options=[2, 4, 8],
        value=4
    )

    scaling_cooldown = st.sidebar.select_slider(
        "Cooldown after a scaling action (intervals)",
        options=[1, 2, 3],
        value=2,
        help=(
            "Prevents the engine from changing the same resource "
            "again immediately in the next few intervals."
        )
    )

else:
    selected_interval = None
    initial_cores = 8
    initial_memory = 16
    CPU_THRESHOLD = 80
    MEMORY_THRESHOLD = 80
    CPU_SCALE_DOWN_THRESHOLD = 20
    MEMORY_SCALE_DOWN_THRESHOLD = 50
    cpu_scale_step = 2
    memory_scale_step = 4
    scaling_cooldown = 2


# ==========================================================
# 10. SEQUENTIAL AUTOMATIC RESOURCE ALLOCATION
# ==========================================================

st.header("XGBoost Predictive Scaling")

if prediction_available:

    def simulate_automatic_allocation(
        source_df,
        initial_cpu,
        initial_mem,
        cpu_up_threshold,
        cpu_down_threshold,
        memory_up_threshold,
        memory_down_threshold,
        cpu_step,
        memory_step,
        cooldown_intervals
    ):
        """
        Replay predictions in chronological order.

        CPU and memory are controlled independently. Each resource:
        - Requires two consecutive predictions to cross a threshold.
        - Carries its allocation forward from the preceding interval.
        - Enforces minimum/maximum allocation limits.
        - Waits for its own cooldown after a scaling action.
        """

        result = source_df.sort_values("interval").reset_index(drop=True).copy()

        cpu_allocation = int(initial_cpu)
        memory_allocation = int(initial_mem)

        cpu_cooldown_remaining = 0
        memory_cooldown_remaining = 0

        cpu_allocations = []
        memory_allocations = []
        cpu_actions = []
        memory_actions = []
        overall_actions = []
        cpu_cooldowns = []
        memory_cooldowns = []

        for position, row in result.iterrows():

            cpu_now = float(row["predicted_cpu_util"])
            memory_now = float(row["predicted_memory_util"])

            if position > 0:
                cpu_previous = float(
                    result.iloc[position - 1]["predicted_cpu_util"]
                )
                memory_previous = float(
                    result.iloc[position - 1]["predicted_memory_util"]
                )
            else:
                cpu_previous = None
                memory_previous = None

            # ---------------- CPU POLICY ----------------

            cpu_action = "MAINTAIN"

            cpu_high = (
                cpu_previous is not None
                and cpu_previous >= cpu_up_threshold
                and cpu_now >= cpu_up_threshold
            )

            cpu_low = (
                cpu_previous is not None
                and cpu_previous < cpu_down_threshold
                and cpu_now < cpu_down_threshold
            )

            if cpu_cooldown_remaining > 0:
                cpu_action = "MAINTAIN (COOLDOWN)"
                cpu_cooldown_remaining -= 1

            elif cpu_high:
                new_cpu = min(cpu_allocation + cpu_step, 16)

                if new_cpu > cpu_allocation:
                    cpu_allocation = new_cpu
                    cpu_action = "SCALE UP"
                    cpu_cooldown_remaining = cooldown_intervals
                else:
                    cpu_action = "AT MAXIMUM"

            elif cpu_low:
                new_cpu = max(cpu_allocation - cpu_step, 2)

                if new_cpu < cpu_allocation:
                    cpu_allocation = new_cpu
                    cpu_action = "SCALE DOWN"
                    cpu_cooldown_remaining = cooldown_intervals
                else:
                    cpu_action = "AT MINIMUM"

            # ---------------- MEMORY POLICY ----------------

            memory_action = "MAINTAIN"

            memory_high = (
                memory_previous is not None
                and memory_previous >= memory_up_threshold
                and memory_now >= memory_up_threshold
            )

            memory_low = (
                memory_previous is not None
                and memory_previous < memory_down_threshold
                and memory_now < memory_down_threshold
            )

            if memory_cooldown_remaining > 0:
                memory_action = "MAINTAIN (COOLDOWN)"
                memory_cooldown_remaining -= 1

            elif memory_high:
                new_memory = min(memory_allocation + memory_step, 32)

                if new_memory > memory_allocation:
                    memory_allocation = new_memory
                    memory_action = "SCALE UP"
                    memory_cooldown_remaining = cooldown_intervals
                else:
                    memory_action = "AT MAXIMUM"

            elif memory_low:
                new_memory = max(memory_allocation - memory_step, 4)

                if new_memory < memory_allocation:
                    memory_allocation = new_memory
                    memory_action = "SCALE DOWN"
                    memory_cooldown_remaining = cooldown_intervals
                else:
                    memory_action = "AT MINIMUM"

            # ---------------- OVERALL POLICY ----------------

            if cpu_action == "SCALE UP" or memory_action == "SCALE UP":
                overall_action = "SCALE UP"
            elif cpu_action == "SCALE DOWN" or memory_action == "SCALE DOWN":
                overall_action = "SCALE DOWN"
            elif (
                "COOLDOWN" in cpu_action
                or "COOLDOWN" in memory_action
            ):
                overall_action = "MAINTAIN (COOLDOWN)"
            else:
                overall_action = "MAINTAIN"

            cpu_allocations.append(cpu_allocation)
            memory_allocations.append(memory_allocation)
            cpu_actions.append(cpu_action)
            memory_actions.append(memory_action)
            overall_actions.append(overall_action)
            cpu_cooldowns.append(cpu_cooldown_remaining)
            memory_cooldowns.append(memory_cooldown_remaining)

        result["cpu_scaling_action"] = cpu_actions
        result["memory_scaling_action"] = memory_actions
        result["overall_scaling_action"] = overall_actions
        result["automatic_cpu_allocation"] = cpu_allocations
        result["automatic_memory_allocation"] = memory_allocations
        result["cpu_cooldown_remaining"] = cpu_cooldowns
        result["memory_cooldown_remaining"] = memory_cooldowns

        return result

    allocation_history = simulate_automatic_allocation(
        source_df=prediction_df,
        initial_cpu=initial_cores,
        initial_mem=initial_memory,
        cpu_up_threshold=CPU_THRESHOLD,
        cpu_down_threshold=CPU_SCALE_DOWN_THRESHOLD,
        memory_up_threshold=MEMORY_THRESHOLD,
        memory_down_threshold=MEMORY_SCALE_DOWN_THRESHOLD,
        cpu_step=cpu_scale_step,
        memory_step=memory_scale_step,
        cooldown_intervals=scaling_cooldown
    )

    selected_position = allocation_history.index[
        allocation_history["interval"] == selected_interval
    ][0]

    selected_row = allocation_history.iloc[selected_position]

    predicted_cpu = float(selected_row["predicted_cpu_util"])
    predicted_memory = float(selected_row["predicted_memory_util"])

    previous_cpu = None
    previous_memory = None

    if selected_position > 0:
        previous_row = allocation_history.iloc[selected_position - 1]
        previous_cpu = float(previous_row["predicted_cpu_util"])
        previous_memory = float(previous_row["predicted_memory_util"])

    current_cores = (
        initial_cores
        if selected_position == 0
        else int(
            allocation_history.iloc[selected_position - 1][
                "automatic_cpu_allocation"
            ]
        )
    )

    current_memory = (
        initial_memory
        if selected_position == 0
        else int(
            allocation_history.iloc[selected_position - 1][
                "automatic_memory_allocation"
            ]
        )
    )

    recommended_cores = int(selected_row["automatic_cpu_allocation"])
    recommended_memory = int(selected_row["automatic_memory_allocation"])

    cpu_action = selected_row["cpu_scaling_action"]
    memory_action = selected_row["memory_scaling_action"]
    action = selected_row["overall_scaling_action"]

    st.info(f"Predictions for workload interval: {selected_interval}")

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

    st.caption(
        "CPU and memory are evaluated independently. A high memory "
        "prediction does not force CPU to scale up, and a low CPU "
        "prediction does not force memory to scale down."
    )

    st.divider()

    # ------------------------------------------------------
    # CONSECUTIVE INTERVAL VALIDATION
    # ------------------------------------------------------

    st.subheader("Consecutive Interval Validation")

    if previous_cpu is not None:

        p1, p2 = st.columns(2)

        p1.metric("Previous CPU Prediction", f"{previous_cpu:.2f}%")
        p2.metric("Previous Memory Prediction", f"{previous_memory:.2f}%")

        st.caption(
            f"CPU: scale up at {CPU_THRESHOLD}% or above; scale down "
            f"below {CPU_SCALE_DOWN_THRESHOLD}%. "
            f"Memory: scale up at {MEMORY_THRESHOLD}% or above; "
            f"scale down below {MEMORY_SCALE_DOWN_THRESHOLD}%. "
            "Both the current and immediately previous predictions "
            "must cross the relevant threshold."
        )

    else:
        st.warning(
            "This is the first workload interval, so previous interval "
            "predictions are unavailable. The allocation remains unchanged."
        )

    st.divider()

    # ------------------------------------------------------
    # AUTOMATIC RESOURCE ALLOCATION
    # ------------------------------------------------------

    st.subheader("Automatically Recommended Resource Allocation")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("Previous CPU Allocation", f"{current_cores} cores")
    r2.metric("Recommended CPU", f"{recommended_cores} cores")
    r3.metric("Previous Memory Allocation", f"{current_memory} GB")
    r4.metric("Recommended Memory", f"{recommended_memory} GB")

    st.caption(
        f"Automatic policy: CPU step {cpu_scale_step} core(s), "
        f"memory step {memory_scale_step} GB, and "
        f"{scaling_cooldown}-interval cooldown after each resource scaling action."
    )

    st.divider()

    # ------------------------------------------------------
    # UTILIZATION PROGRESS
    # ------------------------------------------------------

    st.subheader("Predicted Resource Utilization")

    st.progress(
        max(0.0, min(1.0, predicted_cpu / 100)),
        text=f"Predicted CPU utilization: {predicted_cpu:.2f}%"
    )

    st.progress(
        max(0.0, min(1.0, predicted_memory / 100)),
        text=f"Predicted memory utilization: {predicted_memory:.2f}%"
    )

    st.divider()

    # ------------------------------------------------------
    # AUTOMATIC ALLOCATION HISTORY
    # ------------------------------------------------------

    st.subheader("Sequential Automatic Allocation History")

    history_columns = [
        "interval",
        "predicted_cpu_util",
        "predicted_memory_util",
        "cpu_scaling_action",
        "memory_scaling_action",
        "automatic_cpu_allocation",
        "automatic_memory_allocation"
    ]

    st.dataframe(
        allocation_history.loc[
            allocation_history.index <= selected_position,
            history_columns
        ].tail(30),
        use_container_width=True,
        hide_index=True
    )

    history_plot = allocation_history[
        [
            "interval",
            "automatic_cpu_allocation",
            "automatic_memory_allocation"
        ]
    ].melt(
        id_vars="interval",
        var_name="Resource",
        value_name="Allocated Amount"
    )

    fig_allocation_history = px.line(
        history_plot,
        x="interval",
        y="Allocated Amount",
        color="Resource",
        title="Automatically Recommended CPU and Memory Allocation"
    )

    st.plotly_chart(
        fig_allocation_history,
        use_container_width=True
    )

    # Download the sequential simulation output.
    st.download_button(
        "Download Automatic Scaling History",
        data=allocation_history.to_csv(index=False),
        file_name="automatic_scaling_history.csv",
        mime="text/csv"
    )

    # ------------------------------------------------------
    # PREDICTION HISTORY
    # ------------------------------------------------------

    st.subheader("XGBoost Prediction History")

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
        "The trained models are loaded, but prediction requires "
        "the original model input features."
    )


st.caption(
    "The interactive predictive scaling section uses trained XGBoost "
    "predictions and a sequential allocation simulation. It recommends "
    "resource allocations but does not provision actual cloud resources "
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
