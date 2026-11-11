import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLOUDSIM_PATH = os.path.join(
    DATA_DIR, "week8_cloudsim_simulation_results.csv"
)

OPTIMIZED_PATH = os.path.join(
    DATA_DIR, "final_combined_results.csv"
)

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Intelligent Cloud Resource Allocation",
    page_icon="☁️",
    layout="wide"
)

st.title("Intelligent Cloud Resource Allocation")
st.caption("Machine Learning Based Predictive Scaling Dashboard")

st.info(
    "This dashboard displays saved workload optimization and "
    "CloudSim policy-comparison results. CloudSim evaluates "
    "each workload interval independently under three policies."
)

# ==========================================
# 3. LOAD PROJECT DATA
# ==========================================

@st.cache_data
def load_data():
    cloudsim_df = pd.read_csv(CLOUDSIM_PATH)
    optimized_df = pd.read_csv(OPTIMIZED_PATH)

    return cloudsim_df, optimized_df


if not os.path.exists(CLOUDSIM_PATH):
    st.error(f"CloudSim file not found: {CLOUDSIM_PATH}")
    st.stop()

if not os.path.exists(OPTIMIZED_PATH):
    st.error(f"Optimized file not found: {OPTIMIZED_PATH}")
    st.stop()

try:
    cloudsim_df, optimized_df = load_data()
except Exception as e:
    st.error(f"Error loading project data: {e}")
    st.stop()

# Convert boolean-like columns safely
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
        cloudsim_df[col] = convert_boolean(cloudsim_df[col])

# ==========================================
# 4. DATA VALIDATION
# ==========================================

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
    st.error("Some required columns are missing.")
    st.write("CloudSim missing columns:", missing_cloudsim)
    st.write("Optimized missing columns:", missing_optimized)
    st.stop()

st.success(
    f"Project data loaded successfully: "
    f"{len(cloudsim_df)} CloudSim records and "
    f"{len(optimized_df)} optimized workload records."
)

# ==========================================
# 5. SIDEBAR - INTERACTIVE SCALING DEMO
# ==========================================

st.sidebar.header("Interactive Workload Controls")

cpu = st.sidebar.slider(
    "CPU utilization (%)", 0, 100, 72
)

memory = st.sidebar.slider(
    "Memory utilization (%)", 0, 100, 75
)

current_cores = st.sidebar.selectbox(
    "Currently allocated CPU cores",
    [6, 8, 10, 12, 14, 16],
    index=2
)

CPU_THRESHOLD = 80
MEMORY_THRESHOLD = 80

if cpu >= CPU_THRESHOLD or memory >= MEMORY_THRESHOLD:
    action = "SCALE UP"
    recommended_cores = min(current_cores + 2, 16)

elif cpu < 40 and memory < 40:
    action = "SCALE DOWN"
    recommended_cores = max(current_cores - 2, 6)

else:
    action = "MAINTAIN"
    recommended_cores = current_cores

st.subheader("Interactive Scaling Demonstration")

c1, c2, c3 = st.columns(3)

c1.metric("CPU Utilization", f"{cpu}%")
c2.metric("Memory Utilization", f"{memory}%")
c3.metric("Recommended Action", action)

st.write(f"Current CPU allocation: **{current_cores} cores**")
st.write(f"Recommended CPU allocation: **{recommended_cores} cores**")

st.progress(cpu / 100, text="CPU utilization")
st.progress(memory / 100, text="Memory utilization")

st.warning(
    "This section demonstrates threshold-based scaling rules. "
    "It does not execute the trained XGBoost model or live CloudSim."
)

st.divider()

# ==========================================
# 6. PROJECT OVERVIEW
# ==========================================

st.header("Project Results Overview")

total_intervals = optimized_df["interval"].nunique()
total_policies = cloudsim_df["policy"].nunique()

cpu_wastage = pd.to_numeric(
    optimized_df["cpu_wastage"], errors="coerce"
).mean()

memory_wastage = pd.to_numeric(
    optimized_df["memory_wastage"], errors="coerce"
).mean()

cpu_sla_count = cloudsim_df[
    "cpu_sla_violation"
].sum()

memory_sla_count = cloudsim_df[
    "memory_sla_violation"
].sum()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Workload Intervals", total_intervals)
m2.metric("CloudSim Policies", total_policies)
m3.metric("Average CPU Wastage", f"{cpu_wastage:.2f}")
m4.metric("Average Memory Wastage", f"{memory_wastage:.2f} GB")

m5, m6 = st.columns(2)

m5.metric("CloudSim CPU SLA Violations", int(cpu_sla_count))
m6.metric("CloudSim Memory SLA Violations", int(memory_sla_count))

st.divider()

# ==========================================
# 7. RESOURCE OPTIMIZATION
# ==========================================

st.header("CPU and Memory Resource Optimization")

tab1, tab2, tab3, tab4 = st.tabs([
    "CPU Optimization",
    "Memory Optimization",
    "CloudSim Comparison",
    "Interval Explorer"
])

# ==========================================
# TAB 1 - CPU OPTIMIZATION
# ==========================================

with tab1:

    st.subheader("Actual CPU Requirement vs Optimized Allocation")

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

    st.plotly_chart(fig_cpu, use_container_width=True)

    cpu_waste_plot = px.line(
        optimized_df,
        x="interval",
        y="cpu_wastage",
        title="CPU Resource Wastage Across Intervals"
    )

    st.plotly_chart(cpu_waste_plot, use_container_width=True)

    st.metric(
        "Average CPU Resource Wastage",
        f"{cpu_wastage:.2f} cores"
    )

# ==========================================
# TAB 2 - MEMORY OPTIMIZATION
# ==========================================

with tab2:

    st.subheader("Actual Memory Requirement vs Optimized Allocation")

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

    st.plotly_chart(fig_memory, use_container_width=True)

    memory_waste_plot = px.line(
        optimized_df,
        x="interval",
        y="memory_wastage",
        title="Memory Resource Wastage Across Intervals"
    )

    st.plotly_chart(memory_waste_plot, use_container_width=True)

    st.metric(
        "Average Memory Resource Wastage",
        f"{memory_wastage:.2f} GB"
    )

# ==========================================
# TAB 3 - CLOUDSIM POLICY COMPARISON
# ==========================================

with tab3:

    st.subheader("CloudSim Policy Comparison")

    policy_summary = cloudsim_df.groupby(
        "policy"
    ).agg(
        intervals=("interval", "nunique"),
        average_execution_time=(
            "execution_time_seconds", "mean"
        ),
        completed_cloudlets=(
            "cloudlet_finished", "sum"
        ),
        vm_creation_count=("vm_created", "sum"),
        cpu_sla_violations=(
            "cpu_sla_violation", "sum"
        ),
        memory_sla_violations=(
            "memory_sla_violation", "sum"
        )
    ).reset_index()

    st.dataframe(
        policy_summary,
        use_container_width=True
    )

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

    st.plotly_chart(fig_sla, use_container_width=True)

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

    allocation_plot = cloudsim_df.groupby(
        ["interval", "policy"]
    )["allocated_cpu_cores"].mean().reset_index()

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

# ==========================================
# TAB 4 - INTERVAL EXPLORER
# ==========================================

with tab4:

    st.subheader("Explore Workload Intervals")

    min_interval = int(
        optimized_df["interval"].min()
    )

    max_interval = int(
        optimized_df["interval"].max()
    )

    selected_interval = st.slider(
        "Select workload interval",
        min_interval,
        max_interval,
        min_interval
    )

    selected_optimized = optimized_df[
        optimized_df["interval"] == selected_interval
    ]

    selected_cloudsim = cloudsim_df[
        cloudsim_df["interval"] == selected_interval
    ]

    st.write("### Optimized Allocation")

    st.dataframe(
        selected_optimized,
        use_container_width=True
    )

    st.write("### CloudSim Policy Results")

    st.dataframe(
        selected_cloudsim,
        use_container_width=True
    )

# ==========================================
# 8. DOWNLOAD RESULTS
# ==========================================

st.divider()

st.header("Download Project Results")

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

st.caption(
    "Intelligent Cloud Resource Allocation | "
    "Machine Learning and CloudSim Based Analysis"
)
