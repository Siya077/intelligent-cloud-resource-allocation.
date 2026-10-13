import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Intelligent Cloud Resource Allocation",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Intelligent Cloud Resource Allocation")
st.caption("Machine Learning Based Predictive Scaling Dashboard")

st.info(
    "This dashboard displays saved Week 8 replay results. "
    "The full 447-task CloudSim simulation is still pending validation."
)

st.sidebar.header("Workload Controls")

cpu = st.sidebar.slider("CPU utilization (%)", 0, 100, 72)
memory = st.sidebar.slider("Memory utilization (%)", 0, 100, 75)
current_cores = st.sidebar.selectbox(
    "Currently allocated CPU cores",
    [6, 8, 10, 12, 14, 16],
    index=2
)

cpu_threshold = 80
memory_threshold = 80

if cpu >= cpu_threshold or memory >= memory_threshold:
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
    "The sidebar is a demonstration of scaling rules. "
    "It does not run your trained XGBoost model or a live CloudSim simulation."
)

st.divider()
st.header("Week 8 Workload Analysis")

uploaded_file = st.file_uploader(
    "Upload week8_predictive_scaling_results.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required = [
        "actual_cpu",
        "actual_memory",
        "predicted_cpu",
        "predicted_memory"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        if "interval" not in df.columns:
            df["interval"] = range(1, len(df) + 1)

        st.success(f"Loaded {len(df)} workload intervals.")

        cpu_mae = (
            df["actual_cpu"] - df["predicted_cpu"]
        ).abs().mean()

        memory_mae = (
            df["actual_memory"] - df["predicted_memory"]
        ).abs().mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Intervals", len(df))
        m2.metric("CPU Prediction MAE", f"{cpu_mae:.2f}")
        m3.metric("Memory Prediction MAE", f"{memory_mae:.2f}")

        cpu_chart = df.melt(
            id_vars="interval",
            value_vars=["actual_cpu", "predicted_cpu"],
            var_name="Type",
            value_name="Utilization (%)"
        )

        st.plotly_chart(
            px.line(
                cpu_chart,
                x="interval",
                y="Utilization (%)",
                color="Type",
                title="Actual vs Predicted CPU Utilization"
            ),
            use_container_width=True
        )

        memory_chart = df.melt(
            id_vars="interval",
            value_vars=["actual_memory", "predicted_memory"],
            var_name="Type",
            value_name="Utilization (%)"
        )

        st.plotly_chart(
            px.line(
                memory_chart,
                x="interval",
                y="Utilization (%)",
                color="Type",
                title="Actual vs Predicted Memory Utilization"
            ),
            use_container_width=True
        )

        if {
            "static_allocated_cpu",
            "dynamic_allocated_cpu"
        }.issubset(df.columns):

            allocation_chart = df.melt(
                id_vars="interval",
                value_vars=[
                    "static_allocated_cpu",
                    "dynamic_allocated_cpu"
                ],
                var_name="Allocation Type",
                value_name="CPU Cores"
            )

            st.plotly_chart(
                px.line(
                    allocation_chart,
                    x="interval",
                    y="CPU Cores",
                    color="Allocation Type",
                    title="Static vs Dynamic CPU Allocation"
                ),
                use_container_width=True
            )

        if "scaling_action" in df.columns:
            counts = (
                df["scaling_action"]
                .value_counts()
                .rename_axis("Action")
                .reset_index(name="Count")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Action",
                    y="Count",
                    title="Scaling Action Distribution"
                ),
                use_container_width=True
            )

        st.subheader("Workload Records")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Results CSV",
            data=df.to_csv(index=False),
            file_name="dashboard_results.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Upload your Week 8 CSV above to display actual project results."
    )
