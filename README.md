
# Intelligent Cloud Resource Allocation Using Machine Learning and Hyperparameter Optimization

## Project Overview

This project focuses on optimizing cloud resource allocation using Machine Learning and predictive scaling techniques.

It uses workload utilization data to predict CPU and memory requirements and dynamically allocate cloud resources. XGBoost is used for prediction, while Optuna is used for hyperparameter optimization. CloudSim is used to evaluate resource allocation policies.

## Objectives

- Predict future CPU and memory utilization using Machine Learning.
- Optimize cloud resource allocation using predictive scaling.
- Reduce unnecessary resource wastage while maintaining SLA requirements.
- Compare static, existing dynamic, and ML-based optimized allocation policies.

## Technologies Used

- Python
- Machine Learning
- XGBoost
- Optuna
- Pandas and NumPy
- Matplotlib and Plotly
- Google Colab
- CloudSim
- Streamlit
- GitHub

## System Workflow

1. Workload dataset collection and preprocessing.
2. CPU and memory utilization prediction.
3. Hyperparameter optimization using Optuna.
4. Dynamic resource allocation based on predicted utilization.
5. CloudSim simulation and policy comparison.
6. Performance analysis and dashboard visualization.

## Dashboard Features

- Interactive CPU and memory scaling demonstration.
- CPU resource optimization analysis.
- Memory resource optimization analysis.
- CloudSim policy comparison.
- SLA violation analysis.
- Workload interval explorer.
- Downloadable simulation results.

## Project Results

The project evaluates three resource allocation policies:

- STATIC
- EXISTING_DYNAMIC
- OPTIMIZED_ML

The dashboard displays saved CloudSim results and optimized workload allocation results across 447 workload intervals.

## Repository Structure

```text
intelligent-cloud-resource-allocation/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── week8_cloudsim_simulation_results.csv
    └── final_combined_results.csv
```

## Run the Dashboard Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Future Scope

- Integrate real-time workload monitoring.
- Improve memory resource optimization.
- Extend predictive scaling and CloudSim evaluation.
- Deploy the dashboard for cloud-based monitoring.

## Author

B.Tech Computer Science and Engineering
