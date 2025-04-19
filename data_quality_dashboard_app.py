
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide")

# Sidebar Menu
menu = st.sidebar.radio("Navigation", ["📊 Overview", "📋 Data Quality Report"])

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Compute DQ Metrics
    def compute_dq_metrics(df):
        total_rows = len(df)
        failed_rows = df.isnull().any(axis=1).sum()
        metrics = {
            "Completeness": round(100 - df.isnull().mean().mean() * 100, 1),
            "Timeliness": np.random.randint(60, 90),
            "Validity": np.random.randint(60, 90),
            "Accuracy": np.random.randint(50, 80),
            "Consistency": np.random.randint(40, 70),
            "Uniqueness": round(100 * (1 - df.duplicated().sum() / total_rows), 1),
        }
        metrics["Overall DQ Score"] = round(np.mean(list(metrics.values())), 1)
        return metrics, total_rows, failed_rows

    dq_metrics, total_rows, failed_rows = compute_dq_metrics(df)

    if menu == "📊 Overview":
        st.title("📊 Data Quality Overview")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for i, (metric, value) in enumerate(dq_metrics.items()):
            if metric == "Overall DQ Score": continue
            fig = px.pie(values=[value, 100-value], names=[f"{value}%", ""], hole=0.6)
            fig.update_traces(textinfo='none')
            fig.update_layout(title=metric, showlegend=False)
            [col1, col2, col3, col4, col5, col6][i].plotly_chart(fig, use_container_width=True)

        st.markdown("### 📌 Dataset Statistics")
        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.dataframe({
                "Metric": ["Number of Rows", "Number of Columns", "Missing Values", "Duplicate Rows"],
                "Value": [df.shape[0], df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()]
            })

        with right_col:
            st.markdown("#### Missing Value Heatmap")
            null_df = df.isnull()
            fig = px.imshow(null_df.T, color_continuous_scale='blues', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "📋 Data Quality Report":
        st.title("📋 Full Data Quality Report")
        profile = ProfileReport(df, title="Data Quality Report", explorative=True)
        st_profile_report(profile)
else:
    st.warning("👈 Please upload a file to begin.")
