"""Streamlit monitoring dashboard - thin view, all logic is in ``fishpond.dashboard``."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from mlflow.tracking import MlflowClient

from fishpond import dashboard, simulate
from fishpond.config import get_settings
from fishpond.orchestration import FishpondClient

settings = get_settings()
st.set_page_config(page_title="Fishpond monitoring", layout="wide")
st.title("Fishpond weight model - monitoring")
st.caption(
    f"API: {settings.api_url} | MLflow: {settings.mlflow_tracking_uri} | registered model: {settings.model_name}"
)

with st.sidebar:
    st.header("Simulate production traffic")
    amount = st.slider("Number of readings", 10, 500, 200, step=10)
    if st.button("Send normal readings"):
        records = simulate.sample_records(dashboard.load_monitoring_data(settings).reference, amount)
        sent = simulate.send_records(FishpondClient(settings.api_url), records)
        st.success(f"{sent} predictions received")
    if st.button("Send drifted readings (heat wave)"):
        records = simulate.sample_records(dashboard.load_monitoring_data(settings).reference, amount, drift=True)
        sent = simulate.send_records(FishpondClient(settings.api_url), records)
        st.warning(f"{sent} drifted predictions received")

data = dashboard.load_monitoring_data(settings)
summary = dashboard.prediction_summary(data.current)
columns = st.columns(4)
columns[0].metric("Logged predictions", summary["predictions"])
columns[1].metric("Requests", summary["requests"])
mean = summary["mean_prediction"]
columns[2].metric("Mean predicted weight (g)", "-" if mean is None else f"{mean:.1f}")
columns[3].metric("Served model versions", ", ".join(summary["model_versions"]) or "-")

overview_tab, drift_tab, log_tab = st.tabs(["Model registry", "Evidently drift report", "Prediction log"])

with overview_tab:
    st.subheader("Registered model versions (champion / challenger)")
    st.dataframe(dashboard.registry_overview(MlflowClient(settings.mlflow_tracking_uri), settings.model_name))
    st.subheader("Predictions per model version")
    st.dataframe(dashboard.predictions_by_version(data.current))

with drift_tab:
    report = dashboard.drift_report(data)
    if report is None:
        st.info("No logged predictions yet - send some readings from the sidebar or call POST /predict.")
    else:
        status = "DRIFT DETECTED" if report.dataset_drift else "no dataset drift"
        drifted = ", ".join(report.drifted_columns) or "-"
        st.markdown(f"**{status}** - share of drifted columns: {report.drift_share:.0%} ({drifted})")
        components.html(report.html, height=1200, scrolling=True)

with log_tab:
    st.dataframe(data.current.tail(500))
