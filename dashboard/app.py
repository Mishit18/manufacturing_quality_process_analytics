from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Manufacturing Quality Command Center", layout="wide")
st.title("Manufacturing Quality Command Center")
st.caption("Simulated operations analysis with an independent UCI Steel Plates Faults benchmark")

oee = pd.read_csv(OUTPUTS / "oee_by_line_machine.csv")
bottlenecks = pd.read_csv(OUTPUTS / "bottleneck_rankings.csv")
spc = pd.read_csv(OUTPUTS / "spc_control_chart_data.csv")
real_metrics = pd.read_csv(OUTPUTS / "uci_steel_faults_metrics.csv").iloc[0]

left, middle, right, fourth = st.columns(4)
left.metric("Lowest OEE", f"{oee['oee'].min():.1%}")
middle.metric("SPC breach-days", int(spc["spc_breach"].sum()))
right.metric("Real-data accuracy", f"{real_metrics['accuracy']:.1%}")
fourth.metric("Real-data macro-F1", f"{real_metrics['macro_f1']:.1%}")

st.plotly_chart(
    px.bar(
        bottlenecks.head(12),
        x="line",
        y="bottleneck_score",
        color="machine",
        hover_data=["oee", "defect_rate", "scrap_cost_inr"],
        title="Priority Bottlenecks",
    ),
    use_container_width=True,
)

st.subheader("Management action queue")
st.dataframe(pd.read_csv(OUTPUTS / "recommended_actions.csv"), use_container_width=True)

st.subheader("Real UCI steel-fault benchmark feature importance")
importance = pd.read_csv(OUTPUTS / "uci_steel_faults_importance.csv").head(12)
st.plotly_chart(
    px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h"),
    use_container_width=True,
)
