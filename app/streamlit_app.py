"""
app/streamlit_app.py
====================
Interactive fleet-maintenance demo for the NASA C-MAPSS FD001 models.

Pick a test engine in the sidebar; the app shows:
  * its raw sensor trajectories over its observed life (Plotly),
  * the model's predicted Remaining Useful Life as a large number,
  * a color-coded maintenance alert (green > 50, yellow 30-50, red <= 30),
  * a plain-English "what does this mean?" box with the business action +
    estimated cost avoided,
  * the SHAP waterfall explaining THIS engine's prediction.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when Streamlit runs this file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st

from data import download_data, load_rul, load_test
from evaluation import FAILURE_COST, SCHEDULED_COST
from explain import build_explainer, load_model, load_test_eval
from features import informative_sensor_cols
from targets import add_test_rul, last_cycle_per_engine

# Alert thresholds (cycles of remaining life).
GREEN_MIN = 50      # > 50  -> healthy
YELLOW_MIN = 30     # 30-50 -> plan soon ; <= 30 -> act now

st.set_page_config(page_title="Prognostic Fleet Maintenance", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders (run once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_model_and_explainer():
    """Load the tuned XGBoost model, its feature list, and a SHAP explainer."""
    model, feature_cols = load_model()
    test_last, X = load_test_eval(feature_cols)
    explainer = build_explainer(model)
    explanation = explainer(X)            # SHAP for all 100 test engines, once
    preds = np.clip(model.predict(X), 0, 125)
    return model, feature_cols, test_last, X, explanation, preds


@st.cache_data
def get_raw_test():
    """Raw (unengineered) test sensor readings + per-row true RUL, for plotting."""
    download_data()
    raw = load_test()
    raw_rul = add_test_rul(raw, load_rul())
    sensors = informative_sensor_cols(raw)
    return raw_rul, sensors


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def alert_for(rul: float):
    """Map a predicted RUL to (color, label, action, dollars-avoided text)."""
    if rul > GREEN_MIN:
        return ("#1a9850", "HEALTHY",
                "Continue normal operation; keep monitoring.",
                "No maintenance cost incurred.")
    if rul >= YELLOW_MIN:
        return ("#f0a30a", "PLAN MAINTENANCE",
                "Schedule maintenance within the next few dozen cycles.",
                f"Planned service (~${SCHEDULED_COST:,}) avoids a possible "
                f"unplanned failure (~${FAILURE_COST:,}).")
    return ("#d73027", "ACT NOW",
            "Schedule immediate maintenance — failure is imminent.",
            f"Acting now (~${SCHEDULED_COST:,}) avoids ~"
            f"${FAILURE_COST - SCHEDULED_COST:,} in unplanned-failure cost.")


def sensor_figure(engine_df, sensors, normalize):
    """Plotly line chart of an engine's sensors over its observed cycles."""
    fig = go.Figure()
    for s in sensors:
        y = engine_df[s]
        if normalize:                      # z-score so different scales overlay
            y = (y - y.mean()) / (y.std() or 1.0)
        fig.add_trace(go.Scatter(x=engine_df["time_cycles"], y=y,
                                 mode="lines", name=s))
    fig.update_layout(
        xaxis_title="cycle",
        yaxis_title="z-score" if normalize else "raw reading",
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("✈️ Prognostic Fleet Maintenance — Remaining Useful Life")
st.caption("NASA C-MAPSS FD001 · tuned XGBoost · cost-aware, SHAP-explained")

model, feature_cols, test_last, X, explanation, preds = get_model_and_explainer()
raw_rul, sensors = get_raw_test()

# ---- Sidebar: engine picker -------------------------------------------------
engine_ids = sorted(test_last["unit_number"].astype(int).tolist())
with st.sidebar:
    st.header("Select engine")
    engine_id = st.selectbox("Test engine ID", engine_ids, index=0)
    default_sensors = [s for s in ["sensor_4", "sensor_11", "sensor_15"] if s in sensors]
    chosen = st.multiselect("Sensors to plot", sensors, default=default_sensors)
    normalize = st.checkbox("Normalize sensors (z-score)", value=True)

# Locate this engine's row in the test-eval frame.
pos = int(test_last.index.get_loc(test_last.index[test_last["unit_number"] == engine_id][0]))
pred_rul = float(preds[pos])
true_rul = float(test_last.iloc[pos]["RUL"])
color, label, action, savings = alert_for(pred_rul)

# ---- Top row: big RUL number + alert ---------------------------------------
c1, c2 = st.columns([1, 2])
with c1:
    st.metric(label=f"Predicted RUL — engine #{engine_id}",
              value=f"{pred_rul:.0f} cycles",
              delta=f"true: {true_rul:.0f}")
with c2:
    st.markdown(
        f"""
        <div style="background:{color};padding:18px;border-radius:10px;color:white">
          <div style="font-size:22px;font-weight:700">{label}</div>
          <div style="font-size:15px;margin-top:4px">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Roadmap item 4: plain-English business meaning + estimated savings.
    st.info(f"**What does this mean?** {savings}")

st.divider()

# ---- Sensor trajectories ----------------------------------------------------
st.subheader(f"Sensor readings over engine #{engine_id}'s observed life")
engine_df = raw_rul[raw_rul["unit_number"] == engine_id].sort_values("time_cycles")
if chosen:
    st.plotly_chart(sensor_figure(engine_df, chosen, normalize), use_container_width=True)
else:
    st.warning("Pick at least one sensor in the sidebar.")

# ---- SHAP waterfall for this engine ----------------------------------------
st.subheader("Why this prediction? (SHAP)")
st.caption("base value + Σ(feature contributions) = predicted RUL — an exact decomposition.")
fig = plt.figure()
shap.plots.waterfall(explanation[pos], max_display=12, show=False)
st.pyplot(fig, clear_figure=True, bbox_inches="tight")

st.caption(
    f"Alert thresholds — green > {GREEN_MIN} · yellow {YELLOW_MIN}-{GREEN_MIN} "
    f"· red ≤ {YELLOW_MIN} cycles."
)
