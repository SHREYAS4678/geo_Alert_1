"""
GeoAlert - Live Dashboard
============================
Free, open-source, zero-hosting-cost UI: Streamlit's own Community Cloud
tier can host this for free too, or just run it locally during a viva.

Run:  streamlit run dashboard.py
(with backend/app.py already running in another terminal on :5000)
"""
import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

API_URL = os.environ.get("GEOALERT_API_URL", "http://localhost:5000")
NODE_ID = "demo-node-01"

st.set_page_config(page_title="GeoAlert Dashboard", page_icon="⛰️", layout="wide")

RISK_COLORS = {"Safe": "#2E8B57", "Watch": "#D9A441", "Warning": "#E07A2F", "Critical": "#C0392B"}


def api_get(path, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend not reachable at {API_URL} — is `python backend/app.py` running? ({e})")
        return None


def api_post(path, json_body):
    try:
        r = requests.post(f"{API_URL}{path}", json=json_body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


st.title("⛰️ GeoAlert — Live Monitoring Dashboard")
st.caption("IoT sensor fusion + ML ensemble + an agentic AI layer that decides when to alert")

# ---- Demo controls (works with zero physical hardware) ----
with st.sidebar:
    st.header("🧪 Demo — no hardware needed")
    st.write("Push a synthetic reading through the *real* backend pipeline "
             "(ensemble + agent) to see the whole system react live.")
    col_a, col_b = st.columns(2)
    if col_a.button("🟢 Safe", use_container_width=True):
        st.session_state["last_sim"] = api_post("/api/simulate", {"scenario": "safe", "node_id": NODE_ID})
    if col_b.button("🟡 Warning", use_container_width=True):
        st.session_state["last_sim"] = api_post("/api/simulate", {"scenario": "warning", "node_id": NODE_ID})
    col_c, col_d = st.columns(2)
    if col_c.button("🔴 Critical", use_container_width=True):
        st.session_state["last_sim"] = api_post("/api/simulate", {"scenario": "critical", "node_id": NODE_ID})
    if col_d.button("🎲 Random", use_container_width=True):
        st.session_state["last_sim"] = api_post("/api/simulate", {"scenario": "random", "node_id": NODE_ID})

    st.divider()
    st.caption(f"Backend: `{API_URL}`")
    if st.button("🔄 Refresh"):
        st.rerun()

# ---- Current status ----
latest = api_get("/api/latest", node_id=NODE_ID)

if latest:
    risk_name = latest.get("risk_name", "—")
    color = RISK_COLORS.get(risk_name, "#888")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"### Risk level\n<span style='color:{color};font-size:28px;font-weight:700'>{risk_name}</span>",
                unsafe_allow_html=True)
    c2.metric("Confidence", f"{(latest.get('risk_confidence') or 0)*100:.0f}%")
    c3.metric("Soil moisture", f"{latest.get('soil_moisture_pct', 0):.0f}%")
    c4.metric("Tilt", f"{latest.get('tilt_deg', 0):.2f}°")
    c5.metric("Rainfall (1h)", f"{latest.get('rainfall_1h_mm', 0):.1f} mm")
else:
    st.info("No readings yet — click a demo button in the sidebar, or POST to /api/ingest from the ESP32.")

# ---- Latest agent decision ----
if st.session_state.get("last_sim"):
    sim = st.session_state["last_sim"]
    agent_info = sim.get("agent", {})
    st.subheader("🤖 Agent decision")
    badge = "🔔 Alert sent" if agent_info.get("alert_sent") else (
        "🟠 Alert triggered (SMTP not configured — see README)" if agent_info.get("alert_triggered") else "🔕 No alert")
    st.markdown(f"**{badge}**  ·  mode: `{agent_info.get('mode')}`")
    st.write(agent_info.get("narrative", ""))

# ---- History charts ----
st.subheader("📈 Sensor history")
hist = api_get("/api/history", node_id=NODE_ID, hours=48)
if hist:
    df = pd.DataFrame(hist)
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values("time")

    tab1, tab2, tab3 = st.tabs(["Soil & Rainfall", "Tilt & Vibration", "Risk over time"])
    with tab1:
        st.line_chart(df.set_index("time")[["soil_moisture_pct", "rainfall_1h_mm"]])
    with tab2:
        st.line_chart(df.set_index("time")[["tilt_deg", "vibration_events_10min"]])
    with tab3:
        st.line_chart(df.set_index("time")[["risk_label"]])
        st.caption("0=Safe · 1=Watch · 2=Warning · 3=Critical")
else:
    st.caption("No history yet.")

# ---- Alert log ----
st.subheader("📨 Alert history")
alerts = api_get("/api/alerts")
if alerts:
    adf = pd.DataFrame(alerts)
    adf["time"] = pd.to_datetime(adf["ts"], unit="s")
    st.dataframe(adf[["time", "node_id", "severity", "message", "sent"]].sort_values("time", ascending=False),
                 use_container_width=True, hide_index=True)
else:
    st.caption("No alerts sent yet.")
