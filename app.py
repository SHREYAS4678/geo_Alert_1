"""
GeoAlert - Backend API
=========================
Free-tier friendly by design: Flask's built-in dev server is fine for a
college demo (or deploy free on Render/PythonAnywhere's free tier later),
SQLite needs no hosted database, and the only paid-capable dependency
(Gemini) has a genuinely free tier (see backend/agent.py).

Run:  python app.py          (starts on http://localhost:5000)

Endpoints:
  POST /api/ingest      <- ESP32 posts a sensor reading here
  GET  /api/latest       -> most recent reading (+ risk) per node
  GET  /api/history       -> recent readings for charts
  GET  /api/alerts        -> recent agent alerts
  POST /api/simulate      -> push a synthetic reading (demo without hardware)
"""
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
import database as db
import ensemble
import agent

app = Flask(__name__)
CORS(app)  # dashboard (Streamlit) runs on a different port -> needs CORS

db.init_db()

# The two derived rolling-window fields the tabular models expect
# (vibration_events_3h, tilt_delta_3h, soil_trend_3h) are computed here from
# history rather than sent by the ESP32, so the firmware only ever has to
# transmit instantaneous sensor values.
def build_feature_row(node_id: str, raw: dict) -> dict:
    history = db.recent_readings(node_id, hours=3.0, limit=50)
    vib_3h = raw["vibration_events_10min"] + sum(
        (h["vibration_events_10min"] or 0) for h in history)
    if history:
        tilt_then = history[-1]["tilt_deg"] or raw["tilt_deg"]
        soil_then = history[-1]["soil_moisture_pct"] or raw["soil_moisture_pct"]
    else:
        tilt_then, soil_then = raw["tilt_deg"], raw["soil_moisture_pct"]

    row = dict(raw)
    row["vibration_events_3h"] = vib_3h
    row["tilt_delta_3h"] = raw["tilt_deg"] - tilt_then
    row["soil_trend_3h"] = raw["soil_moisture_pct"] - soil_then
    return row, history


REQUIRED_FIELDS = ["node_id", "soil_moisture_pct", "rainfall_1h_mm", "rainfall_24h_mm",
                   "temperature_c", "humidity_pct", "vibration_events_10min", "tilt_deg"]


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """Primary endpoint: the ESP32 (or the /api/simulate helper below) posts
    a JSON body with the REQUIRED_FIELDS above. Runs the ensemble, runs the
    agent, stores everything, and returns the full decision for logging."""
    payload = request.get_json(force=True, silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    node_id = payload["node_id"]
    raw = {k: float(payload[k]) for k in REQUIRED_FIELDS if k != "node_id"}
    features, history = build_feature_row(node_id, raw)

    label, name, confidence, per_model = ensemble.predict_risk(features, history)
    db.insert_reading(node_id, features, label, name, confidence)

    agent_result = agent.run_agent(node_id, features, label, name, confidence, per_model)
    if agent_result["alert_triggered"]:
        for s in agent_result["send_log"]:
            db.insert_alert(node_id, s["severity"], s["message"], agent_result["narrative"], s["sent"])

    return jsonify({
        "node_id": node_id,
        "risk_label": label,
        "risk_name": name,
        "confidence": confidence,
        "per_model": per_model,
        "agent": agent_result,
    })


@app.route("/api/latest", methods=["GET"])
def latest():
    node_id = request.args.get("node_id")
    row = db.latest_reading(node_id)
    return jsonify(row or {})


@app.route("/api/history", methods=["GET"])
def history():
    node_id = request.args.get("node_id")
    hours = float(request.args.get("hours", 24))
    if node_id:
        return jsonify(db.recent_readings(node_id, hours=hours, limit=500))
    return jsonify(db.all_readings(limit=500))


@app.route("/api/alerts", methods=["GET"])
def alerts():
    return jsonify(db.recent_alerts(limit=50))


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Push one synthetic reading through the full pipeline — lets you demo
    the dashboard + agent live in a viva without any physical hardware."""
    import random
    body = request.get_json(force=True, silent=True) or {}
    scenario = body.get("scenario", "random")  # "safe" | "warning" | "critical" | "random"

    presets = {
        "safe": dict(soil_moisture_pct=28, rainfall_1h_mm=0.2, rainfall_24h_mm=3, temperature_c=27,
                     humidity_pct=55, vibration_events_10min=0, tilt_deg=0.4),
        "warning": dict(soil_moisture_pct=72, rainfall_1h_mm=8, rainfall_24h_mm=55, temperature_c=24,
                        humidity_pct=86, vibration_events_10min=4, tilt_deg=2.6),
        "critical": dict(soil_moisture_pct=91, rainfall_1h_mm=15, rainfall_24h_mm=98, temperature_c=23,
                         humidity_pct=95, vibration_events_10min=11, tilt_deg=4.8),
    }
    if scenario == "random":
        raw = random.choice(list(presets.values()))
        raw = {k: v * random.uniform(0.9, 1.1) for k, v in raw.items()}
    else:
        raw = presets.get(scenario, presets["safe"])

    node_id = body.get("node_id", "demo-node-01")
    payload = dict(raw)
    features, history = build_feature_row(node_id, payload)
    label, name, confidence, per_model = ensemble.predict_risk(features, history)
    db.insert_reading(node_id, features, label, name, confidence)
    agent_result = agent.run_agent(node_id, features, label, name, confidence, per_model)
    if agent_result["alert_triggered"]:
        for s in agent_result["send_log"]:
            db.insert_alert(node_id, s["severity"], s["message"], agent_result["narrative"], s["sent"])

    return jsonify({"node_id": node_id, "risk_name": name, "confidence": confidence,
                    "agent": agent_result, "injected_reading": payload})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
