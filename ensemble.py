"""
GeoAlert - Ensemble Inference
===============================
Loads the four models trained by ml/train_models.py and reproduces the same
soft-voting combination at inference time. Import `predict_risk(features)`
from app.py for every incoming sensor reading.
"""
import os
import json
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
CLASS_NAMES = ["Safe", "Watch", "Warning", "Critical"]

_cache = {}


def _load():
    if _cache:
        return _cache
    import tensorflow as tf  # local import: keeps Flask startup fast if TF isn't needed yet

    _cache["rf"] = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    _cache["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    _cache["svm"] = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
    _cache["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    _cache["lstm"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
    norm = np.load(os.path.join(MODEL_DIR, "lstm_norm_stats.npz"))
    _cache["lstm_mean"], _cache["lstm_std"] = norm["mean"], norm["std"]
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        meta = json.load(f)
    _cache["feature_columns"] = meta["feature_columns"]
    w = meta["ensemble_weights"]
    _cache["weights"] = (w["rf"], w["xgboost"], w["svm"], w["lstm"])
    return _cache


def _tabular_row(features: dict, feature_columns: list) -> np.ndarray:
    """features must already contain the two derived fields
    (vibration_events_3h, tilt_delta_3h, soil_trend_3h) computed by the
    caller from recent history — see backend/app.py `build_feature_row`."""
    return np.array([[features[c] for c in feature_columns]], dtype=np.float32)


def _sequence_row(history_rows: list, feature_columns_seq=("soil_moisture_pct", "rainfall_1h_mm",
                                                             "temperature_c", "humidity_pct",
                                                             "vibration_events_10min", "tilt_deg"),
                   n_steps: int = 48) -> np.ndarray:
    """Build a (1, 48, 6) tensor from up to the last 48 stored readings,
    oldest first, padding by repeating the earliest row if a node hasn't
    accumulated 48 readings yet (cold start)."""
    rows = list(reversed(history_rows))[-n_steps:]  # oldest -> newest
    while len(rows) < n_steps:
        rows.insert(0, rows[0])
    seq = np.array([[r.get(c, 0.0) or 0.0 for c in feature_columns_seq] for r in rows], dtype=np.float32)
    return seq[np.newaxis, :, :]  # (1, 48, 6)


def predict_risk(features: dict, history_rows: list):
    """
    features:     dict with the CURRENT reading's tabular feature columns
                  (see ml/models/metrics.json -> feature_columns)
    history_rows: list of recent DB rows for this node (most-recent-first),
                  used to build the LSTM's 24h sequence and the rolling
                  aggregate fields. May be empty for a brand-new node's
                  very first reading — the sequence then bootstraps by
                  treating the current reading as if it had been steady,
                  which is the sensible cold-start assumption.
    Returns: (risk_label:int, risk_name:str, confidence:float, per_model:dict)
    """
    m = _load()
    X = _tabular_row(features, m["feature_columns"])
    X_s = m["scaler"].transform(X)
    X_seq = _sequence_row(history_rows if history_rows else [features])
    X_seq_n = (X_seq - m["lstm_mean"]) / m["lstm_std"]

    p_rf = m["rf"].predict_proba(X)[0]
    p_xgb = m["xgb"].predict_proba(X)[0]
    p_svm = m["svm"].predict_proba(X_s)[0]
    p_lstm = m["lstm"].predict(X_seq_n, verbose=0)[0]

    w_rf, w_xgb, w_svm, w_lstm = m["weights"]
    combined = w_rf * p_rf + w_xgb * p_xgb + w_svm * p_svm + w_lstm * p_lstm
    combined = combined / combined.sum()

    label = int(np.argmax(combined))
    per_model = {
        "random_forest": {CLASS_NAMES[i]: round(float(p_rf[i]), 3) for i in range(4)},
        "xgboost": {CLASS_NAMES[i]: round(float(p_xgb[i]), 3) for i in range(4)},
        "svm": {CLASS_NAMES[i]: round(float(p_svm[i]), 3) for i in range(4)},
        "lstm": {CLASS_NAMES[i]: round(float(p_lstm[i]), 3) for i in range(4)},
        "ensemble": {CLASS_NAMES[i]: round(float(combined[i]), 3) for i in range(4)},
    }
    return label, CLASS_NAMES[label], round(float(combined[label]), 4), per_model


if __name__ == "__main__":
    # quick smoke test with a synthetic "elevated risk" reading and a flat history
    demo_features = {
        "soil_moisture_pct": 78.0, "rainfall_1h_mm": 9.0, "rainfall_24h_mm": 62.0,
        "temperature_c": 25.0, "humidity_pct": 88.0, "vibration_events_10min": 7,
        "vibration_events_3h": 22, "tilt_deg": 3.4, "tilt_delta_3h": 1.1, "soil_trend_3h": 14.0,
    }
    demo_history = [dict(demo_features, ts=i) for i in range(10)]
    label, name, conf, per_model = predict_risk(demo_features, demo_history)
    print(f"Predicted risk: {name} (class {label}), confidence {conf}")
    print(json.dumps(per_model, indent=2))
