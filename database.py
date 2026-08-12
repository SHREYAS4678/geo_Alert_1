"""
GeoAlert - Storage Layer
==========================
SQLite is used deliberately: it's a single file, ships with Python's
standard library, and needs no server or account — genuinely free, with
zero setup, which matches the brief. Swap DB_PATH for a Postgres/Firebase
URL later if the deployment ever needs multiple concurrent writers.
"""
import sqlite3
import time
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "geoalert.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    node_id TEXT NOT NULL,
    soil_moisture_pct REAL,
    rainfall_1h_mm REAL,
    rainfall_24h_mm REAL,
    temperature_c REAL,
    humidity_pct REAL,
    vibration_events_10min REAL,
    tilt_deg REAL,
    risk_label INTEGER,
    risk_name TEXT,
    risk_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);
CREATE INDEX IF NOT EXISTS idx_readings_node ON readings(node_id);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    node_id TEXT NOT NULL,
    severity TEXT,
    message TEXT,
    agent_reasoning TEXT,
    sent INTEGER DEFAULT 0
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_reading(node_id: str, features: dict, risk_label: int, risk_name: str, risk_confidence: float):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO readings
               (ts, node_id, soil_moisture_pct, rainfall_1h_mm, rainfall_24h_mm,
                temperature_c, humidity_pct, vibration_events_10min, tilt_deg,
                risk_label, risk_name, risk_confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), node_id,
             features.get("soil_moisture_pct"), features.get("rainfall_1h_mm"), features.get("rainfall_24h_mm"),
             features.get("temperature_c"), features.get("humidity_pct"),
             features.get("vibration_events_10min"), features.get("tilt_deg"),
             risk_label, risk_name, risk_confidence),
        )


def recent_readings(node_id: str, hours: float = 3.0, limit: int = 200):
    cutoff = time.time() - hours * 3600
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM readings WHERE node_id=? AND ts>=? ORDER BY ts DESC LIMIT ?",
            (node_id, cutoff, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def latest_reading(node_id: str = None):
    with get_conn() as conn:
        if node_id:
            row = conn.execute(
                "SELECT * FROM readings WHERE node_id=? ORDER BY ts DESC LIMIT 1", (node_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def all_readings(limit: int = 500):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def insert_alert(node_id: str, severity: str, message: str, agent_reasoning: str, sent: bool):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO alerts (ts, node_id, severity, message, agent_reasoning, sent) VALUES (?,?,?,?,?,?)",
            (time.time(), node_id, severity, message, agent_reasoning, int(sent)),
        )


def recent_alerts(limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialised database at {DB_PATH}")
