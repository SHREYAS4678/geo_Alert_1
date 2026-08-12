"""
GeoAlert - Synthetic Sensor Dataset Generator
================================================
No public labelled landslide-sensor dataset is freely available at the
resolution GeoAlert needs (per-sensor, 30-min cadence), so this script
generates a physically-motivated synthetic dataset instead — the standard,
defensible approach for a student IoT+ML project when real field data
hasn't been collected yet.

Each simulated "site-day" is a 48-step time series (30-min steps = 24h) for
one monitoring node, built from small stochastic processes that mimic how
these quantities actually behave in the field:

  rainfall_mm      - bursty AR(1) process with occasional "storm" spikes
  soil_moisture_%  - a leaky integrator of rainfall (rises fast, drains slow)
  temperature_c    - smooth diurnal sine wave + noise
  humidity_%       - anti-correlated with temperature, boosted by rainfall
  vibration_events - Poisson-distributed, rate increases with soil saturation
  tilt_deg         - slow stochastic drift that accelerates when soil is
                     saturated AND vibration is elevated (the instability proxy)

A site's final risk label (0=Safe, 1=Watch, 2=Warning, 3=Critical) is derived
from a weighted combination of the state in the last few steps, then jittered
with noise so the classes aren't trivially separable - so the ML models in
train_models.py have to do real work, matching a genuine classification task.

Output:
  data/geoalert_sequences.npz   - (N, 48, 6) sequences + (N,) labels, for the LSTM
  data/geoalert_tabular.csv     - one row per sequence (engineered features), for RF/XGBoost/SVM
"""
import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_SITES = 7000          # number of independent 24h site-day sequences
STEPS = 48               # 30-minute steps across 24 hours
FEATURES = ["soil_moisture", "rainfall_mm", "temperature_c", "humidity_pct", "vibration_events", "tilt_deg"]

rng = np.random.default_rng(RNG_SEED)


def simulate_site():
    """Simulate one 24h sequence for a single monitoring node."""
    rainfall = np.zeros(STEPS)
    soil = np.zeros(STEPS)
    temp = np.zeros(STEPS)
    hum = np.zeros(STEPS)
    vib = np.zeros(STEPS)
    tilt = np.zeros(STEPS)

    storm = rng.random() < 0.35                       # 35% of days include a storm
    storm_start = rng.integers(4, STEPS - 10) if storm else -1
    storm_len = rng.integers(4, 14)
    storm_intensity = rng.uniform(4, 22)               # mm per 30-min step at peak

    soil[0] = rng.uniform(15, 35)
    tilt[0] = rng.uniform(0.0, 1.5)
    rain_prev = 0.0

    for t in range(STEPS):
        # --- rainfall: base drizzle chance + storm window ---
        base_rain = max(0.0, rng.normal(0.3, 0.6))
        if storm and storm_start <= t < storm_start + storm_len:
            progress = (t - storm_start) / storm_len
            envelope = np.sin(np.pi * progress)         # ramps up then down
            storm_rain = storm_intensity * envelope * rng.uniform(0.7, 1.15)
        else:
            storm_rain = 0.0
        rainfall[t] = max(0.0, 0.4 * rain_prev + base_rain + storm_rain)
        rain_prev = rainfall[t]

        # --- soil moisture: leaky integrator (fills fast, drains slowly) ---
        prev_soil = soil[t - 1] if t > 0 else soil[0]
        inflow = rainfall[t] * 1.6
        drain = 0.985
        soil[t] = np.clip(prev_soil * drain + inflow, 5, 100)

        # --- temperature: diurnal cycle + noise ---
        hour = (t * 0.5) % 24
        temp[t] = 24 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.6)

        # --- humidity: inverse to temp, boosted by rain ---
        hum[t] = np.clip(70 - 1.1 * (temp[t] - 24) + 0.6 * rainfall[t] + rng.normal(0, 2), 25, 100)

        # --- vibration: Poisson, rate rises with saturation ---
        sat_factor = max(0.0, (soil[t] - 55) / 45)
        vib_rate = 0.4 + 6.5 * sat_factor ** 1.4
        vib[t] = rng.poisson(vib_rate)

        # --- tilt: slow drift, accelerates when saturated AND shaking ---
        prev_tilt = tilt[t - 1] if t > 0 else tilt[0]
        instability = sat_factor * (0.3 + 0.15 * vib[t])
        tilt[t] = max(0.0, prev_tilt + rng.normal(0.01, 0.02) + instability * rng.uniform(0.0, 0.09))

    seq = np.stack([soil, rainfall, temp, hum, vib, tilt], axis=1)  # (STEPS, 6)

    # --- risk score from the last 6 steps (3 hours) of state ---
    tail = seq[-6:]
    soil_n = np.clip(tail[:, 0].mean() / 100, 0, 1)
    rain_n = np.clip(tail[:, 1].sum() / 60, 0, 1)
    vib_n = np.clip(tail[:, 4].sum() / 30, 0, 1)
    tilt_n = np.clip(tail[:, 5].max() / 6, 0, 1)

    risk_score = 0.32 * soil_n + 0.22 * rain_n + 0.21 * vib_n + 0.25 * tilt_n
    risk_score = np.clip(risk_score + rng.normal(0, 0.045), 0, 1)  # label noise -> non-trivial task

    if risk_score < 0.35:
        label = 0   # Safe
    elif risk_score < 0.55:
        label = 1   # Watch
    elif risk_score < 0.75:
        label = 2   # Warning
    else:
        label = 3   # Critical

    return seq, label


def engineer_tabular_row(seq):
    """Collapse a (48, 6) sequence into the tabular feature row a live ESP32
    reading would actually provide: current instantaneous values plus a
    couple of short-window rolling aggregates (which the backend keeps in
    SQLite and computes the same way at inference time)."""
    soil, rain, temp, hum, vib, tilt = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3], seq[:, 4], seq[:, 5]
    return {
        "soil_moisture_pct": soil[-1],
        "rainfall_1h_mm": rain[-2:].sum(),
        "rainfall_24h_mm": rain.sum(),
        "temperature_c": temp[-1],
        "humidity_pct": hum[-1],
        "vibration_events_10min": vib[-1],
        "vibration_events_3h": vib[-6:].sum(),
        "tilt_deg": tilt[-1],
        "tilt_delta_3h": tilt[-1] - tilt[-6],
        "soil_trend_3h": soil[-1] - soil[-6],
    }


def main():
    os.makedirs("data", exist_ok=True) if not os.path.isdir("../data") else None
    out_dir = "data" if os.path.isdir("data") else "../data"

    sequences, labels, rows = [], [], []
    for _ in range(N_SITES):
        seq, label = simulate_site()
        sequences.append(seq)
        labels.append(label)
        row = engineer_tabular_row(seq)
        row["risk_label"] = label
        rows.append(row)

    sequences = np.array(sequences, dtype=np.float32)   # (N, 48, 6)
    labels = np.array(labels, dtype=np.int64)

    np.savez_compressed(os.path.join(out_dir, "geoalert_sequences.npz"),
                         X=sequences, y=labels, feature_names=np.array(FEATURES))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "geoalert_tabular.csv"), index=False)

    print(f"Generated {N_SITES} site-day sequences ({STEPS} steps x {len(FEATURES)} features)")
    print("Class balance:")
    names = {0: "Safe", 1: "Watch", 2: "Warning", 3: "Critical"}
    for k, v in sorted(pd.Series(labels).value_counts().items()):
        print(f"  {names[k]:9s} ({k}): {v:5d}  ({100*v/len(labels):.1f}%)")
    print(f"\nSaved:\n  {out_dir}/geoalert_sequences.npz  (for LSTM)\n  {out_dir}/geoalert_tabular.csv    (for RF / XGBoost / SVM)")


if __name__ == "__main__":
    main()
