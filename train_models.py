"""
GeoAlert - Ensemble Model Training
====================================
Trains the four algorithms named in the project report (Random Forest,
XGBoost, SVM, LSTM) and combines them into a soft-voting ensemble, matching
Chapter 7's "RF + XGBoost + LSTM Hybrid Model" design (SVM included as the
report's Chapter 1/5 text also names it as one of the compared algorithms).

Run after generate_dataset.py. Everything here is free/open-source:
scikit-learn, XGBoost, TensorFlow/Keras — no paid API or service required.

Outputs -> ml/models/
  rf_model.joblib, xgb_model.joblib, svm_model.joblib, scaler.joblib
  lstm_model.keras
  metrics.json          <- real, measured accuracy/F1 per model + ensemble
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import joblib

DATA_DIR = "data" if os.path.isdir("data") else "../data"
MODEL_DIR = "ml/models" if os.path.isdir("ml/models") else "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CLASS_NAMES = ["Safe", "Watch", "Warning", "Critical"]
SEED = 42


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "geoalert_tabular.csv"))
    feature_cols = [c for c in df.columns if c != "risk_label"]
    X = df[feature_cols].values
    y = df["risk_label"].values

    seq_npz = np.load(os.path.join(DATA_DIR, "geoalert_sequences.npz"), allow_pickle=True)
    X_seq, y_seq = seq_npz["X"], seq_npz["y"]
    assert np.array_equal(y, y_seq), "tabular and sequence labels must line up 1:1"
    return X, y, X_seq, feature_cols


def train_tabular(X_train, y_train, X_test, y_test):
    sample_w = compute_sample_weight("balanced", y_train)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train, sample_weight=sample_w)

    svm = SVC(
        kernel="rbf", C=8.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=SEED,
    ).fit(X_train_s, y_train)

    results = {}
    for name, model, xte in [("random_forest", rf, X_test), ("xgboost", xgb_clf, X_test), ("svm", svm, X_test_s)]:
        pred = model.predict(xte)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        }
        print(f"\n--- {name} ---")
        print(classification_report(y_test, pred, target_names=CLASS_NAMES, zero_division=0))

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    return rf, xgb_clf, svm, scaler, results


def train_lstm(X_seq_train, y_train, X_seq_test, y_test):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)

    # normalise sequences feature-wise using train statistics only
    mean = X_seq_train.reshape(-1, X_seq_train.shape[-1]).mean(axis=0)
    std = X_seq_train.reshape(-1, X_seq_train.shape[-1]).std(axis=0) + 1e-6
    X_seq_train_n = (X_seq_train - mean) / std
    X_seq_test_n = (X_seq_test - mean) / std
    np.savez(os.path.join(MODEL_DIR, "lstm_norm_stats.npz"), mean=mean, std=std)

    n_timesteps, n_features = X_seq_train.shape[1], X_seq_train.shape[2]
    model = keras.Sequential([
        layers.Input(shape=(n_timesteps, n_features)),
        layers.LSTM(48, return_sequences=True),
        layers.Dropout(0.25),
        layers.LSTM(24),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class_w = {i: w for i, w in enumerate(
        len(y_train) / (4 * np.bincount(y_train, minlength=4) + 1e-6))}

    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    model.fit(
        X_seq_train_n, y_train, validation_split=0.15, epochs=60, batch_size=64,
        class_weight=class_w, callbacks=[early_stop], verbose=2,
    )

    pred_prob = model.predict(X_seq_test_n, verbose=0)
    pred = pred_prob.argmax(axis=1)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
    }
    print("\n--- lstm ---")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, zero_division=0))

    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    return model, metrics


def ensemble_predict_proba(rf, xgb_clf, svm, scaler, lstm, X_test, X_test_s_for_svm, X_seq_test, lstm_mean, lstm_std,
                            weights=(0.27, 0.27, 0.16, 0.30)):
    """Soft-vote across all four models. LSTM gets the largest single weight
    since it alone sees the full 24h trend rather than a snapshot."""
    p_rf = rf.predict_proba(X_test)
    p_xgb = xgb_clf.predict_proba(X_test)
    p_svm = svm.predict_proba(X_test_s_for_svm)
    X_seq_n = (X_seq_test - lstm_mean) / lstm_std
    p_lstm = lstm.predict(X_seq_n, verbose=0)

    w_rf, w_xgb, w_svm, w_lstm = weights
    combined = w_rf * p_rf + w_xgb * p_xgb + w_svm * p_svm + w_lstm * p_lstm
    return combined


def main():
    print("Loading dataset...")
    X, y, X_seq, feature_cols = load_data()

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    X_seq_train, X_seq_test = X_seq[idx_train], X_seq[idx_test]

    print(f"\nTrain size: {len(idx_train)}  |  Test size: {len(idx_test)}")
    print(f"Features used by RF/XGBoost/SVM ({len(feature_cols)}): {feature_cols}")

    print("\n================ Training tabular models (RF, XGBoost, SVM) ================")
    rf, xgb_clf, svm, scaler, tab_results = train_tabular(X_train, y_train, X_test, y_test)

    print("\n================ Training LSTM (sequence model) ================")
    lstm, lstm_metrics = train_lstm(X_seq_train, y_train, X_seq_test, y_test)

    print("\n================ Building the soft-voting ensemble ================")
    X_test_s = scaler.transform(X_test)
    norm = np.load(os.path.join(MODEL_DIR, "lstm_norm_stats.npz"))
    proba = ensemble_predict_proba(rf, xgb_clf, svm, scaler, lstm, X_test, X_test_s, X_seq_test,
                                    norm["mean"], norm["std"])
    ensemble_pred = proba.argmax(axis=1)
    ensemble_metrics = {
        "accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "f1_macro": round(f1_score(y_test, ensemble_pred, average="macro"), 4),
    }
    print(classification_report(y_test, ensemble_pred, target_names=CLASS_NAMES, zero_division=0))

    all_metrics = {**tab_results, "lstm": lstm_metrics, "ensemble": ensemble_metrics,
                   "feature_columns": feature_cols, "class_names": CLASS_NAMES,
                   "ensemble_weights": {"rf": 0.27, "xgboost": 0.27, "svm": 0.16, "lstm": 0.30}}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n================ SUMMARY (accuracy | macro-F1) ================")
    for name, m in all_metrics.items():
        if isinstance(m, dict) and "accuracy" in m:
            print(f"  {name:14s}  {m['accuracy']*100:5.1f}%   |  {m['f1_macro']*100:5.1f}%")
    print(f"\nModels + metrics.json saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
