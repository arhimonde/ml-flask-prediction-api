"""
Model Training and Serialization Script for Jetson / Edge Deployment.
Trains a Pipeline (Scaler + Random Forest Classifier) for industrial IoT/Sensor telemetry
and serializes it using joblib.
"""

import os
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sensor_classifier.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Define expected features and target classes
FEATURES = ["temperature", "vibration", "pressure", "rotational_speed", "voltage"]
CLASSES = ["NORMAL", "WARNING", "CRITICAL"]

def generate_synthetic_telemetry(n_samples: int = 3000, random_state: int = 42):
    """
    Generates realistic sensor telemetry:
    - Normal: nominal operating conditions
    - Warning: mild elevated temperature, vibration or voltage anomalies
    - Critical: extreme vibration, severe overheating, or drop in pressure
    """
    rng = np.random.RandomState(random_state)
    samples_per_class = n_samples // 3

    # 1. Normal State
    temp_norm = rng.normal(loc=65.0, scale=4.0, size=samples_per_class)
    vib_norm = rng.normal(loc=1.5, scale=0.3, size=samples_per_class)
    press_norm = rng.normal(loc=4.0, scale=0.2, size=samples_per_class)
    rpm_norm = rng.normal(loc=1800.0, scale=30.0, size=samples_per_class)
    volt_norm = rng.normal(loc=230.0, scale=3.0, size=samples_per_class)
    y_norm = np.zeros(samples_per_class, dtype=int)

    # 2. Warning State (moderate friction / wear)
    temp_warn = rng.normal(loc=82.0, scale=6.0, size=samples_per_class)
    vib_warn = rng.normal(loc=3.8, scale=0.6, size=samples_per_class)
    press_warn = rng.normal(loc=3.4, scale=0.4, size=samples_per_class)
    rpm_warn = rng.normal(loc=1720.0, scale=60.0, size=samples_per_class)
    volt_warn = rng.normal(loc=222.0, scale=6.0, size=samples_per_class)
    y_warn = np.ones(samples_per_class, dtype=int)

    # 3. Critical State (bearing failure / overheating imminent)
    temp_crit = rng.normal(loc=105.0, scale=8.0, size=samples_per_class)
    vib_crit = rng.normal(loc=7.5, scale=1.2, size=samples_per_class)
    press_crit = rng.normal(loc=2.5, scale=0.6, size=samples_per_class)
    rpm_crit = rng.normal(loc=1500.0, scale=120.0, size=samples_per_class)
    volt_crit = rng.normal(loc=205.0, scale=12.0, size=samples_per_class)
    y_crit = np.full(samples_per_class, 2, dtype=int)

    X = np.vstack([
        np.column_stack([temp_norm, vib_norm, press_norm, rpm_norm, volt_norm]),
        np.column_stack([temp_warn, vib_warn, press_warn, rpm_warn, volt_warn]),
        np.column_stack([temp_crit, vib_crit, press_crit, rpm_crit, volt_crit]),
    ])
    y = np.concatenate([y_norm, y_warn, y_crit])

    # Shuffle
    indices = rng.permutation(len(y))
    return X[indices], y[indices]

def train_and_export():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Generating telemetry training data...")
    X, y = generate_synthetic_telemetry(n_samples=4500)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Scikit-learn Pipeline combining feature scaling and a lightweight Random Forest
    # n_estimators=60 is fast and memory-efficient for embedded systems like Jetson Nano/Orin
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=60,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)

    print(f"Model Training Complete! Test Accuracy: {acc * 100:.2f}%")

    # Serializing model using joblib with compression
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    file_size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"Saved serialized model to: {MODEL_PATH} ({file_size_kb:.1f} KB)")

    # Serializing metadata
    metadata = {
        "model_name": "IoT_Sensor_Health_Classifier",
        "version": "1.0.0",
        "framework": "scikit-learn",
        "features": FEATURES,
        "classes": CLASSES,
        "accuracy": round(float(acc), 4),
        "test_metrics": report,
        "input_schema": {
            feature: {"type": "float", "unit": unit, "sample_normal": sample_val}
            for feature, unit, sample_val in zip(
                FEATURES,
                ["°C", "mm/s", "bar", "RPM", "V"],
                [65.2, 1.45, 4.05, 1805.0, 230.1]
            )
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {METADATA_PATH}")

if __name__ == "__main__":
    train_and_export()
