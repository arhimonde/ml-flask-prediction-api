"""
Production-Ready Flask REST API for Machine Learning Model Inference.
Optimized for Edge Computing (NVIDIA Jetson / Raspberry Pi / x86 Servers).
"""

import os
import time
import json
import psutil
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, make_response

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sensor_classifier.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

app = Flask(__name__)
START_TIME = time.time()

# ---------------------------------------------------------
# Model & Metadata Loading (Executed once at startup)
# ---------------------------------------------------------
model = None
metadata = {}

def load_inference_artifacts():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run train_model.py first.")
    
    print(f"[{datetime.now().isoformat()}] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_name": "Generic_Classifier",
            "version": "1.0.0",
            "features": ["feature_1", "feature_2"],
            "classes": ["Class_0", "Class_1"]
        }
    print(f"[{datetime.now().isoformat()}] Model successfully loaded into memory.")

try:
    load_inference_artifacts()
except Exception as e:
    print(f"Warning: Model could not be loaded at boot: {e}")

# ---------------------------------------------------------
# CORS & Header Middleware
# ---------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ---------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "status": "error",
        "code": 400,
        "message": getattr(error, "description", "Bad Request")
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "code": 404,
        "message": "Endpoint not found. Consult GET / for available routes."
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "code": 405,
        "message": "HTTP Method Not Allowed for this route."
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "code": 500,
        "message": "Internal Server Error during inference execution."
    }), 500

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Welcome and API documentation index."""
    return jsonify({
        "service": "Edge ML Prediction REST API",
        "description": "Flask inference microservice for Machine Learning on NVIDIA Jetson & Linux hosts",
        "version": metadata.get("version", "1.0.0"),
        "model_name": metadata.get("model_name", "Unknown"),
        "endpoints": {
            "GET /": "API summary and available routes",
            "GET /health": "Liveness and system resource health check",
            "GET /info": "Model architecture, feature schema and target classes",
            "POST /predict": "Submit sensor/sample payload for real-time inference"
        },
        "sample_predict_payload": {
            "temperature": 66.4,
            "vibration": 1.48,
            "pressure": 3.98,
            "rotational_speed": 1802.0,
            "voltage": 230.2
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """System health, uptime, memory, and model status."""
    uptime_sec = round(time.time() - START_TIME, 2)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "uptime_seconds": uptime_sec,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "total_system_memory_used_percent": psutil.virtual_memory().percent
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200

@app.route("/info", methods=["GET"])
def model_info():
    """Detailed model metadata and expected schema."""
    return jsonify({
        "status": "success",
        "metadata": metadata
    }), 200

def _parse_and_validate_sample(item, expected_features):
    """Helper to validate that all expected features are present and numeric."""
    vector = []
    missing_features = []
    invalid_types = []

    for feat in expected_features:
        if feat not in item:
            missing_features.append(feat)
        else:
            val = item[feat]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                invalid_types.append({feat: str(val)})
            else:
                vector.append(float(val))

    if missing_features:
        raise ValueError(f"Missing required feature(s): {', '.join(missing_features)}")
    if invalid_types:
        raise TypeError(f"Invalid non-numeric values for feature(s): {invalid_types}")

    return vector

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Supports:
      1. Single item JSON: {"temperature": 65.0, "vibration": 1.5, ...}
      2. Wrapped single item: {"data": {...}}
      3. Batch list JSON: [{"temperature": 65.0, ...}, {...}]
      4. Batch object JSON: {"batch": [{...}, {...}]}
      5. Raw feature array: {"features": [65.0, 1.5, 4.0, 1800.0, 230.0]}
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded on server."
        }), 503

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request Content-Type must be 'application/json'"
        }), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "status": "error",
            "message": "Malformed or empty JSON body."
        }), 400

    expected_features = metadata.get("features", [])
    classes = metadata.get("classes", [])

    is_batch = False
    samples_to_process = []

    try:
        # Determine payload shape
        if isinstance(payload, list):
            is_batch = True
            if len(payload) == 0:
                return jsonify({"status": "error", "message": "Batch array cannot be empty."}), 400
            for i, item in enumerate(payload):
                if isinstance(item, dict):
                    samples_to_process.append(_parse_and_validate_sample(item, expected_features))
                elif isinstance(item, list) and len(item) == len(expected_features):
                    samples_to_process.append([float(x) for x in item])
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Invalid format at index {i} in batch."
                    }), 400

        elif isinstance(payload, dict):
            if "batch" in payload and isinstance(payload["batch"], list):
                is_batch = True
                if len(payload["batch"]) == 0:
                    return jsonify({"status": "error", "message": "Batch array cannot be empty."}), 400
                for i, item in enumerate(payload["batch"]):
                    samples_to_process.append(_parse_and_validate_sample(item, expected_features))
            elif "features" in payload and isinstance(payload["features"], list):
                raw_feats = payload["features"]
                if len(raw_feats) != len(expected_features):
                    return jsonify({
                        "status": "error",
                        "message": f"Expected {len(expected_features)} features, received {len(raw_feats)}."
                    }), 400
                samples_to_process.append([float(x) for x in raw_feats])
            elif "data" in payload and isinstance(payload["data"], dict):
                samples_to_process.append(_parse_and_validate_sample(payload["data"], expected_features))
            else:
                # Direct dictionary mapping
                samples_to_process.append(_parse_and_validate_sample(payload, expected_features))
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON root type. Expected Object or Array."
            }), 400

    except (ValueError, TypeError) as val_err:
        return jsonify({
            "status": "error",
            "message": str(val_err),
            "expected_schema": expected_features
        }), 400

    # Execute Model Inference with precise timing
    start_inference = time.perf_counter()
    X = np.array(samples_to_process, dtype=np.float32)

    try:
        predictions = model.predict(X)
        has_proba = hasattr(model, "predict_proba")
        probabilities = model.predict_proba(X) if has_proba else None
    except Exception as infer_err:
        return jsonify({
            "status": "error",
            "message": f"Model inference calculation failed: {str(infer_err)}"
        }), 500

    inference_duration_ms = round((time.perf_counter() - start_inference) * 1000, 3)

    # Format Results
    results = []
    for idx, pred_idx in enumerate(predictions):
        label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        res_item = {
            "prediction": label,
            "class_id": int(pred_idx)
        }
        if probabilities is not None:
            probs = probabilities[idx]
            conf = float(np.max(probs))
            prob_dict = {
                (classes[c] if c < len(classes) else f"class_{c}"): round(float(probs[c]), 4)
                for c in range(len(probs))
            }
            res_item["confidence"] = round(conf, 4)
            res_item["probabilities"] = prob_dict
        results.append(res_item)

    response_data = {
        "status": "success",
        "model_version": metadata.get("version", "1.0.0"),
        "latency_ms": inference_duration_ms,
        "sample_count": len(results),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if is_batch:
        response_data["predictions"] = results
    else:
        response_data["result"] = results[0]

    return jsonify(response_data), 200

# ---------------------------------------------------------
# Local Server Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("=" * 60)
    print(f" Starting ML Inference API on http://{host}:{port}")
    print(f" Model: {metadata.get('model_name', 'Loaded')}")
    print(f" Features: {metadata.get('features', [])}")
    print("=" * 60)
    app.run(host=host, port=port, debug=debug)
