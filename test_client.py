"""
CLI Test Client for ML Prediction Flask API.
Demonstrates calling endpoints, inspecting responses, testing single and batch predictions.
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5001"

def make_request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            res_code = response.getcode()
            res_data = json.loads(response.read().decode("utf-8"))
            return res_code, res_data
    except urllib.error.HTTPError as e:
        err_data = json.loads(e.read().decode("utf-8"))
        return e.code, err_data
    except urllib.error.URLError as e:
        print(f"Error connecting to {url}: {e.reason}")
        print("Make sure 'python3 app.py' is running in another terminal window.")
        sys.exit(1)

def main():
    print("=" * 65)
    print(" 🚀 ML INFERENCE REST API - EDGE TEST SUITE")
    print("=" * 65)

    # 1. Health Check
    print("\n[1] Checking Service Health (GET /health)...")
    code, res = make_request("/health")
    print(f"Status {code}: Status={res.get('status')}, Model Loaded={res.get('model_loaded')}, Memory RSS={res.get('system', {}).get('memory_rss_mb')} MB")

    # 2. Model Metadata
    print("\n[2] Fetching Model Info (GET /info)...")
    code, res = make_request("/info")
    meta = res.get("metadata", {})
    print(f"Status {code}: Model='{meta.get('model_name')}', Version={meta.get('version')}, Classes={meta.get('classes')}")
    print(f"Features: {meta.get('features')}")

    # 3. Single Prediction (Normal State)
    print("\n[3] Single Sample Prediction (Normal Case)...")
    normal_payload = {
        "temperature": 66.2,
        "vibration": 1.45,
        "pressure": 4.02,
        "rotational_speed": 1805.0,
        "voltage": 230.1
    }
    code, res = make_request("/predict", method="POST", data=normal_payload)
    print(f"Status {code}: Result={res.get('result', {}).get('prediction')} (Confidence: {res.get('result', {}).get('confidence') * 100:.1f}%) | Latency: {res.get('latency_ms')} ms")

    # 4. Single Prediction (Critical Anomaly)
    print("\n[4] Single Sample Prediction (Critical State)...")
    critical_payload = {
        "temperature": 112.5,
        "vibration": 8.7,
        "pressure": 2.1,
        "rotational_speed": 1420.0,
        "voltage": 198.0
    }
    code, res = make_request("/predict", method="POST", data=critical_payload)
    print(f"Status {code}: Result={res.get('result', {}).get('prediction')} (Confidence: {res.get('result', {}).get('confidence') * 100:.1f}%) | Latency: {res.get('latency_ms')} ms")

    # 5. Batch Prediction
    print("\n[5] Batch Prediction (3 Samples)...")
    batch_payload = [
        {"temperature": 65.0, "vibration": 1.5, "pressure": 4.0, "rotational_speed": 1800.0, "voltage": 230.0},
        {"temperature": 84.0, "vibration": 3.9, "pressure": 3.4, "rotational_speed": 1710.0, "voltage": 220.0},
        {"temperature": 105.0, "vibration": 7.5, "pressure": 2.5, "rotational_speed": 1500.0, "voltage": 205.0}
    ]
    code, res = make_request("/predict", method="POST", data=batch_payload)
    preds = res.get("predictions", [])
    print(f"Status {code}: Processed {res.get('sample_count')} samples in {res.get('latency_ms')} ms:")
    for i, p in enumerate(preds):
        print(f"  • Sample #{i+1}: {p['prediction']} (Confidence: {p['confidence'] * 100:.1f}%)")

    # 6. Error Handling (Missing Features)
    print("\n[6] Validation Error Handling (Missing Keys)...")
    bad_payload = {"temperature": 65.0, "vibration": 1.5}
    code, res = make_request("/predict", method="POST", data=bad_payload)
    print(f"Status {code}: Expected Error Captured -> {res.get('message')}")

    print("\n" + "=" * 65)
    print(" ✅ All tests completed successfully.")
    print("=" * 65)

if __name__ == "__main__":
    main()
