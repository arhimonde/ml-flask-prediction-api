"""
Automated Unit and Integration Tests for ML Prediction Flask API.
Verifies all endpoints, schemas, validation errors, single and batch predictions.
"""

import unittest
import json
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, model, metadata

class FlaskMLApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_root_index(self):
        """Test GET / documentation route."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("endpoints", data)
        self.assertIn("model_name", data)
        self.assertIn("sample_predict_payload", data)

    def test_health_check(self):
        """Test GET /health returns 200 and valid system telemetry."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("uptime_seconds", data)
        self.assertIn("system", data)

    def test_info_metadata(self):
        """Test GET /info returns feature schema and classes."""
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("features", data["metadata"])
        self.assertIn("classes", data["metadata"])

    def test_predict_single_normal(self):
        """Test single prediction on nominal operational parameters."""
        payload = {
            "temperature": 65.0,
            "vibration": 1.5,
            "pressure": 4.0,
            "rotational_speed": 1800.0,
            "voltage": 230.0
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("result", data)
        self.assertEqual(data["result"]["prediction"], "NORMAL")
        self.assertGreater(data["result"]["confidence"], 0.7)
        self.assertIn("latency_ms", data)

    def test_predict_single_critical(self):
        """Test prediction on severe telemetry anomaly (bearing failure)."""
        payload = {
            "temperature": 110.0,
            "vibration": 8.5,
            "pressure": 2.2,
            "rotational_speed": 1450.0,
            "voltage": 195.0
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["result"]["prediction"], "CRITICAL")
        self.assertGreater(data["result"]["confidence"], 0.8)

    def test_predict_batch_array(self):
        """Test batch prediction array."""
        payload = [
            {"temperature": 64.0, "vibration": 1.4, "pressure": 4.1, "rotational_speed": 1810.0, "voltage": 231.0},
            {"temperature": 83.0, "vibration": 3.9, "pressure": 3.3, "rotational_speed": 1715.0, "voltage": 221.0},
            {"temperature": 108.0, "vibration": 7.9, "pressure": 2.4, "rotational_speed": 1490.0, "voltage": 200.0}
        ]
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["sample_count"], 3)
        self.assertEqual(len(data["predictions"]), 3)
        self.assertEqual(data["predictions"][0]["prediction"], "NORMAL")
        self.assertEqual(data["predictions"][1]["prediction"], "WARNING")
        self.assertEqual(data["predictions"][2]["prediction"], "CRITICAL")

    def test_predict_missing_feature(self):
        """Test that missing required feature returns 400 Bad Request."""
        payload = {
            "temperature": 65.0,
            "vibration": 1.5,
            # missing pressure, rotational_speed, voltage
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Missing required feature", data["message"])

    def test_predict_invalid_data_type(self):
        """Test that non-numeric value returns 400 Bad Request."""
        payload = {
            "temperature": "sixty-five",
            "vibration": 1.5,
            "pressure": 4.0,
            "rotational_speed": 1800.0,
            "voltage": 230.0
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Invalid non-numeric values", data["message"])

    def test_predict_non_json_content_type(self):
        """Test that missing or incorrect Content-Type returns 415."""
        response = self.client.post(
            "/predict",
            data="raw text string",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 415)

if __name__ == "__main__":
    unittest.main()
