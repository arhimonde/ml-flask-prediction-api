# 🧠 Edge ML Flask Prediction REST API

API REST ligera y robusta basada en **Flask** y **Joblib / Scikit-Learn** para servir predicciones de Machine Learning en tiempo real. Diseñada específicamente para dispositivos Edge como **NVIDIA Jetson** (Nano, Xavier, Orin) y servidores Linux.

---

## 📐 Arquitectura del Patrón `/predict`

```
┌─────────────────────────┐          HTTP POST /predict (JSON)        ┌───────────────────────────────┐
│     Cliente Externo     │ ────────────────────────────────────────> │       Flask REST API          │
│ (App Web, ROS2, Cámara, │ <──────────────────────────────────────── │    (Servidor en Jetson)       │
│  Dashboard IoT, Script) │          JSON con Predicciones y Métricas  └──────────────┬────────────────┘
└─────────────────────────┘                                                          │
                                                                       ┌─────────────▼────────────────┐
                                                                       │ Modelo Cargado en Memoria    │
                                                                       │ (Pipeline Joblib en RAM)     │
                                                                       │   • StandardScaler           │
                                                                       │   • Random Forest Classifier │
                                                                       └──────────────────────────────┘
```

### 💡 Principios Clave de Diseño para Edge / Jetson:
1. **Carga única en memoria (Zero-overhead per request):** El modelo entrenado se deserializa una sola vez al arrancar la aplicación (`app.py`), evitando lecturas repetidas de disco que agotarían la memoria e incrementarían la latencia.
2. **Pipeline integrado:** El escalado de datos (`StandardScaler`) y el clasificador están empaquetados juntos en el pipeline exportado con `joblib`, evitando discrepancias de preprocesamiento.
3. **Validación estricta de esquema:** Chequeo de campos faltantes y tipos numéricos con respuestas de error HTTP claras (`400 Bad Request`, `415 Media Type`).
4. **Soporte Individual y por Lotes (Batch):** Permite procesar una sola muestra o un array de múltiples lecturas simultáneamente.
5. **CORS Habilitado:** Para consumir directamente desde paneles web / interfaces frontend en el navegador.

---

## 📁 Estructura del Proyecto

```
ml_flask_prediction_api/
├── app.py                     # Servidor Flask REST con endpoints (/health, /info, /predict)
├── train_model.py             # Entrenamiento del modelo y exportación con Joblib
├── test_client.py             # Cliente de prueba interactivo en Python
├── requirements.txt           # Dependencias del proyecto
├── models/
│   ├── sensor_classifier.joblib  # Modelo entrenado serializado (comprimido)
│   └── model_metadata.json       # Metadatos del modelo, versiones y esquema
├── tests/
│   └── test_api.py            # Suite de pruebas unitarias automatizadas
└── systemd/
    └── jetson_ml_api.service  # Configuración del servicio para auto-arranque en Jetson
```

---

## 🚀 Guía Rápida de Uso

### 1. Requisitos
```bash
pip install -r requirements.txt
```

### 2. Entrenar el Modelo (si no está entrenado)
```bash
python3 train_model.py
```
> Esto generará `models/sensor_classifier.joblib` y `models/model_metadata.json`.

### 3. Iniciar el Servidor de Desarrollo
```bash
python3 app.py
```
Por defecto se ejecuta en `http://0.0.0.0:5001`.

### 4. Ejecutar la Suite de Pruebas Unitarias
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

### 5. Probar con el Cliente de Prueba
En otra terminal:
```bash
python3 test_client.py
```

---

## 📡 Documentación de Endpoints

### 1. `GET /health` - Verificación de Estado y Recursos
Devuelve el estado de la API, si el modelo está en memoria, uptime y consumo de RAM/CPU:
```bash
curl -X GET http://127.0.0.1:5001/health
```
**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 15.3,
  "system": {
    "cpu_percent": 4.1,
    "memory_rss_mb": 48.2,
    "total_system_memory_used_percent": 32.5
  },
  "timestamp": "2026-09-27T12:26:00Z"
}
```

---

### 2. `GET /info` - Esquema y Metadatos
```bash
curl -X GET http://127.0.0.1:5001/info
```

---

### 3. `POST /predict` - Inferencia de Modelo

#### A. Inferencia Individual (Ejemplo Estado Normal):
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 66.2,
    "vibration": 1.45,
    "pressure": 4.02,
    "rotational_speed": 1805.0,
    "voltage": 230.1
  }'
```

**Respuesta (`200 OK`):**
```json
{
  "status": "success",
  "model_version": "1.0.0",
  "latency_ms": 2.45,
  "sample_count": 1,
  "result": {
    "prediction": "NORMAL",
    "class_id": 0,
    "confidence": 0.985,
    "probabilities": {
      "NORMAL": 0.985,
      "WARNING": 0.015,
      "CRITICAL": 0.0
    }
  },
  "timestamp": "2026-09-27T12:26:15Z"
}
```

#### B. Inferencia de Estado Crítico:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 112.0,
    "vibration": 8.6,
    "pressure": 2.1,
    "rotational_speed": 1420.0,
    "voltage": 198.0
  }'
```

**Respuesta (`200 OK`):**
```json
{
  "status": "success",
  "result": {
    "prediction": "CRITICAL",
    "class_id": 2,
    "confidence": 0.992
  }
}
```

#### C. Inferencia por Lotes (Batch):
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"temperature": 65.0, "vibration": 1.5, "pressure": 4.0, "rotational_speed": 1800.0, "voltage": 230.0},
    {"temperature": 83.0, "vibration": 3.8, "pressure": 3.4, "rotational_speed": 1720.0, "voltage": 222.0},
    {"temperature": 105.0, "vibration": 7.5, "pressure": 2.5, "rotational_speed": 1500.0, "voltage": 205.0}
  ]'
```

---

## ⚡ Despliegue en Producción en NVIDIA Jetson

Para entornos de producción o ejecución 24/7 en NVIDIA Jetson (Jetson Nano, Xavier NX, AGX Orin):

### 1. Ejecutar con WSGI (Gunicorn)
El servidor de desarrollo integrado de Flask es mono-hilo y no apto para producción. En Jetson, se recomienda **Gunicorn**:
```bash
gunicorn --workers 2 --bind 0.0.0.0:5001 --timeout 60 app:app
```
> **Nota para Jetson:** Dado que los núcleos de CPU y la memoria unificada (UMA) son limitados, 2 workers (`--workers 2`) ofrecen una excelente concurrencia sin sobrecargar la RAM del sistema.

### 2. Configurar como Servicio Systemd (Autoarranque)
Copia el archivo `systemd/jetson_ml_api.service` a `/etc/systemd/system/`:
```bash
sudo cp systemd/jetson_ml_api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jetson_ml_api
sudo systemctl start jetson_ml_api
sudo systemctl status jetson_ml_api
```

---

## 🔌 Integración con Otras Aplicaciones

### Ejemplo en Python (Script / Nodo ROS2):
```python
import requests

payload = {
    "temperature": 67.0,
    "vibration": 1.6,
    "pressure": 3.95,
    "rotational_speed": 1795.0,
    "voltage": 229.8
}

res = requests.post("http://192.168.1.50:5001/predict", json=payload)
data = res.json()
print("Diagnóstico:", data["result"]["prediction"])
```

### Ejemplo en JavaScript / Frontend (Fetch):
```javascript
const response = await fetch("http://192.168.1.50:5001/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    temperature: 67.0,
    vibration: 1.6,
    pressure: 3.95,
    rotational_speed: 1795.0,
    voltage: 229.8
  })
});
const result = await response.json();
console.log("Prediction:", result.result.prediction);
```
