from unittest.mock import patch, MagicMock
import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ── Smoke test ─────────────────────────────────────────────
def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Unit test: decode de imagem ────────────────────────────
def test_decode_image_valid_base64():
    from app.main import _decode_image

    # Cria imagem 10x10 px branca e codifica em base64
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode()

    result = _decode_image(b64)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)


# ── Integration test: inferência completa ────────────────── 
def test_predict_returns_detections():
    # 1. Cria uma imagem falsa em memória para o payload do teste
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

     # 2. Configura a estrutura simulada do retorno do YOLO
    mock_box = MagicMock()
    mock_box.xyxy = [[10.0, 20.0, 50.0, 60.0]]
    mock_box.cls = [0]
    mock_box.conf = [0.95]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model_instance = MagicMock()
    mock_model_instance.return_value = [mock_result]
    mock_model_instance.names = {0: "person"}

    with patch("app.main.load_model", return_value=mock_model_instance):
        response = client.post(
            "/predict",
            json={
                "image_base64": b64,
                "confidence": 0.3,
                "model_name": "yolov8n.pt",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["label"] == "person"
    assert data["inference_ms"] >= 0