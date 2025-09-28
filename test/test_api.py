from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_volatility():
    payload = {
        "lag": 7,
        "features": [0.01, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "volatility_forecast" in response.json()

def test_predict_from_prices():
    payload = {
        "lag": 7,
        "prices": [30000, 30200, 30100, 30500, 30750, 30800, 31000, 31200, 31300, 31400, 31500, 31600, 31700, 31800],
        "method": "rolling"
    }
    response = client.post("/predict-from-prices", json=payload)
    assert response.status_code == 200
    assert "volatility_forecast" in response.json()
    assert "volatility_input" in response.json()

def test_model_not_found():
    payload = {
        "lag": 99,
        "features": [0.01] * 99
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404