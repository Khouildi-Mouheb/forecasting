from datetime import date, timedelta
from fastapi.testclient import TestClient
from forecasting.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_forecast_endpoint():
    base_date = date(2026, 1, 1)
    history = [
        {"date": (base_date + timedelta(days=i)).isoformat(), "quantity": float(20 + i)}
        for i in range(25)
    ]

    payload = {
        "itemId": "JCD_EXTRA_VIRGIN_5L",
        "horizon": 7,
        "modelType": "Prophet",
        "history": history,
    }

    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["itemId"] == "JCD_EXTRA_VIRGIN_5L"
    assert len(res_data["predictedValues"]) == 7
    assert res_data["modelUsed"] == "Prophet"
    assert "confidenceInterval" in res_data
    assert len(res_data["confidenceInterval"]["lower"]) == 7
