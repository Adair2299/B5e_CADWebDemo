
# Test FastAPI endpoints for health, prediction, execution, and errors.

import json
import pytest
from fastapi.testclient import TestClient
import main
from executor import Executor

client = TestClient(main.app)

@pytest.fixture(autouse=True)
def reset_executor(monkeypatch):
    monkeypatch.setattr(main, "executor", Executor())

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_initial_state():
    response = client.get("/state")
    assert response.status_code == 200
    assert response.json()["objects"] == []

def test_predict(monkeypatch):
    command = {"command": "CREATE_CIRCLE", "x": 0, "y": 0, "radius": 5}
    monkeypatch.setattr(main.predictor, "predict", lambda text: json.dumps(command))
    response = client.post("/predict", json={"text": "Draw a circle"})
    assert response.status_code == 200
    assert response.json()["command"] == command

def test_predict_invalid(monkeypatch):
    def invalid_prediction(text):
        raise ValueError("Invalid prediction")
    monkeypatch.setattr(main.predictor, "predict", invalid_prediction)
    response = client.post("/predict", json={"text": "invalid"})
    assert response.status_code == 400

def test_execute():
    command = {"command": "CREATE_CIRCLE", "x": 0, "y": 0, "radius": 5}
    response = client.post("/execute", json={"command": command})
    assert response.status_code == 200
    assert len(response.json()["objects"]) == 1
    assert len(client.get("/state").json()["objects"]) == 1

def test_execute_invalid():
    command = {"command": "CREATE_CIRCLE", "x": 0, "y": 0, "radius": -5}
    response = client.post("/execute", json={"command": command})
    assert response.status_code == 400
    assert client.get("/state").json()["objects"] == []

def test_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
