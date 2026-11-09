from fastapi.testclient import TestClient
from app import app

def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_chat():
    r = TestClient(app).post("/chat", json={"message":"Hello", "max_new_tokens":3})
    assert r.status_code == 200
    assert "response" in r.json()
