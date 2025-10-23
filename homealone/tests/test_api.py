# tests/test_api.py
from fastapi.testclient import TestClient
from homealone.service.server import app

def test_contract():
    c = TestClient(app)
    r = c.get("/recommend/1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert 1 <= len(r.text.split(",")) <= 20
