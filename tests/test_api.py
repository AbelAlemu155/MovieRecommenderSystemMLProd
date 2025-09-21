"""
Contract test ensuring the HTTP interface matches the course spec.
Run with:  pytest -q
"""

from fastapi.testclient import TestClient
from src.service.server import app

def test_recommend_contract():
    c = TestClient(app)
    r = c.get("/recommend/123")
    # 1) Must succeed
    assert r.status_code == 200
    # 2) Must be plain text, not JSON
    assert r.headers["content-type"].startswith("text/plain")
    # 3) Must be exactly one line of <=20 comma-separated IDs
    ids = r.text.split(",")
    assert 1 <= len(ids) <= 20
    assert all(ids)  # no empty tokens

def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and "msg" in body
