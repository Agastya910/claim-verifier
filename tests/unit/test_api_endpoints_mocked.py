import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.routes import app, get_db

"""
Unit Test: API Endpoints (Mocked)
This test verifies the API routes behave correctly given mocked DB responses.
Requires:
- No external dependencies
When to use it:
- Run this when modifying API routes or request/response schemas.
- Run in CI/CD pipelines.
"""

client = TestClient(app)

# Mock DB dependency
mock_db_session = MagicMock()

def override_get_db():
    try:
        yield mock_db_session
    finally:
        pass

app.dependency_overrides[get_db] = override_get_db

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_companies():
    # Setup mock return values for the two execute calls
    mock_result_trans = MagicMock()
    mock_result_trans.scalars().all.return_value = ["AAPL"]
    
    mock_result_fin = MagicMock()
    mock_result_fin.scalars().all.return_value = ["NVDA"]
    
    mock_db_session.execute.side_effect = [mock_result_trans, mock_result_fin]
    
    response = client.get("/api/companies")
    assert response.status_code == 200
    assert "AAPL" in response.json()["companies"]
    assert "NVDA" in response.json()["companies"]

def test_ingest_trigger():
    payload = {"ticker": "MSFT", "quarters": [[2023, 4]]}
    with patch("src.api.routes.BackgroundTasks.add_task") as mock_add_task:
        response = client.post("/api/ingest", json=payload)
        assert response.status_code == 200
        assert "triggered" in response.json()["message"]
        assert mock_add_task.called
