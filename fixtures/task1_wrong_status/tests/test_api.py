"""Tests for the Task 1 FastAPI app."""

import pytest
from fastapi.testclient import TestClient
from app.main import app, db

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_around_tests():
    """Clear the database before each test."""
    db.clear()
    yield

def test_create_item_status():
    """Verify that POST /api/items returns 201 Created."""
    response = client.post("/api/items", json={"name": "New Item"})
    # This should fail if the BUG is present
    assert response.status_code == 201, f"Expected 201 Created for resource creation, got {response.status_code}"
    assert "id" in response.json()
    assert response.json()["item"]["name"] == "New Item"

def test_get_item():
    """Verify that GET /api/items/{item_id} returns an item."""
    # First, create an item
    client.post("/api/items", json={"name": "Persistent Item"})
    response = client.get("/api/items/1")
    assert response.status_code == 200
    assert response.json()["name"] == "Persistent Item"
