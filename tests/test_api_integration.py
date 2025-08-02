"""Integration test for full map generation via API (with mocked generation)."""
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock
import uuid

import pytest
from fastapi.testclient import TestClient

from py_fmg.api.main import app


@pytest.fixture
def mock_db():
    """Mock the database completely."""
    mock_db_instance = MagicMock()
    
    # Mock get_session context manager
    mock_session = MagicMock()
    mock_db_instance.get_session.return_value.__enter__.return_value = mock_session
    mock_db_instance.get_session.return_value.__exit__.return_value = None
    
    # Mock session.execute for health check
    mock_session.execute.return_value = None
    
    # Mock job and map objects
    job_id = str(uuid.uuid4())
    map_id = str(uuid.uuid4())
    
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_job.grid_seed = "test123"
    mock_job.map_seed = "test456"
    mock_job.status = "pending"
    mock_job.progress_percent = 0
    mock_job.error_message = None
    mock_job.map_id = None
    
    mock_map = MagicMock()
    mock_map.id = map_id
    mock_map.name = "API Test Map"
    mock_map.seed = "test123"
    mock_map.grid_seed = "test123"
    mock_map.map_seed = "test456"
    mock_map.width = 400.0
    mock_map.height = 300.0
    mock_map.cells_count = 4500
    mock_map.generation_time_seconds = 1.0
    mock_map.created_at = "2025-01-01T00:00:00"
    
    # Track job status updates
    job_statuses = ["pending", "running", "completed"]
    job_progress = [0, 50, 100]
    status_call_count = [0]
    
    def mock_query_job(job_class):
        query_mock = MagicMock()
        filter_mock = MagicMock()
        
        def mock_first():
            # Simulate job progression
            idx = min(status_call_count[0], len(job_statuses) - 1)
            mock_job.status = job_statuses[idx]
            mock_job.progress_percent = job_progress[idx]
            
            if mock_job.status == "completed":
                mock_job.map_id = map_id
                
            status_call_count[0] += 1
            return mock_job
        
        filter_mock.first.side_effect = mock_first
        query_mock.filter.return_value = filter_mock
        return query_mock
    
    def mock_query_map(map_class):
        query_mock = MagicMock()
        filter_mock = MagicMock()
        filter_mock.first.return_value = mock_map
        query_mock.filter.return_value = filter_mock
        query_mock.order_by.return_value.all.return_value = [mock_map]
        return query_mock
    
    def mock_query(model_class):
        if hasattr(model_class, '__name__') and 'Job' in model_class.__name__:
            return mock_query_job(model_class)
        else:
            return mock_query_map(model_class)
    
    mock_session.query.side_effect = mock_query
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    mock_session.flush.return_value = None
    
    return mock_db_instance


@pytest.fixture
def mock_generation():
    """Mock the map generation process."""
    async def mock_run_map_generation(job_id: str, request) -> None:
        # Simulate fast generation for testing
        pass
    
    return mock_run_map_generation


@pytest.fixture
def client(mock_db, mock_generation):
    """Create a test client with mocked dependencies."""
    with patch('py_fmg.api.main.db', mock_db), \
         patch('py_fmg.api.main.run_map_generation', mock_generation):
        return TestClient(app)


def test_api_health_check(client: TestClient) -> None:
    """Test API health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_api_root_endpoint(client: TestClient) -> None:
    """Test API root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Fantasy Map Generator API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_api_full_map_generation_flow(client: TestClient) -> None:
    """Test the complete map generation flow via API."""
    
    # Step 1: Start map generation
    generation_request = {
        "grid_seed": "test123",
        "map_seed": "test456",
        "width": 400,
        "height": 300,
        "cells_desired": 2000,
        "template_name": "default",
        "map_name": "API Test Map"
    }
    
    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200
    
    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["status"] == "pending"
    assert job_data["progress_percent"] == 0
    assert job_data["message"] == "Map generation job started"
    
    job_id = job_data["job_id"]
    
    # Step 2: Check job status (should be running)
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    
    status_data = response.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ["pending", "running"]
    
    # Step 3: Check job status again (should be completed)
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    
    status_data = response.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "completed"
    assert status_data["progress_percent"] == 100
    assert status_data["map_id"] is not None
    
    map_id = status_data["map_id"]
    
    # Step 4: Retrieve generated map details
    response = client.get(f"/maps/{map_id}")
    assert response.status_code == 200
    
    map_data = response.json()
    assert map_data["id"] == map_id
    assert map_data["name"] == "API Test Map"
    assert map_data["grid_seed"] == "test123"
    assert map_data["map_seed"] == "test456"
    assert map_data["width"] == 400.0
    assert map_data["height"] == 300.0
    assert map_data["cells_count"] > 0
    
    # Step 5: List maps includes our generated map
    response = client.get("/maps")
    assert response.status_code == 200
    
    maps_list = response.json()
    assert isinstance(maps_list, list)
    assert len(maps_list) >= 1
    
    # Find our map in the list
    our_map = next((m for m in maps_list if m["id"] == map_id), None)
    assert our_map is not None
    assert our_map["name"] == "API Test Map"


def test_api_legacy_seed_generation(client: TestClient) -> None:
    """Test map generation using legacy single seed parameter."""
    generation_request = {
        "seed": "legacy789",
        "width": 200,
        "height": 150,
        "cells_desired": 1000,
        "template_name": "default"
    }
    
    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200
    
    job_data = response.json()
    assert job_data["status"] == "pending"


def test_api_validation_errors(client: TestClient) -> None:
    """Test API request validation."""
    
    # Test invalid width (too small)
    response = client.post("/maps/generate", json={
        "width": 50,  # Below minimum of 100
        "height": 300,
        "cells_desired": 2000
    })
    assert response.status_code == 422
    
    # Test invalid cells_desired (too small)
    response = client.post("/maps/generate", json={
        "width": 400,
        "height": 300,
        "cells_desired": 500  # Below minimum of 1000
    })
    assert response.status_code == 422
    
    # Test invalid width (too large)
    response = client.post("/maps/generate", json={
        "width": 3000,  # Above maximum of 2000
        "height": 300,
        "cells_desired": 2000
    })
    assert response.status_code == 422


def test_api_job_not_found(client: TestClient) -> None:
    """Test requesting status for non-existent job."""
    response = client.get("/jobs/nonexistent-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_api_map_not_found(client: TestClient) -> None:
    """Test requesting non-existent map."""
    response = client.get("/maps/nonexistent-map-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Map not found"


@pytest.mark.parametrize("template_name", ["default", "small_continent", "archipelago"])
def test_api_different_templates(client: TestClient, template_name: str) -> None:
    """Test map generation with different templates."""
    generation_request = {
        "grid_seed": f"template_{template_name}",
        "map_seed": f"map_{template_name}",
        "width": 300,
        "height": 200,
        "cells_desired": 1500,
        "template_name": template_name,
        "map_name": f"Template Test - {template_name}"
    }
    
    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200
    
    job_data = response.json()
    assert job_data["status"] == "pending"