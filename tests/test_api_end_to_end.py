"""End-to-end test for full map generation via the API."""
import time
import pytest
import requests


@pytest.fixture
def api_url():
    """Base URL for the running API server."""
    return "http://localhost:8000"


def test_api_full_map_generation_end_to_end(api_url: str) -> None:
    """Test generating a complete map via the API endpoints."""
    
    # Test 1: Check API health
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert health_data["database"] == "connected"
    
    # Test 2: Start map generation with single seed
    generation_request = {
        "seed": "123456789",  # Single seed used throughout like FMG
        "width": 1200,
        "height": 1000,
        "cells_desired": 10000,
        "template_name": "continents",
        "map_name": "API End-to-End Test Map"
    }
    
    response = requests.post(f"{api_url}/maps/generate", json=generation_request)
    assert response.status_code == 200
    
    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["status"] == "pending"
    assert job_data["progress_percent"] == 0
    assert job_data["message"] == "Map generation job started"
    
    job_id = job_data["job_id"]
    
    # Test 3: Poll job status until completion
    max_wait_time = 120  # 2 minutes max wait time for smaller map
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        response = requests.get(f"{api_url}/jobs/{job_id}")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["job_id"] == job_id
        
        print(f"Job status: {status_data['status']}, Progress: {status_data['progress_percent']}%")
        
        if status_data["status"] == "completed":
            assert status_data["progress_percent"] == 100
            assert status_data["map_id"] is not None
            map_id = status_data["map_id"]
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Map generation failed: {status_data.get('error_message', 'Unknown error')}")
        
        # Wait before next poll
        time.sleep(5)
    else:
        pytest.fail(f"Map generation timed out after {max_wait_time} seconds")
    
    # Test 4: Retrieve generated map details
    response = requests.get(f"{api_url}/maps/{map_id}")
    assert response.status_code == 200
    
    map_data = response.json()
    assert map_data["id"] == map_id
    assert map_data["name"] == "API End-to-End Test Map"
    assert map_data["seed"] == "123456789"
    assert map_data["width"] == 1200.0
    assert map_data["height"] == 1000.0
    assert map_data["cells_count"] > 0
    assert map_data["generation_time_seconds"] is not None
    
    # Test 5: List maps includes our generated map
    response = requests.get(f"{api_url}/maps")
    assert response.status_code == 200
    
    maps_list = response.json()
    assert isinstance(maps_list, list)
    assert len(maps_list) > 0
    
    # Find our map in the list
    our_map = next((m for m in maps_list if m["id"] == map_id), None)
    assert our_map is not None
    assert our_map["name"] == "API End-to-End Test Map"
    
    print(f"✓ Map generated successfully: {map_id}")
    print(f"✓ Map has {map_data['cells_count']} cells")
    print(f"✓ Generation took {map_data['generation_time_seconds']:.2f} seconds")


def test_api_basic_endpoints(api_url: str) -> None:
    """Test basic API endpoints without full generation."""
    
    # Test root endpoint
    response = requests.get(f"{api_url}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Fantasy Map Generator API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"
    
    # Test invalid job ID
    response = requests.get(f"{api_url}/jobs/invalid-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"
    
    # Test invalid map ID
    response = requests.get(f"{api_url}/maps/invalid-map-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Map not found"


def test_api_validation(api_url: str) -> None:
    """Test API request validation."""
    
    # Test invalid width (too small)
    response = requests.post(f"{api_url}/maps/generate", json={
        "width": 50,  # Below minimum of 100
        "height": 300,
        "cells_desired": 2000
    })
    assert response.status_code == 422
    
    # Test invalid cells_desired (too small)
    response = requests.post(f"{api_url}/maps/generate", json={
        "width": 400,
        "height": 300, 
        "cells_desired": 500  # Below minimum of 1000
    })
    assert response.status_code == 422
    
    # Test invalid width (too large)
    response = requests.post(f"{api_url}/maps/generate", json={
        "width": 3000,  # Above maximum of 2000
        "height": 300,
        "cells_desired": 2000
    })
    assert response.status_code == 422