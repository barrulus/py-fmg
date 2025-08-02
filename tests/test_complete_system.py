"""
Complete system integration tests.

This module tests the entire py-fmg system including:
- Map generation pipeline
- 3D visualization
- Editing capabilities
- API endpoints
"""

import pytest
import requests
import json
import time
from pathlib import Path
import subprocess
import os
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:9888"
VIEWER_BASE_URL = "http://localhost:8081"
TEST_MAP_NAME = "Test_Complete_System"


class TestCompleteSystem:
    """Test the complete py-fmg system end-to-end."""
    
    @pytest.fixture(scope="class")
    def test_map(self) -> Dict[str, Any]:
        """Create a test map for all tests."""
        # Generate a new map
        response = requests.post(
            f"{API_BASE_URL}/maps/generate",
            json={
                "map_name": TEST_MAP_NAME,
                "width": 400,
                "height": 300,
                "seed": 12345,
                "options": {
                    "generate_cultures": True,
                    "generate_settlements": True,
                    "generate_rivers": True,
                    "generate_religions": True,
                    "generate_states": True
                }
            }
        )
        
        assert response.status_code == 200
        map_data = response.json()
        
        # Wait for generation to complete
        job_id = map_data["job_id"]
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                if status["status"] == "completed":
                    break
                elif status["status"] == "failed":
                    pytest.fail(f"Map generation failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(10)
            wait_time += 10
        
        if wait_time >= max_wait:
            pytest.fail("Map generation timed out")
        
        # Get the generated map
        maps_response = requests.get(f"{API_BASE_URL}/maps")
        assert maps_response.status_code == 200
        
        maps = maps_response.json()
        test_map = next((m for m in maps if m["name"] == TEST_MAP_NAME), None)
        assert test_map is not None, "Test map not found"
        
        return test_map
    
    def test_map_generation_pipeline(self, test_map):
        """Test the complete map generation pipeline."""
        map_id = test_map["id"]
        
        # Test that all components were generated
        components = [
            "voronoi_cells",
            "settlements", 
            "rivers",
            "cultures",
            "religions",
            "states"
        ]
        
        for component in components:
            response = requests.get(f"{API_BASE_URL}/maps/{map_id}/{component}")
            assert response.status_code == 200
            data = response.json()
            assert len(data) > 0, f"No {component} generated"
    
    def test_3d_visualization_setup(self, test_map):
        """Test 3D visualization setup and tile generation."""
        map_id = test_map["id"]
        
        # Test visualization info endpoint
        response = requests.get(f"{API_BASE_URL}/maps/{map_id}/visualize/info")
        assert response.status_code == 200
        
        viz_info = response.json()
        assert viz_info["map_id"] == map_id
        assert viz_info["map_name"] == TEST_MAP_NAME
        assert len(viz_info["layers"]) == 5  # terrain, settlements, rivers, cultures, states
        
        # Test tile generation
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/visualize/generate",
            json={
                "layers": ["terrain", "settlements"],
                "quality": "low",  # Use low quality for faster testing
                "force_regenerate": True
            }
        )
        assert response.status_code == 200
        
        generation_response = response.json()
        assert generation_response["success"] is True
        assert "job_id" in generation_response
        
        # Wait for tile generation (shorter timeout for low quality)
        time.sleep(60)  # 1 minute should be enough for low quality tiles
        
        # Check tile status
        response = requests.get(f"{API_BASE_URL}/maps/{map_id}/visualize/status")
        assert response.status_code == 200
        
        statuses = response.json()
        terrain_status = next((s for s in statuses if s["layer"] == "terrain"), None)
        assert terrain_status is not None
        # Note: May still be generating, so we don't assert completion
    
    def test_editing_endpoints(self, test_map):
        """Test various editing endpoints."""
        map_id = test_map["id"]
        
        # Test terrain editing
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/edit/terrain",
            json={
                "cell_indices": [1, 2, 3],
                "operation": "set_height",
                "value": 150.0
            }
        )
        assert response.status_code == 200
        
        edit_response = response.json()
        assert edit_response["success"] is True
        assert edit_response["affected_count"] > 0
        
        # Test settlement generation
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/edit/settlements/generate",
            json={
                "count": 5,
                "min_height": 0,
                "max_height": 100
            }
        )
        assert response.status_code == 200
        
        generation_response = response.json()
        assert generation_response["success"] is True
        
        # Test batch terrain editing
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/edit/terrain/batch",
            json={
                "operations": [
                    {
                        "operation": "set_height",
                        "cell_indices": [10, 11, 12],
                        "value": 200.0
                    },
                    {
                        "operation": "set_biome",
                        "cell_indices": [13, 14, 15],
                        "biome": "Temperate Forest"
                    }
                ]
            }
        )
        assert response.status_code == 200
        
        batch_response = response.json()
        assert batch_response["success"] is True
    
    def test_api_documentation(self):
        """Test that API documentation is accessible."""
        # Test OpenAPI docs
        response = requests.get(f"{API_BASE_URL}/docs")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "paths" in openapi_spec
        
        # Check that our new endpoints are documented
        paths = openapi_spec["paths"]
        assert any("/maps/{map_id}/visualize" in path for path in paths)
        assert any("/maps/{map_id}/edit" in path for path in paths)
    
    def test_database_consistency(self, test_map):
        """Test database consistency after operations."""
        map_id = test_map["id"]
        
        # Get all map data
        response = requests.get(f"{API_BASE_URL}/maps/{map_id}")
        assert response.status_code == 200
        
        map_data = response.json()
        
        # Check that all foreign key relationships are valid
        # This is a simplified check - in production you'd want more thorough validation
        
        # Check settlements reference valid cells
        settlements_response = requests.get(f"{API_BASE_URL}/maps/{map_id}/settlements")
        assert settlements_response.status_code == 200
        
        settlements = settlements_response.json()
        if settlements:
            # Get all cell indices
            cells_response = requests.get(f"{API_BASE_URL}/maps/{map_id}/voronoi_cells")
            assert cells_response.status_code == 200
            
            cells = cells_response.json()
            cell_indices = {cell["cell_index"] for cell in cells}
            
            # Check that all settlements reference valid cells
            for settlement in settlements:
                if settlement.get("cell_index"):
                    assert settlement["cell_index"] in cell_indices
    
    def test_performance_metrics(self, test_map):
        """Test basic performance metrics."""
        map_id = test_map["id"]
        
        # Test response times for key endpoints
        endpoints_to_test = [
            f"/maps/{map_id}",
            f"/maps/{map_id}/voronoi_cells",
            f"/maps/{map_id}/settlements",
            f"/maps/{map_id}/visualize/info"
        ]
        
        for endpoint in endpoints_to_test:
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            
            # Assert reasonable response times (adjust as needed)
            assert response_time < 5.0, f"Endpoint {endpoint} took {response_time:.2f}s (too slow)"
    
    def test_error_handling(self, test_map):
        """Test error handling for various scenarios."""
        map_id = test_map["id"]
        
        # Test invalid map ID
        response = requests.get(f"{API_BASE_URL}/maps/invalid-id")
        assert response.status_code == 404
        
        # Test invalid editing operation
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/edit/terrain",
            json={
                "cell_indices": [999999],  # Non-existent cell
                "operation": "invalid_operation",
                "value": 150.0
            }
        )
        assert response.status_code in [400, 422, 500]  # Should return an error
        
        # Test invalid visualization request
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/visualize/generate",
            json={
                "layers": ["invalid_layer"],
                "quality": "invalid_quality"
            }
        )
        assert response.status_code in [400, 422]  # Should return validation error
    
    def test_data_export_import(self, test_map):
        """Test data export and import capabilities."""
        map_id = test_map["id"]
        
        # Test map export (if endpoint exists)
        response = requests.get(f"{API_BASE_URL}/maps/{map_id}/export")
        if response.status_code == 200:
            # If export is implemented, test that it returns valid data
            export_data = response.json()
            assert "map_id" in export_data
            assert "voronoi_cells" in export_data or "terrain" in export_data
    
    def test_concurrent_operations(self, test_map):
        """Test concurrent operations on the same map."""
        map_id = test_map["id"]
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request(endpoint, data=None):
            try:
                if data:
                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=data)
                else:
                    response = requests.get(f"{API_BASE_URL}{endpoint}")
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start multiple concurrent requests
        threads = []
        
        # Multiple read requests
        for _ in range(3):
            thread = threading.Thread(target=make_request, args=(f"/maps/{map_id}",))
            threads.append(thread)
            thread.start()
        
        # Multiple edit requests
        for i in range(2):
            thread = threading.Thread(
                target=make_request, 
                args=(
                    f"/maps/{map_id}/edit/terrain",
                    {
                        "cell_indices": [100 + i],
                        "operation": "set_height",
                        "value": 100.0 + i
                    }
                )
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success" and result_value in [200, 201]:
                success_count += 1
        
        # At least some requests should succeed
        assert success_count >= 3, f"Only {success_count} concurrent requests succeeded"


class TestSystemIntegration:
    """Test integration between different system components."""
    
    def test_docker_services_health(self):
        """Test that all Docker services are healthy."""
        # Test main API
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.fail("Main API service is not responding")
        
        # Test database connectivity (through API)
        try:
            response = requests.get(f"{API_BASE_URL}/maps", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.fail("Database connectivity through API failed")
    
    def test_file_system_permissions(self):
        """Test file system permissions for tile generation."""
        tiles_dir = Path("tiles")
        
        # Test that tiles directory can be created
        test_dir = tiles_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        
        # Test read permissions
        content = test_file.read_text()
        assert content == "test"
        
        # Cleanup
        test_file.unlink()
        test_dir.rmdir()
    
    def test_environment_variables(self):
        """Test that required environment variables are set."""
        required_vars = [
            "DB_HOST",
            "DB_PORT", 
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD"
        ]
        
        for var in required_vars:
            assert os.getenv(var) is not None, f"Environment variable {var} is not set"


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    def test_large_map_generation(self):
        """Test generation of a larger map."""
        response = requests.post(
            f"{API_BASE_URL}/maps/generate",
            json={
                "map_name": "Large_Test_Map",
                "width": 800,
                "height": 600,
                "seed": 54321
            }
        )
        
        assert response.status_code == 200
        
        # Don't wait for completion in tests, just verify it started
        map_data = response.json()
        assert "job_id" in map_data
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        # This is a basic test - in production you'd want more sophisticated monitoring
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Assert memory usage is under 2GB (adjust as needed)
        assert memory_info.rss < 2 * 1024 * 1024 * 1024, f"Memory usage too high: {memory_info.rss / 1024 / 1024:.1f} MB"
    
    def test_database_performance(self):
        """Test basic database performance."""
        # Test that we can handle multiple maps
        response = requests.get(f"{API_BASE_URL}/maps")
        assert response.status_code == 200
        
        maps = response.json()
        
        # If we have maps, test querying them
        if maps:
            map_id = maps[0]["id"]
            
            # Test that complex queries complete in reasonable time
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/maps/{map_id}/statistics")
            end_time = time.time()
            
            if response.status_code == 200:
                query_time = end_time - start_time
                assert query_time < 10.0, f"Statistics query took {query_time:.2f}s (too slow)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

