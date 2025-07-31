"""Test the complete map generation pipeline through the main.py API."""

import pytest
import asyncio
import time
from httpx import AsyncClient
from datetime import datetime

from py_fmg.api.main import app
from py_fmg.config.heightmap_templates import TEMPLATES

# Test configuration constants
TEST_WIDTH = 800  # Smaller for faster testing
TEST_HEIGHT = 600
TEST_CELLS_DESIRED = 5000  # Smaller for faster testing
DEFAULT_SEED = "test123456"

# Test timeouts
JOB_TIMEOUT_SECONDS = 120  # 2 minutes max for generation
POLL_INTERVAL_SECONDS = 2  # Check job status every 2 seconds


class TestMainPipelineAPI:
    """Test the main API pipeline end-to-end."""

    async def wait_for_job_completion(self, client: AsyncClient, job_id: str, timeout: int = JOB_TIMEOUT_SECONDS):
        """Poll job status until completion or timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = await client.get(f"/jobs/{job_id}")
            assert response.status_code == 200
            
            job_data = response.json()
            status = job_data["status"]
            progress = job_data["progress_percent"]
            
            print(f"Job {job_id}: {status} ({progress}%)")
            
            if status == "completed":
                return job_data
            elif status == "failed":
                error_msg = job_data.get("error_message", "Unknown error")
                pytest.fail(f"Job failed: {error_msg}")
            elif status in ["pending", "running"]:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            else:
                pytest.fail(f"Unknown job status: {status}")
        
        pytest.fail(f"Job timeout after {timeout} seconds")

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test that the API is healthy."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test the root endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Fantasy Map Generator API"
            assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_single_template_generation(self):
        """Test generating a single map with the atoll template."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start map generation
            request_data = {
                "grid_seed": DEFAULT_SEED,
                "map_seed": DEFAULT_SEED,
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "atoll",
                "map_name": "Test Atoll Map"
            }
            
            response = await client.post("/maps/generate", json=request_data)
            assert response.status_code == 200
            
            job_data = response.json()
            job_id = job_data["job_id"]
            assert job_data["status"] == "pending"
            assert job_data["progress_percent"] == 0
            
            # Wait for completion
            completed_job = await self.wait_for_job_completion(client, job_id)
            map_id = completed_job["map_id"]
            assert map_id is not None
            
            # Verify the map was created
            response = await client.get(f"/maps/{map_id}")
            assert response.status_code == 200
            
            map_data = response.json()
            assert map_data["name"] == "Test Atoll Map"
            assert map_data["grid_seed"] == DEFAULT_SEED
            assert map_data["map_seed"] == DEFAULT_SEED
            assert map_data["width"] == TEST_WIDTH
            assert map_data["height"] == TEST_HEIGHT
            assert map_data["cells_count"] > 0
            assert map_data["created_at"] is not None

    @pytest.mark.asyncio
    async def test_dual_seed_system(self):
        """Test the dual seed system (grid_seed vs map_seed)."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Generate two maps with same grid_seed but different map_seed
            grid_seed = "grid123"
            map_seed1 = "map456"
            map_seed2 = "map789"
            
            # First map
            request_data1 = {
                "grid_seed": grid_seed,
                "map_seed": map_seed1,
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "archipelago",
                "map_name": "Dual Seed Test 1"
            }
            
            response1 = await client.post("/maps/generate", json=request_data1)
            assert response1.status_code == 200
            job_id1 = response1.json()["job_id"]
            
            # Second map
            request_data2 = {
                "grid_seed": grid_seed,
                "map_seed": map_seed2,
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "archipelago",
                "map_name": "Dual Seed Test 2"
            }
            
            response2 = await client.post("/maps/generate", json=request_data2)
            assert response2.status_code == 200
            job_id2 = response2.json()["job_id"]
            
            # Wait for both to complete
            completed_job1 = await self.wait_for_job_completion(client, job_id1)
            completed_job2 = await self.wait_for_job_completion(client, job_id2)
            
            map_id1 = completed_job1["map_id"]
            map_id2 = completed_job2["map_id"]
            
            # Verify both maps
            response1 = await client.get(f"/maps/{map_id1}")
            response2 = await client.get(f"/maps/{map_id2}")
            
            map1_data = response1.json()
            map2_data = response2.json()
            
            # Both should have same grid_seed but different map_seed
            assert map1_data["grid_seed"] == grid_seed
            assert map2_data["grid_seed"] == grid_seed
            assert map1_data["map_seed"] == map_seed1
            assert map2_data["map_seed"] == map_seed2
            
            # Maps should be different (different heights due to different map_seed)
            assert map1_data["id"] != map2_data["id"]

    @pytest.mark.asyncio
    async def test_legacy_seed_compatibility(self):
        """Test that legacy single seed parameter still works."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Use legacy 'seed' parameter
            request_data = {
                "seed": "legacy123",  # Old parameter name
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "continent",
                "map_name": "Legacy Seed Test"
            }
            
            response = await client.post("/maps/generate", json=request_data)
            assert response.status_code == 200
            
            job_id = response.json()["job_id"]
            completed_job = await self.wait_for_job_completion(client, job_id)
            map_id = completed_job["map_id"]
            
            # Verify the map
            response = await client.get(f"/maps/{map_id}")
            map_data = response.json()
            
            # Legacy seed should be copied to both grid_seed and map_seed
            assert map_data["seed"] == "legacy123"
            assert map_data["grid_seed"] == "legacy123"
            assert map_data["map_seed"] == "legacy123"

    @pytest.mark.asyncio
    async def test_multiple_templates(self):
        """Test generation with multiple templates."""
        templates_to_test = ["atoll", "continent", "archipelago"]
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            job_ids = []
            
            # Start all jobs
            for template in templates_to_test:
                request_data = {
                    "grid_seed": f"multi_{template}",
                    "map_seed": f"multi_{template}",
                    "width": TEST_WIDTH,
                    "height": TEST_HEIGHT,
                    "cells_desired": TEST_CELLS_DESIRED,
                    "template_name": template,
                    "map_name": f"Multi Template {template.title()}"
                }
                
                response = await client.post("/maps/generate", json=request_data)
                assert response.status_code == 200
                job_ids.append((template, response.json()["job_id"]))
            
            # Wait for all to complete
            map_ids = {}
            for template, job_id in job_ids:
                completed_job = await self.wait_for_job_completion(client, job_id)
                map_ids[template] = completed_job["map_id"]
            
            # Verify all maps
            for template, map_id in map_ids.items():
                response = await client.get(f"/maps/{map_id}")
                assert response.status_code == 200
                
                map_data = response.json()
                assert map_data["name"] == f"Multi Template {template.title()}"
                assert map_data["cells_count"] > 0

    @pytest.mark.asyncio
    async def test_list_maps(self):
        """Test listing all generated maps."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Generate a test map first
            request_data = {
                "grid_seed": "list_test",
                "map_seed": "list_test",
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "atoll",
                "map_name": "List Test Map"
            }
            
            response = await client.post("/maps/generate", json=request_data)
            job_id = response.json()["job_id"]
            await self.wait_for_job_completion(client, job_id)
            
            # List all maps
            response = await client.get("/maps")
            assert response.status_code == 200
            
            maps = response.json()
            assert isinstance(maps, list)
            assert len(maps) > 0
            
            # Find our test map
            test_map = next((m for m in maps if m["name"] == "List Test Map"), None)
            assert test_map is not None
            assert test_map["grid_seed"] == "list_test"
            assert test_map["map_seed"] == "list_test"

    @pytest.mark.asyncio
    async def test_job_not_found(self):
        """Test error handling for non-existent jobs."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/jobs/nonexistent-job-id")
            assert response.status_code == 404
            assert "Job not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_map_not_found(self):
        """Test error handling for non-existent maps."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/maps/nonexistent-map-id")
            assert response.status_code == 404
            assert "Map not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test error handling for invalid generation parameters."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test invalid width (too small)
            request_data = {
                "width": 50,  # Below minimum of 100
                "height": TEST_HEIGHT,
                "cells_desired": TEST_CELLS_DESIRED,
                "template_name": "atoll"
            }
            
            response = await client.post("/maps/generate", json=request_data)
            assert response.status_code == 422  # Validation error
            
            # Test invalid cells_desired (too small)
            request_data = {
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "cells_desired": 100,  # Below minimum of 1000
                "template_name": "atoll"
            }
            
            response = await client.post("/maps/generate", json=request_data)
            assert response.status_code == 422  # Validation error


def main():
    """Run the API tests."""
    import pytest
    import sys
    
    # Run the tests
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()