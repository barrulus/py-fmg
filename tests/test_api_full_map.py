"""Test full map generation via the API."""

import os
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from py_fmg.api.main import app
from py_fmg.db.connection import db

# Create simplified test models without PostGIS dependencies
TestBase = declarative_base()


class TestMap(TestBase):
    """Simplified map table for testing."""

    __tablename__ = "maps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    seed = Column(String(50), nullable=False)
    grid_seed = Column(String(50), nullable=False)
    map_seed = Column(String(50), nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    cells_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    generation_time_seconds = Column(Float)
    config_json = Column(Text)


class TestGenerationJob(TestBase):
    """Simplified generation job table for testing."""

    __tablename__ = "generation_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    map_id = Column(String, nullable=True)
    status = Column(String(20), default="pending")
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    seed = Column(String(50))
    grid_seed = Column(String(50))
    map_seed = Column(String(50))
    width = Column(Float)
    height = Column(Float)
    cells_desired = Column(Integer)
    template_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


@pytest.fixture(scope="function")
def test_db():
    """Create a test database with SQLite."""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )

    # Create tables using simplified models
    TestBase.metadata.create_all(bind=engine)

    # Create session factory
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Mock the database instance
    original_engine = db.engine
    original_session_local = db.SessionLocal

    db.engine = engine
    db.SessionLocal = TestingSessionLocal

    yield engine

    # Restore original database instance
    db.engine = original_engine
    db.SessionLocal = original_session_local


@pytest.fixture
def client(test_db):
    """Create a test client for the FastAPI app with test database."""
    # Mock the model classes in the API module
    with patch.dict(
        os.environ,
        {
            "DB_USER": "test",
            "DB_HOST": "localhost",
            "DB_NAME": "test",
            "DB_PASSWORD": "test",
            "DB_PORT": "5432",
        },
    ):
        # Replace the models with our test versions
        with patch("py_fmg.api.main.Map", TestMap), patch(
            "py_fmg.api.main.GenerationJob", TestGenerationJob
        ):
            return TestClient(app)


def test_api_full_map_generation(client: TestClient) -> None:
    """Test generating a complete map via the API endpoints."""

    # Test 1: Check API health
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"

    # Test 2: Start map generation
    generation_request = {
        "grid_seed": "test123",
        "map_seed": "test456",
        "width": 400,
        "height": 300,
        "cells_desired": 2000,
        "template_name": "default",
        "map_name": "API Test Map",
    }

    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200

    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["status"] == "pending"
    assert job_data["progress_percent"] == 0

    job_id = job_data["job_id"]

    # Test 3: Poll job status until completion
    max_wait_time = 300  # 5 minutes max wait time
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200

        status_data = response.json()
        assert status_data["job_id"] == job_id

        if status_data["status"] == "completed":
            assert status_data["progress_percent"] == 100
            assert status_data["map_id"] is not None
            map_id = status_data["map_id"]
            break
        elif status_data["status"] == "failed":
            pytest.fail(
                f"Map generation failed: {status_data.get('error_message', 'Unknown error')}"
            )

        # Wait before next poll
        time.sleep(2)
    else:
        pytest.fail(f"Map generation timed out after {max_wait_time} seconds")

    # Test 4: Retrieve generated map details
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
    assert map_data["generation_time_seconds"] is not None

    # Test 5: List maps includes our generated map
    response = client.get("/maps")
    assert response.status_code == 200

    maps_list = response.json()
    assert isinstance(maps_list, list)
    assert len(maps_list) > 0

    # Find our map in the list
    our_map = next((m for m in maps_list if m["id"] == map_id), None)
    assert our_map is not None
    assert our_map["name"] == "API Test Map"


def test_api_invalid_job_id(client: TestClient) -> None:
    """Test requesting status for non-existent job."""
    response = client.get("/jobs/invalid-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_api_invalid_map_id(client: TestClient) -> None:
    """Test requesting non-existent map."""
    response = client.get("/maps/invalid-map-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Map not found"


def test_api_generation_with_legacy_seed(client: TestClient) -> None:
    """Test map generation using legacy single seed parameter."""
    generation_request = {
        "seed": "legacy789",
        "width": 200,
        "height": 150,
        "cells_desired": 1000,
        "template_name": "default",
    }

    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200

    job_data = response.json()
    job_id = job_data["job_id"]

    # Wait for completion (simplified polling)
    max_attempts = 60  # 2 minutes max
    for _ in range(max_attempts):
        response = client.get(f"/jobs/{job_id}")
        status_data = response.json()

        if status_data["status"] == "completed":
            map_id = status_data["map_id"]

            # Verify map uses legacy seed for both grid and map seeds
            response = client.get(f"/maps/{map_id}")
            map_data = response.json()
            assert map_data["seed"] == "legacy789"
            assert map_data["grid_seed"] == "legacy789"
            assert map_data["map_seed"] == "legacy789"
            break
        elif status_data["status"] == "failed":
            pytest.fail(
                f"Map generation failed: {status_data.get('error_message', 'Unknown error')}"
            )

        time.sleep(2)
    else:
        pytest.fail("Legacy seed map generation timed out")


def test_api_validation_errors(client: TestClient) -> None:
    """Test API request validation."""

    # Test invalid width
    response = client.post(
        "/maps/generate",
        json={
            "width": 50,  # Below minimum of 100
            "height": 300,
            "cells_desired": 2000,
        },
    )
    assert response.status_code == 422

    # Test invalid cells_desired
    response = client.post(
        "/maps/generate",
        json={
            "width": 400,
            "height": 300,
            "cells_desired": 500,  # Below minimum of 1000
        },
    )
    assert response.status_code == 422


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
        "map_name": f"Template Test - {template_name}",
    }

    response = client.post("/maps/generate", json=generation_request)
    assert response.status_code == 200

    job_data = response.json()
    job_id = job_data["job_id"]

    # Simplified completion check
    max_attempts = 60
    for _ in range(max_attempts):
        response = client.get(f"/jobs/{job_id}")
        status_data = response.json()

        if status_data["status"] == "completed":
            map_id = status_data["map_id"]

            # Verify the map was created successfully
            response = client.get(f"/maps/{map_id}")
            assert response.status_code == 200
            map_data = response.json()
            assert template_name in map_data["name"]
            break
        elif status_data["status"] == "failed":
            pytest.fail(
                f"Template {template_name} generation failed: {status_data.get('error_message', 'Unknown error')}"
            )

        time.sleep(2)
    else:
        pytest.fail(f"Template {template_name} generation timed out")
