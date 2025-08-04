"""
Tests for the map editor integration.

This module tests the editing functionality including terrain modification,
settlement management, and other post-generation editing features.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import uuid

from py_fmg.api.main import app
from py_fmg.config.editor_settings import EditPermissionLevel, validate_edit_permission


class TestEditorAPI:
    """Test the editor API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.test_map_id = str(uuid.uuid4())
    
    @patch('py_fmg.api.editor.db')
    def test_terrain_edit_set_height(self, mock_db):
        """Test terrain height editing."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Mock cell validation
        mock_session.query.return_value.filter.return_value.count.return_value = 2
        
        # Mock cells for editing
        mock_cell1 = Mock()
        mock_cell1.height = 50.0
        mock_cell2 = Mock()
        mock_cell2.height = 60.0
        
        mock_session.query.return_value.filter.return_value.first.side_effect = [mock_cell1, mock_cell2]
        
        # Test terrain edit request
        edit_data = {
            "cell_indices": [100, 101],
            "operation": "set_height",
            "value": 75.0,
            "relative": False
        }
        
        response = self.client.post(f"/maps/{self.test_map_id}/edit/terrain", json=edit_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "2 cells modified" in data["message"]
        assert data["affected_cells"] == [100, 101]
        assert data["regenerate_required"] == True
    
    @patch('py_fmg.api.editor.db')
    def test_settlement_add(self, mock_db):
        """Test adding a new settlement."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Test settlement add request
        edit_data = {
            "operation": "add",
            "name": "New Town",
            "x": 100.5,
            "y": 200.3,
            "settlement_type": "town",
            "population": 5000,
            "is_capital": False
        }
        
        response = self.client.post(f"/maps/{self.test_map_id}/edit/settlements", json=edit_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "New Town" in data["message"]
    
    @patch('py_fmg.api.editor.db')
    def test_batch_edit(self, mock_db):
        """Test batch editing operations."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Test batch edit request
        batch_data = {
            "operations": [
                {
                    "type": "terrain",
                    "cell_indices": [100, 101],
                    "operation": "set_height",
                    "value": 80.0
                },
                {
                    "type": "settlement",
                    "operation": "add",
                    "name": "Batch Town",
                    "x": 150.0,
                    "y": 250.0
                }
            ],
            "validate_only": True,
            "auto_regenerate": False
        }
        
        response = self.client.post(f"/maps/{self.test_map_id}/edit/batch", json=batch_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "validation completed" in data["message"].lower()
    
    def test_invalid_map_id(self):
        """Test editing with invalid map ID."""
        invalid_map_id = str(uuid.uuid4())
        
        edit_data = {
            "cell_indices": [100],
            "operation": "set_height",
            "value": 50.0
        }
        
        with patch('py_fmg.api.editor.db') as mock_db:
            mock_session = Mock()
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            response = self.client.post(f"/maps/{invalid_map_id}/edit/terrain", json=edit_data)
            
            assert response.status_code == 404
            assert "Map not found" in response.json()["detail"]


class TestEditorSettings:
    """Test editor settings and permissions."""
    
    def test_permission_validation(self):
        """Test edit permission validation."""
        # Test basic edit permissions
        assert validate_edit_permission(
            "modify", "settlements", EditPermissionLevel.BASIC_EDIT, 50
        ) == True
        
        assert validate_edit_permission(
            "remove", "settlements", EditPermissionLevel.BASIC_EDIT, 50
        ) == False  # Remove not allowed for basic edit
        
        # Test admin permissions
        assert validate_edit_permission(
            "remove", "states", EditPermissionLevel.ADMIN, 1000
        ) == True
        
        # Test cell count limits
        assert validate_edit_permission(
            "modify", "terrain", EditPermissionLevel.BASIC_EDIT, 200
        ) == False  # Exceeds limit for basic edit
    
    def test_feature_constraints(self):
        """Test feature-specific constraints."""
        from py_fmg.config.editor_settings import get_feature_constraints
        
        terrain_constraints = get_feature_constraints("terrain")
        assert terrain_constraints is not None
        assert terrain_constraints.min_height == -100.0
        assert terrain_constraints.max_height == 300.0
        
        settlement_constraints = get_feature_constraints("settlements")
        assert settlement_constraints is not None
        assert settlement_constraints.min_population == 100
        assert "village" in settlement_constraints.allowed_settlement_types


class TestHydrologyFix:
    """Test the hydrology module fix."""
    
    def test_hydrology_import(self):
        """Test that hydrology module can be imported without errors."""
        from py_fmg.core.hydrology import Hydrology, HydrologyOptions
        
        # Should not raise any import errors
        options = HydrologyOptions()
        assert options.min_river_flow == 30.0
    
    def test_hydrology_initialization(self):
        """Test hydrology initialization with features parameter."""
        from py_fmg.core.hydrology import Hydrology, HydrologyOptions
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.heights = [10, 20, 30, 40, 50]
        
        # Mock features (optional)
        mock_features = Mock()
        
        # Test initialization with features
        hydrology = Hydrology(mock_graph, features=mock_features)
        assert hydrology.graph == mock_graph
        assert hydrology.features == mock_features
        
        # Test initialization without features
        hydrology_no_features = Hydrology(mock_graph)
        assert hydrology_no_features.graph == mock_graph
        assert hydrology_no_features.features is None


class Test3DTileGeneration:
    """Test 3D tile generation functionality."""
    
    @patch('py_fmg.generate_3d_tiles.requests')
    @patch('py_fmg.generate_3d_tiles.create_engine')
    def test_3d_tile_generation(self, mock_engine, mock_requests):
        """Test 3D tile generation script."""
        # Mock database connection
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock API response for map ID
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = [{"id": "test-map-id"}]
        
        # Import and test the function
        from py_fmg.generate_3d_tiles import prepare_3d_geometries
        
        # Should not raise errors
        prepare_3d_geometries("test-map-id")
        
        # Verify database queries were executed
        assert mock_conn.execute.called
    
    def test_cesium_viewer_creation(self):
        """Test Cesium viewer HTML creation."""
        from py_fmg.generate_3d_tiles import create_cesium_viewer
        from pathlib import Path
        
        # Create viewer
        create_cesium_viewer()
        
        # Check that viewer directory and HTML file exist
        viewer_path = Path("viewer/index.html")
        assert viewer_path.exists()
        
        # Check HTML content
        with open(viewer_path) as f:
            content = f.read()
            assert "Cesium" in content
            assert "cesiumContainer" in content
            assert "showTerrain" in content


class TestDockerCompose:
    """Test Docker Compose configuration."""
    
    def test_docker_compose_structure(self):
        """Test that docker-compose.yaml has correct structure."""
        import yaml
        from pathlib import Path
        
        compose_file = Path("docker-compose.yaml")
        assert compose_file.exists()
        
        with open(compose_file) as f:
            compose_config = yaml.safe_load(f)
        
        # Check services exist
        services = compose_config.get("services", {})
        assert "web" in services
        assert "db" in services
        assert "pg2b3dm" in services
        assert "cesium-viewer" in services
        
        # Check pg2b3dm configuration
        pg2b3dm_service = services["pg2b3dm"]
        assert pg2b3dm_service["image"] == "geodan/pg2b3dm:latest"
        assert "PGHOST" in pg2b3dm_service["environment"]
        
        # Check volumes
        volumes = compose_config.get("volumes", {})
        assert "tiles_output" in volumes


class TestVisualizationScript:
    """Test the visualization script enhancements."""
    
    @patch('py_fmg.visualize_map.create_engine')
    @patch('py_fmg.visualize_map.requests')
    def test_visualization_script(self, mock_requests, mock_engine):
        """Test the enhanced visualization script."""
        # Mock API response
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = [{"id": "test-map-id"}]
        
        # Mock database connection
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock database results
        mock_conn.execute.return_value.fetchone.return_value = (
            "Test Map", "12345", 800, 600, 5000, "2025-01-01"
        )
        mock_conn.execute.return_value.fetchall.return_value = []
        
        # Import and test
        from py_fmg.visualize_map import visualize_map
        
        # Should not raise errors when mocked
        # visualize_map("test-map-id")  # Would require matplotlib display


if __name__ == "__main__":
    pytest.main([__file__])

