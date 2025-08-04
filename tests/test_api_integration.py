
"""
Tests for API integration with new culture, religion, and settlement endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

"""Integration test for full map generation via API (with mocked generation)."""
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock
import uuid

import pytest
from fastapi.testclient import TestClient


from py_fmg.api.main import app



class TestNewAPIEndpoints:
    """Test the new API endpoints for cultures, religions, and settlements."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.test_map_id = "550e8400-e29b-41d4-a716-446655440000"
    
    @patch('py_fmg.api.main.db')
    def test_get_map_cultures_endpoint(self, mock_db):
        """Test the /maps/{map_id}/cultures endpoint."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Make request
        response = self.client.get(f"/maps/{self.test_map_id}/cultures")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of cultures
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check culture structure
        culture = data[0]
        required_fields = ["id", "name", "color", "type", "area_km2", "population", "expansionism", "center_cell"]
        for field in required_fields:
            assert field in culture
        
        # Verify data types
        assert isinstance(culture["id"], int)
        assert isinstance(culture["name"], str)
        assert isinstance(culture["color"], str)
        assert isinstance(culture["area_km2"], float)
        assert isinstance(culture["population"], int)
        assert isinstance(culture["expansionism"], float)
    
    @patch('py_fmg.api.main.db')
    def test_get_map_religions_endpoint(self, mock_db):
        """Test the /maps/{map_id}/religions endpoint."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Make request
        response = self.client.get(f"/maps/{self.test_map_id}/religions")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of religions
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check religion structure
        religion = data[0]
        required_fields = ["id", "name", "color", "type", "form", "expansion", "expansionism", "area_km2", "rural_population", "urban_population"]
        for field in required_fields:
            assert field in religion
        
        # Verify data types
        assert isinstance(religion["id"], int)
        assert isinstance(religion["name"], str)
        assert isinstance(religion["type"], str)
        assert isinstance(religion["form"], str)
        assert isinstance(religion["area_km2"], float)
        assert isinstance(religion["rural_population"], float)
        assert isinstance(religion["urban_population"], float)
        
        # Check optional deity field
        assert "deity" in religion
        # Deity can be None for folk religions
    
    @patch('py_fmg.api.main.db')
    def test_get_map_settlements_endpoint(self, mock_db):
        """Test the /maps/{map_id}/settlements endpoint."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Make request
        response = self.client.get(f"/maps/{self.test_map_id}/settlements")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of settlements
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check settlement structure
        settlement = data[0]
        required_fields = ["id", "name", "type", "population", "is_capital", "is_port"]
        for field in required_fields:
            assert field in settlement
        
        # Verify data types
        assert isinstance(settlement["id"], int)
        assert isinstance(settlement["name"], str)
        assert isinstance(settlement["type"], str)
        assert isinstance(settlement["population"], int)
        assert isinstance(settlement["is_capital"], bool)
        assert isinstance(settlement["is_port"], bool)
        
        # Check optional relationship fields
        optional_fields = ["culture_name", "state_name", "religion_name"]
        for field in optional_fields:
            assert field in settlement
            # These can be None or strings
    
    @patch('py_fmg.api.main.db')
    def test_endpoint_with_nonexistent_map(self, mock_db):
        """Test endpoints with non-existent map ID."""
        # Mock database session returning None for map
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        nonexistent_map_id = "00000000-0000-0000-0000-000000000000"
        
        # Test all new endpoints
        endpoints = ["/cultures", "/religions", "/settlements"]
        
        for endpoint in endpoints:
            response = self.client.get(f"/maps/{nonexistent_map_id}{endpoint}")
            assert response.status_code == 404
            assert "Map not found" in response.json()["detail"]
    
    def test_api_response_models_match_pydantic(self):
        """Test that API response models match our Pydantic models."""
        from py_fmg.api.main import CultureInfo, ReligionInfo, SettlementInfo
        
        # Test CultureInfo model
        culture_data = {
            "id": 1,
            "name": "Test Culture",
            "color": "#ff0000",
            "type": "Highland",
            "area_km2": 1500.0,
            "population": 50000,
            "expansionism": 1.2,
            "center_cell": 100
        }
        culture = CultureInfo(**culture_data)
        assert culture.id == 1
        assert culture.name == "Test Culture"
        
        # Test ReligionInfo model
        religion_data = {
            "id": 1,
            "name": "Test Religion",
            "color": "#gold",
            "type": "Organized",
            "form": "Monotheism",
            "deity": "Test God",
            "expansion": "global",
            "expansionism": 2.0,
            "area_km2": 2000.0,
            "rural_population": 80000.0,
            "urban_population": 20000.0
        }
        religion = ReligionInfo(**religion_data)
        assert religion.id == 1
        assert religion.deity == "Test God"
        
        # Test SettlementInfo model
        settlement_data = {
            "id": 1,
            "name": "Test City",
            "type": "capital",
            "population": 25000,
            "is_capital": True,
            "is_port": False,
            "culture_name": "Test Culture",
            "state_name": "Test Kingdom",
            "religion_name": "Test Religion"
        }
        settlement = SettlementInfo(**settlement_data)
        assert settlement.id == 1
        assert settlement.is_capital == True
    
    def test_api_openapi_documentation(self):
        """Test that OpenAPI documentation includes new endpoints."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Check that new endpoints are documented
        assert "/maps/{map_id}/cultures" in paths
        assert "/maps/{map_id}/religions" in paths
        assert "/maps/{map_id}/settlements" in paths
        
        # Check HTTP methods
        assert "get" in paths["/maps/{map_id}/cultures"]
        assert "get" in paths["/maps/{map_id}/religions"]
        assert "get" in paths["/maps/{map_id}/settlements"]
    
    def test_api_cors_headers(self):
        """Test that CORS headers are properly set for new endpoints."""
        # Test preflight request
        response = self.client.options(f"/maps/{self.test_map_id}/cultures")
        
        # Should have CORS headers (depending on configuration)
        # This test might need adjustment based on actual CORS setup
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled


class TestAPIDataConsistency:
    """Test data consistency across different API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.test_map_id = "550e8400-e29b-41d4-a716-446655440000"
    
    @patch('py_fmg.api.main.db')
    def test_culture_religion_consistency(self, mock_db):
        """Test that cultures and religions have consistent relationships."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Get cultures and religions
        cultures_response = self.client.get(f"/maps/{self.test_map_id}/cultures")
        religions_response = self.client.get(f"/maps/{self.test_map_id}/religions")
        
        assert cultures_response.status_code == 200
        assert religions_response.status_code == 200
        
        cultures = cultures_response.json()
        religions = religions_response.json()
        
        # Basic consistency checks
        assert len(cultures) > 0
        assert len(religions) > 0
        
        # In a real implementation, we might check:
        # - Folk religions should have associated cultures
        # - Culture names should be consistent across endpoints
        # - Population numbers should be reasonable
    
    @patch('py_fmg.api.main.db')
    def test_settlement_state_consistency(self, mock_db):
        """Test that settlements and states have consistent relationships."""
        # Mock database session and map object
        mock_session = Mock()
        mock_map = Mock()
        mock_map.id = self.test_map_id
        
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_map
        
        # Get settlements
        settlements_response = self.client.get(f"/maps/{self.test_map_id}/settlements")
        assert settlements_response.status_code == 200
        
        settlements = settlements_response.json()
        
        # Check that capitals are properly marked
        capitals = [s for s in settlements if s["is_capital"]]
        assert len(capitals) > 0
        
        # Check that capital settlements have higher populations (generally)
        if len(settlements) > 1:
            capital_pops = [s["population"] for s in settlements if s["is_capital"]]
            non_capital_pops = [s["population"] for s in settlements if not s["is_capital"]]
            
            if capital_pops and non_capital_pops:
                avg_capital_pop = sum(capital_pops) / len(capital_pops)
                avg_non_capital_pop = sum(non_capital_pops) / len(non_capital_pops)
                
                # Capitals should generally be larger (this is a soft check)
                # In real data this might not always be true, but for mock data it should be
                assert avg_capital_pop >= avg_non_capital_pop


if __name__ == "__main__":
    pytest.main([__file__])
