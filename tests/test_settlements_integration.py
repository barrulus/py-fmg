"""
Tests for the settlements module integration with Pydantic models.
"""

import pytest
from unittest.mock import Mock

from py_fmg.core.settlements import Settlement, State, SettlementOptions


class TestSettlementModels:
    """Test Pydantic models for settlements."""
    
    def test_settlement_options_creation(self):
        """Test SettlementOptions model creation."""
        options = SettlementOptions()
        
        # Test default values
        assert options.states_number == 30
        assert options.manors_number == 1000
        assert options.growth_rate == 1.0
        assert options.states_growth_rate == 1.0
        assert options.size_variety == 3.0
        assert options.min_state_cells == 10
        
        # Spacing parameters
        assert options.capital_spacing_divisor == 2
        assert options.town_spacing_base == 150
        assert options.town_spacing_power == 0.7
        
        # Cost parameters
        assert options.culture_same_bonus == -9
        assert options.culture_foreign_penalty == 100
        assert options.sea_crossing_penalty == 1000
        assert options.nomadic_sea_penalty == 10000
        
        # Population parameters
        assert options.urbanization_rate == 0.1
        assert options.capital_pop_multiplier == 1.3
        assert options.port_pop_multiplier == 1.3
    
    def test_settlement_options_custom_values(self):
        """Test SettlementOptions with custom values."""
        options = SettlementOptions(
            states_number=50,
            manors_number=2000,
            growth_rate=1.5,
            urbanization_rate=0.15
        )
        
        assert options.states_number == 50
        assert options.manors_number == 2000
        assert options.growth_rate == 1.5
        assert options.urbanization_rate == 0.15
    
    def test_settlement_creation(self):
        """Test Settlement model creation."""
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Test City",
            population=25000.0,
            is_capital=True,
            state_id=1,
            culture_id=1,
            religion_id=1,
            type="Capital"
        )
        
        assert settlement.id == 1
        assert settlement.cell_id == 100
        assert settlement.x == 50.5
        assert settlement.y == 75.2
        assert settlement.name == "Test City"
        assert settlement.population == 25000.0
        assert settlement.is_capital == True
        assert settlement.state_id == 1
        assert settlement.culture_id == 1
        assert settlement.religion_id == 1
        assert settlement.type == "Capital"
        
        # Test default feature values
        assert settlement.citadel == False
        assert settlement.plaza == False
        assert settlement.walls == False
        assert settlement.shanty == False
        assert settlement.temple == False
    
    def test_settlement_with_features(self):
        """Test Settlement model with features."""
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Fortified City",
            population=35000.0,
            citadel=True,
            walls=True,
            temple=True,
            plaza=True
        )
        
        assert settlement.citadel == True
        assert settlement.walls == True
        assert settlement.temple == True
        assert settlement.plaza == True
        assert settlement.shanty == False
    
    def test_state_creation(self):
        """Test State model creation."""
        state = State(
            id=1,
            name="Test Kingdom",
            capital_id=1,
            culture_id=1,
            expansionism=1.2,
            color="#ff0000",
            type="Highland",
            center_cell=500,
            territory_cells=[500, 501, 502, 503]
        )
        
        assert state.id == 1
        assert state.name == "Test Kingdom"
        assert state.capital_id == 1
        assert state.culture_id == 1
        assert state.expansionism == 1.2
        assert state.color == "#ff0000"
        assert state.type == "Highland"
        assert state.center_cell == 500
        assert state.territory_cells == [500, 501, 502, 503]
        assert state.removed == False
        assert state.locked == False
    
    def test_settlement_serialization(self):
        """Test Settlement model serialization."""
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Test City",
            population=25000.0,
            is_capital=True
        )
        
        # Test dict conversion
        settlement_dict = settlement.model_dump()
        assert settlement_dict["id"] == 1
        assert settlement_dict["name"] == "Test City"
        assert settlement_dict["population"] == 25000.0
        assert settlement_dict["is_capital"] == True
        
        # Test JSON serialization
        settlement_json = settlement.model_dump_json()
        assert '"id":1' in settlement_json
        assert '"name":"Test City"' in settlement_json
    
    def test_state_serialization(self):
        """Test State model serialization."""
        state = State(
            id=1,
            name="Test Kingdom",
            capital_id=1,
            culture_id=1,
            territory_cells=[500, 501, 502]
        )
        
        # Test dict conversion
        state_dict = state.model_dump()
        assert state_dict["id"] == 1
        assert state_dict["name"] == "Test Kingdom"
        assert state_dict["territory_cells"] == [500, 501, 502]
        
        # Test JSON serialization
        state_json = state.model_dump_json()
        assert '"id":1' in state_json
        assert '"name":"Test Kingdom"' in state_json


class TestSettlementIntegration:
    """Test integration between different settlement components."""
    
    def test_pydantic_validation(self):
        """Test Pydantic validation works correctly."""
        # Test valid settlement
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Valid Settlement"
        )
        assert settlement.id == 1
        
        # Test invalid data should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Settlement(
                id="not_an_int",  # Should be int
                cell_id=100,
                x=50.5,
                y=75.2,
                name="Invalid Settlement"
            )
    
    def test_settlement_options_validation(self):
        """Test SettlementOptions validation."""
        # Valid options
        options = SettlementOptions(states_number=25)
        assert options.states_number == 25
        
        # Test field descriptions are present
        assert "Target number of states" in str(SettlementOptions.model_fields["states_number"])
    
    def test_model_config_allows_arbitrary_types(self):
        """Test that models allow arbitrary types."""
        state = State(
            id=1,
            name="Test State",
            capital_id=1,
            culture_id=1,
            territory_cells=[1, 2, 3, 4, 5]  # List type should be allowed
        )
        
        assert isinstance(state.territory_cells, list)
        assert len(state.territory_cells) == 5
    
    def test_settlement_relationships(self):
        """Test settlement relationship fields."""
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Connected Settlement",
            state_id=5,
            culture_id=3,
            religion_id=2
        )
        
        # Test that relationship IDs are properly set
        assert settlement.state_id == 5
        assert settlement.culture_id == 3
        assert settlement.religion_id == 2
    
    def test_settlement_port_functionality(self):
        """Test settlement port-related fields."""
        port_settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2,
            name="Port City",
            port_id=10,  # Reference to a port feature
            population=30000.0
        )
        
        assert port_settlement.port_id == 10
        # Port settlements might get population bonuses
        assert port_settlement.population == 30000.0
    
    def test_state_territory_management(self):
        """Test state territory management."""
        state = State(
            id=1,
            name="Expanding Kingdom",
            capital_id=1,
            culture_id=1,
            territory_cells=[100, 101, 102]
        )
        
        # Test initial territory
        assert len(state.territory_cells) == 3
        assert 100 in state.territory_cells
        
        # Test territory expansion (would be done by the generator)
        state.territory_cells.extend([103, 104])
        assert len(state.territory_cells) == 5
        assert 104 in state.territory_cells


if __name__ == "__main__":
    pytest.main([__file__])

