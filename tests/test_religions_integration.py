"""
Tests for the religions module integration with Pydantic models.
"""

import pytest
from unittest.mock import Mock

from py_fmg.core.religions import Religion, ReligionOptions, ReligionGenerator
from py_fmg.core.alea_prng import AleaPRNG


class TestReligionModels:
    """Test Pydantic models for religions."""
    
    def test_religion_options_creation(self):
        """Test ReligionOptions model creation."""
        options = ReligionOptions()
        
        # Test default values
        assert options.religions_number == 8
        assert options.theocracy_chance == 0.1
        assert options.temple_pop_threshold == 50
        assert options.sacred_site_density == 1.0
    
    def test_religion_options_custom_values(self):
        """Test ReligionOptions with custom values."""
        options = ReligionOptions(
            religions_number=12,
            theocracy_chance=0.2,
            temple_pop_threshold=100,
            sacred_site_density=1.5
        )
        
        assert options.religions_number == 12
        assert options.theocracy_chance == 0.2
        assert options.temple_pop_threshold == 100
        assert options.sacred_site_density == 1.5
    
    def test_religion_creation(self):
        """Test Religion model creation."""
        religion = Religion(
            id=1,
            name="Test Religion",
            color="#gold",
            type="Organized",
            form="Monotheism",
            culture_id=1,
            center=100,
            deity="Test God",
            expansion="global",
            expansionism=2.0
        )
        
        assert religion.id == 1
        assert religion.name == "Test Religion"
        assert religion.color == "#gold"
        assert religion.type == "Organized"
        assert religion.form == "Monotheism"
        assert religion.culture_id == 1
        assert religion.center == 100
        assert religion.deity == "Test God"
        assert religion.expansion == "global"
        assert religion.expansionism == 2.0
        assert len(religion.origins) == 0
        assert religion.code == ""
        assert len(religion.cells) == 0
        assert religion.area == 0.0
    
    def test_religion_with_optional_fields(self):
        """Test Religion model with optional fields."""
        religion = Religion(
            id=1,
            name="Folk Religion",
            color="#brown",
            type="Folk",
            form="Shamanism",
            culture_id=1,
            center=100,
            deity=None,  # Folk religions might not have deities
            origins=[2, 3],  # Derived from other religions
            code="FR01"
        )
        
        assert religion.deity is None
        assert religion.origins == [2, 3]
        assert religion.code == "FR01"
    
    def test_religion_serialization(self):
        """Test Religion model serialization."""
        religion = Religion(
            id=1,
            name="Test Religion",
            color="#gold",
            type="Organized",
            form="Monotheism",
            culture_id=1,
            center=100,
            deity="Test God",
            cells={100, 101, 102}
        )
        
        # Test dict conversion
        religion_dict = religion.model_dump()
        assert religion_dict["id"] == 1
        assert religion_dict["name"] == "Test Religion"
        assert religion_dict["type"] == "Organized"
        assert religion_dict["deity"] == "Test God"
        
        # Test JSON serialization
        religion_json = religion.model_dump_json()
        assert '"id":1' in religion_json
        assert '"name":"Test Religion"' in religion_json


class TestReligionGenerator:
    """Test ReligionGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_graph = Mock()
        self.mock_cultures_dict = {1: Mock(), 2: Mock()}
        self.mock_cell_cultures = [1, 1, 2, 2, 0]
        self.mock_settlements_dict = {}
        self.mock_states_dict = {}
        
        # PRNG for reproducible tests
        self.prng = AleaPRNG("test_religions")
    
    def test_religion_generator_initialization(self):
        """Test ReligionGenerator initialization."""
        options = ReligionOptions(religions_number=6)
        
        generator = ReligionGenerator(
            graph=self.mock_graph,
            cultures_dict=self.mock_cultures_dict,
            cell_cultures=self.mock_cell_cultures,
            settlements_dict=self.mock_settlements_dict,
            states_dict=self.mock_states_dict,
            options=options,
            prng=self.prng
        )
        
        assert generator.graph == self.mock_graph
        assert generator.cultures_dict == self.mock_cultures_dict
        assert generator.options.religions_number == 6
        assert generator.prng == self.prng
    
    def test_religion_generator_with_defaults(self):
        """Test ReligionGenerator with default options."""
        generator = ReligionGenerator(
            graph=self.mock_graph,
            cultures_dict=self.mock_cultures_dict,
            cell_cultures=self.mock_cell_cultures,
            settlements_dict=self.mock_settlements_dict,
            states_dict=self.mock_states_dict
        )
        
        assert generator.options.religions_number == 8
        assert generator.prng is not None
    
    def test_religion_forms_available(self):
        """Test that religion forms are available."""
        generator = ReligionGenerator(
            graph=self.mock_graph,
            cultures_dict=self.mock_cultures_dict,
            cell_cultures=self.mock_cell_cultures,
            settlements_dict=self.mock_settlements_dict,
            states_dict=self.mock_states_dict
        )
        
        assert len(generator.FOLK_FORMS) > 0
        assert len(generator.ORGANIZED_FORMS) > 0
        assert len(generator.CULT_FORMS) > 0
        assert "Shamanism" in generator.FOLK_FORMS
        assert "Monotheism" in generator.ORGANIZED_FORMS


class TestReligionIntegration:
    """Test integration between different religion components."""
    
    def test_pydantic_validation(self):
        """Test Pydantic validation works correctly."""
        # Test valid religion
        religion = Religion(
            id=1,
            name="Valid Religion",
            color="#gold",
            type="Organized",
            form="Monotheism",
            culture_id=1,
            center=100
        )
        assert religion.id == 1
        
        # Test invalid data should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Religion(
                id="not_an_int",  # Should be int
                name="Invalid Religion",
                color="#gold",
                type="Organized",
                form="Monotheism",
                culture_id=1,
                center=100
            )
    
    def test_religion_options_validation(self):
        """Test ReligionOptions validation."""
        # Valid options
        options = ReligionOptions(religions_number=10)
        assert options.religions_number == 10
        
        # Test field descriptions are present
        assert "Target organized religions" in str(ReligionOptions.model_fields["religions_number"])
    
    def test_model_config_allows_arbitrary_types(self):
        """Test that models allow arbitrary types."""
        religion = Religion(
            id=1,
            name="Test Religion",
            color="#gold",
            type="Organized",
            form="Monotheism",
            culture_id=1,
            center=100,
            cells=set([1, 2, 3]),  # Set type should be allowed
            origins=[1, 2]  # List type should be allowed
        )
        
        assert isinstance(religion.cells, set)
        assert isinstance(religion.origins, list)
        assert len(religion.cells) == 3
        assert len(religion.origins) == 2
    
    def test_religion_statistics(self):
        """Test religion statistics fields."""
        religion = Religion(
            id=1,
            name="Test Religion",
            color="#gold",
            type="Organized",
            form="Monotheism",
            culture_id=1,
            center=100,
            area=1500.0,
            rural_population=50000.0,
            urban_population=15000.0
        )
        
        assert religion.area == 1500.0
        assert religion.rural_population == 50000.0
        assert religion.urban_population == 15000.0
        
        # Test total population calculation (if implemented)
        total_pop = religion.rural_population + religion.urban_population
        assert total_pop == 65000.0


if __name__ == "__main__":
    pytest.main([__file__])

