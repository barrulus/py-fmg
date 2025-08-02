"""
Tests for the cultures module integration with Pydantic models.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from py_fmg.core.cultures import Culture, CultureOptions, CultureGenerator
from py_fmg.core.alea_prng import AleaPRNG


class TestCultureModels:
    """Test Pydantic models for cultures."""
    
    def test_culture_options_creation(self):
        """Test CultureOptions model creation."""
        options = CultureOptions()
        
        # Test default values
        assert options.cultures_number == 12
        assert options.min_culture_cells == 10
        assert options.expansionism_modifier == 1.0
        assert options.generic_weight == 10.0
        assert options.naval_weight == 2.0
    
    def test_culture_options_custom_values(self):
        """Test CultureOptions with custom values."""
        options = CultureOptions(
            cultures_number=15,
            min_culture_cells=20,
            expansionism_modifier=1.5,
            generic_weight=8.0
        )
        
        assert options.cultures_number == 15
        assert options.min_culture_cells == 20
        assert options.expansionism_modifier == 1.5
        assert options.generic_weight == 8.0
    
    def test_culture_creation(self):
        """Test Culture model creation."""
        culture = Culture(
            id=1,
            name="Test Culture",
            color="#ff0000",
            center=100,
            type="Highland",
            expansionism=1.2
        )
        
        assert culture.id == 1
        assert culture.name == "Test Culture"
        assert culture.color == "#ff0000"
        assert culture.center == 100
        assert culture.type == "Highland"
        assert culture.expansionism == 1.2
        assert culture.removed == False
        assert culture.name_base == 0
        assert len(culture.cells) == 0
    
    def test_culture_with_cells(self):
        """Test Culture model with cells."""
        culture = Culture(
            id=1,
            name="Test Culture",
            color="#ff0000",
            center=100,
            cells={100, 101, 102, 103}
        )
        
        assert len(culture.cells) == 4
        assert 100 in culture.cells
        assert 103 in culture.cells
    
    def test_culture_serialization(self):
        """Test Culture model serialization."""
        culture = Culture(
            id=1,
            name="Test Culture",
            color="#ff0000",
            center=100,
            type="Highland",
            expansionism=1.2,
            cells={100, 101}
        )
        
        # Test dict conversion
        culture_dict = culture.model_dump()
        assert culture_dict["id"] == 1
        assert culture_dict["name"] == "Test Culture"
        assert culture_dict["type"] == "Highland"
        assert culture_dict["expansionism"] == 1.2
        
        # Test JSON serialization
        culture_json = culture.model_dump_json()
        assert '"id":1' in culture_json
        assert '"name":"Test Culture"' in culture_json


class TestCultureGenerator:
    """Test CultureGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock graph
        self.mock_graph = Mock()
        self.mock_graph.points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        self.mock_graph.neighbors = [[] for _ in range(5)]
        
        # Mock features
        self.mock_features = Mock()
        self.mock_features.cells_land = np.array([True, True, True, True, False])
        
        # Mock biome classifier
        self.mock_biomes = Mock()
        
        # PRNG for reproducible tests
        self.prng = AleaPRNG("test_cultures")
    
    def test_culture_generator_initialization(self):
        """Test CultureGenerator initialization."""
        options = CultureOptions(cultures_number=5)
        
        generator = CultureGenerator(
            graph=self.mock_graph,
            features=self.mock_features,
            biome_classifier=self.mock_biomes,
            options=options,
            prng=self.prng
        )
        
        assert generator.graph == self.mock_graph
        assert generator.features == self.mock_features
        assert generator.biome_classifier == self.mock_biomes
        assert generator.options.cultures_number == 5
        assert generator.prng == self.prng
        assert len(generator.cultures) == 0
        assert generator.next_culture_id == 1
    
    def test_culture_generator_with_defaults(self):
        """Test CultureGenerator with default options."""
        generator = CultureGenerator(
            graph=self.mock_graph,
            features=self.mock_features
        )
        
        assert generator.options.cultures_number == 12
        assert generator.biome_classifier is not None
        assert generator.prng is not None
    
    def test_culture_colors_available(self):
        """Test that culture colors are available."""
        generator = CultureGenerator(
            graph=self.mock_graph,
            features=self.mock_features
        )
        
        assert len(generator.culture_colors) > 0
        assert all(color.startswith("#") for color in generator.culture_colors)
        assert len(generator.culture_colors[0]) == 7  # Hex color format


class TestCultureIntegration:
    """Test integration between different culture components."""
    
    def test_pydantic_validation(self):
        """Test Pydantic validation works correctly."""
        # Test valid culture
        culture = Culture(
            id=1,
            name="Valid Culture",
            color="#ff0000",
            center=100
        )
        assert culture.id == 1
        
        # Test invalid data should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Culture(
                id="not_an_int",  # Should be int
                name="Invalid Culture",
                color="#ff0000",
                center=100
            )
    
    def test_culture_options_validation(self):
        """Test CultureOptions validation."""
        # Valid options
        options = CultureOptions(cultures_number=10)
        assert options.cultures_number == 10
        
        # Test field descriptions are present
        assert "Target number of cultures" in str(CultureOptions.model_fields["cultures_number"])
    
    def test_model_config_allows_arbitrary_types(self):
        """Test that models allow arbitrary types (like numpy arrays)."""
        culture = Culture(
            id=1,
            name="Test Culture",
            color="#ff0000",
            center=100,
            cells=set([1, 2, 3])  # Set type should be allowed
        )
        
        assert isinstance(culture.cells, set)
        assert len(culture.cells) == 3


if __name__ == "__main__":
    pytest.main([__file__])

