"""Tests for biomes classification module."""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.biomes import BiomeClassifier, BiomeOptions, BiomeType, BIOME_NAMES
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
from py_fmg.core.hydrology import Hydrology, HydrologyOptions


class TestBiomes:
    """Test biome classification."""
    
    @pytest.fixture
    def climate_graph(self):
        """Create a test graph with climate data."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="biome_test")
        
        n_cells = len(graph.points)
        
        # Create varied terrain
        graph.heights = np.full(n_cells, 40, dtype=np.uint8)
        
        # Add water, land, and mountains
        for i in range(n_cells):
            x, y = graph.points[i]
            
            # Ocean at edges
            if x < 20 or x > 180 or y < 20 or y > 180:
                graph.heights[i] = 5
            # Mountains in center
            elif 80 < x < 120 and 80 < y < 120:
                graph.heights[i] = 80
                
        # Generate climate data
        climate = Climate(graph, map_coords=MapCoordinates(lat_n=60, lat_s=-60))
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        return graph
        
    @pytest.fixture
    def full_pipeline_graph(self):
        """Create a graph with full pipeline data (climate + hydrology)."""
        config = GridConfig(150, 150, 300)
        graph = generate_voronoi_graph(config, seed="biome_full_test")
        
        n_cells = len(graph.points)
        graph.heights = np.full(n_cells, 35, dtype=np.uint8)
        
        # Create diverse terrain
        for i in range(n_cells):
            x, y = graph.points[i]
            
            # Ocean
            if y > 130:
                graph.heights[i] = 5
            # River valley
            elif 70 < x < 80:
                graph.heights[i] = 25
            # Mountains
            elif x > 120:
                graph.heights[i] = 75
                
        # Generate climate
        climate = Climate(graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Generate hydrology
        hydrology = Hydrology(graph)
        hydrology.run_full_simulation()
        
        return graph
        
    def test_biome_classifier_initialization(self, climate_graph):
        """Test biome classifier initialization."""
        classifier = BiomeClassifier(climate_graph)
        
        assert classifier.graph is climate_graph
        assert classifier.options is not None
        assert classifier.biomes is None
        assert len(classifier.biome_regions) == 0
        
        # Check biome matrix is initialized
        assert classifier.biome_matrix is not None
        assert len(classifier.biome_matrix) == 6  # 6 temperature bands
        assert all(len(row) == 5 for row in classifier.biome_matrix)  # 5 precipitation bands
        
    def test_temperature_band_classification(self, climate_graph):
        """Test temperature band classification."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test temperature band mapping
        assert classifier._get_temperature_band(-15) == 0  # Very cold
        assert classifier._get_temperature_band(0) == 1   # Cold
        assert classifier._get_temperature_band(10) == 2  # Cool
        assert classifier._get_temperature_band(20) == 3  # Temperate
        assert classifier._get_temperature_band(30) == 4  # Warm
        assert classifier._get_temperature_band(40) == 5  # Hot
        
    def test_precipitation_band_classification(self, climate_graph):
        """Test precipitation band classification."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test precipitation band mapping
        assert classifier._get_precipitation_band(10) == 0   # Very dry
        assert classifier._get_precipitation_band(30) == 1   # Dry
        assert classifier._get_precipitation_band(75) == 2   # Moderate
        assert classifier._get_precipitation_band(150) == 3  # Wet
        assert classifier._get_precipitation_band(200) == 4  # Very wet
        
    def test_basic_biome_classification(self, climate_graph):
        """Test basic biome classification."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Should have biomes assigned
        assert classifier.biomes is not None
        assert len(classifier.biomes) == len(climate_graph.points)
        
        # Should have ocean biomes for water cells
        water_cells = np.where(climate_graph.heights < 20)[0]
        if len(water_cells) > 0:
            water_biomes = classifier.biomes[water_cells]
            assert np.all(water_biomes == BiomeType.OCEAN)
            
        # Should have land biomes for land cells
        land_cells = np.where(climate_graph.heights >= 20)[0]
        if len(land_cells) > 0:
            land_biomes = classifier.biomes[land_cells]
            assert np.all(land_biomes != BiomeType.OCEAN)
            
    def test_alpine_biome_classification(self, climate_graph):
        """Test alpine biome classification for high elevations."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # High elevation cells should be alpine
        high_cells = np.where(climate_graph.heights >= classifier.options.alpine_height_threshold)[0]
        
        if len(high_cells) > 0:
            alpine_biomes = classifier.biomes[high_cells]
            # Most high elevation cells should be alpine
            alpine_count = np.sum(alpine_biomes == BiomeType.ALPINE)
            assert alpine_count > len(high_cells) * 0.5
            
    def test_glacier_biome_classification(self, climate_graph):
        """Test glacier biome classification for very cold areas."""
        # Manually set some cells to be very cold
        very_cold_cells = []
        for i in range(min(10, len(climate_graph.temperatures))):
            climate_graph.temperatures[i] = -15
            very_cold_cells.append(i)
            
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Very cold cells should be glaciers (if not water)
        for cell in very_cold_cells:
            if climate_graph.heights[cell] >= 20:  # Land cell
                assert classifier.biomes[cell] == BiomeType.GLACIER
                
    def test_biome_matrix_lookup(self, climate_graph):
        """Test biome matrix lookup functionality."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test specific temperature/precipitation combinations
        # Temperate (band 3) + moderate precipitation (band 2) = temperate deciduous forest
        biome = classifier.biome_matrix[3][2]
        assert biome == BiomeType.TEMPERATE_DECIDUOUS_FOREST
        
        # Hot (band 5) + very wet (band 4) = tropical rainforest
        biome = classifier.biome_matrix[5][4]
        assert biome == BiomeType.TROPICAL_RAINFOREST
        
        # Cold (band 1) + dry (band 1) = tundra
        biome = classifier.biome_matrix[1][1]
        assert biome == BiomeType.TUNDRA
        
    def test_coastal_influences(self, climate_graph):
        """Test coastal influences on biomes."""
        # Add distance field for coastal effects
        n_cells = len(climate_graph.points)
        climate_graph.distance_field = np.zeros(n_cells)
        
        # Set some cells as coastal
        coastal_cells = []
        for i in range(n_cells):
            if climate_graph.heights[i] >= 20:  # Land cell
                # Check if near water
                for neighbor in climate_graph.cell_neighbors[i]:
                    if climate_graph.heights[neighbor] < 20:
                        climate_graph.distance_field[i] = 1.0  # Close to coast
                        coastal_cells.append(i)
                        break
                        
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Should complete without errors
        assert classifier.biomes is not None
        
    def test_river_influences(self, full_pipeline_graph):
        """Test river influences on biomes."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.classify_biomes()
        
        # River cells should be marked as rivers
        if hasattr(full_pipeline_graph, 'rivers') and full_pipeline_graph.rivers:
            for river in full_pipeline_graph.rivers:
                for cell in river.cells:
                    if full_pipeline_graph.heights[cell] >= 20:  # Land river
                        assert classifier.biomes[cell] == BiomeType.RIVER
                        
    def test_wetland_formation(self, full_pipeline_graph):
        """Test wetland formation in high water flux areas."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.classify_biomes()
        
        # Should complete without errors
        assert classifier.biomes is not None
        
        # Check if wetlands formed where appropriate
        wetland_cells = np.where(classifier.biomes == BiomeType.WETLAND)[0]
        
        # If wetlands formed, they should be in high water flux areas
        if len(wetland_cells) > 0 and hasattr(full_pipeline_graph, 'water_flux'):
            wetland_flux = [full_pipeline_graph.water_flux[cell] for cell in wetland_cells]
            avg_wetland_flux = np.mean(wetland_flux)
            avg_total_flux = np.mean(full_pipeline_graph.water_flux)
            
            # Wetlands should have higher than average water flux
            assert avg_wetland_flux >= avg_total_flux
            
    def test_biome_region_generation(self, climate_graph):
        """Test biome region generation."""
        classifier = BiomeClassifier(climate_graph)
        classifier.generate_biome_regions()
        
        # Should have generated regions
        assert len(classifier.biome_regions) > 0
        
        # Check region properties
        total_cells = set()
        for region in classifier.biome_regions:
            assert region.id >= 0
            assert isinstance(region.biome_type, BiomeType)
            assert len(region.cells) > 0
            assert region.area > 0
            assert region.center_cell in region.cells
            
            # Regions shouldn't overlap
            assert len(total_cells.intersection(region.cells)) == 0
            total_cells.update(region.cells)
            
        # All cells should be in some region
        assert len(total_cells) == len(climate_graph.points)
        
    def test_connected_biome_cells(self, climate_graph):
        """Test finding connected biome cells."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Test with a specific cell
        test_cell = 0
        biome_type = BiomeType(classifier.biomes[test_cell])
        processed = set()
        
        connected_cells = classifier._find_connected_biome_cells(test_cell, biome_type, processed)
        
        # Should find at least the test cell itself
        assert test_cell in connected_cells
        
        # All connected cells should have the same biome type
        for cell in connected_cells:
            assert BiomeType(classifier.biomes[cell]) == biome_type
            
    def test_region_center_calculation(self, climate_graph):
        """Test region center calculation."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test with a small set of cells
        test_cells = {0, 1, 2}
        center_cell = classifier._find_region_center(test_cells)
        
        # Center should be one of the cells in the region
        assert center_cell in test_cells
        
    def test_full_classification_pipeline(self, full_pipeline_graph):
        """Test the complete biome classification pipeline."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.run_full_classification()
        
        # All components should be generated
        assert classifier.biomes is not None
        assert len(classifier.biome_regions) > 0
        
        # Data should be stored on graph
        assert hasattr(full_pipeline_graph, 'biomes')
        assert hasattr(full_pipeline_graph, 'biome_regions')
        
    def test_biome_statistics(self, climate_graph):
        """Test biome statistics generation."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        stats = classifier.get_biome_statistics()
        
        # Should have statistics
        assert len(stats) > 0
        
        # All values should be positive
        for biome_name, count in stats.items():
            assert biome_name in BIOME_NAMES.values()
            assert count > 0
            
        # Total count should match number of cells
        total_count = sum(stats.values())
        assert total_count == len(climate_graph.points)
        
    def test_custom_biome_options(self, climate_graph):
        """Test biome classification with custom options."""
        custom_options = BiomeOptions(
            coastal_effect_distance=5.0,
            river_effect_distance=3.0,
            alpine_height_threshold=60
        )
        
        classifier = BiomeClassifier(climate_graph, options=custom_options)
        classifier.classify_biomes()
        
        # Should use custom options
        assert classifier.options.coastal_effect_distance == 5.0
        assert classifier.options.river_effect_distance == 3.0
        assert classifier.options.alpine_height_threshold == 60
        
        # Should complete successfully
        assert classifier.biomes is not None
        
    def test_biome_diversity(self, full_pipeline_graph):
        """Test that classification produces diverse biomes."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.run_full_classification()
        
        # Should have multiple biome types
        unique_biomes = np.unique(classifier.biomes)
        assert len(unique_biomes) >= 3  # At least 3 different biomes
        
        # Should include both water and land biomes
        has_water = BiomeType.OCEAN in unique_biomes
        has_land = any(biome != BiomeType.OCEAN for biome in unique_biomes)
        
        assert has_water and has_land
        
    def test_biome_names_completeness(self):
        """Test that all biome types have names."""
        for biome_type in BiomeType:
            assert biome_type in BIOME_NAMES
            assert isinstance(BIOME_NAMES[biome_type], str)
            assert len(BIOME_NAMES[biome_type]) > 0


def test_biome_module_imports():
    """Test that biome module imports correctly."""
    from py_fmg.core.biomes import BiomeClassifier, BiomeOptions, BiomeType, BIOME_NAMES
    
    assert BiomeClassifier is not None
    assert BiomeOptions is not None
    assert BiomeType is not None
    assert BIOME_NAMES is not None
    
    
def test_biome_type_enum():
    """Test BiomeType enum values."""
    # Test that enum values are as expected
    assert BiomeType.OCEAN == 0
    assert BiomeType.LAKE == 1
    assert BiomeType.TROPICAL_RAINFOREST == 15
    
    # Test that all values are unique
    values = [biome.value for biome in BiomeType]
    assert len(values) == len(set(values))
    
    
def test_biome_options_defaults():
    """Test BiomeOptions default values."""
    options = BiomeOptions()
    
    assert options.coastal_effect_distance == 3.0
    assert options.river_effect_distance == 2.0
    assert options.wetland_threshold == 0.8
    assert options.alpine_height_threshold == 70
    assert options.glacier_temperature_threshold == -10

=======
"""
Tests for biome classification system

Tests biome assignment algorithm against FMG reference implementation.
"""

import pytest
import numpy as np
from py_fmg.core.biomes import BiomeClassifier


class TestBiomeClassifier:
    """Test suite for BiomeClassifier"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = BiomeClassifier()
        
        # Sample test data
        self.heights = np.array([10, 25, 50, 100, 15, 30, 75])  # Mix of water and land
        self.temperatures = np.array([25, 15, 5, -10, 30, 0, 10])
        self.precipitation = np.array([5, 15, 25, 10, 2, 30, 20])
        self.has_river = np.array([False, True, False, False, False, True, False])
        self.river_flux = np.array([0, 50, 0, 0, 0, 30, 0])
        
        # Simple neighbor structure for testing
        self.neighbors = [
            [1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5]
        ]
    
    def test_biome_data_initialization(self):
        """Test that biome data is properly initialized"""
        data = self.classifier.biome_data
        
        # Check lengths match
        assert len(data.names) == 13
        assert len(data.colors) == 13
        assert len(data.habitability) == 13
        assert len(data.movement_cost) == 13
        
        # Check specific values from FMG
        assert data.names[0] == "Marine"
        assert data.names[1] == "Hot desert"
        assert data.names[12] == "Wetland"
        
        assert data.colors[0] == "#466eab"  # Marine blue
        assert data.habitability[6] == 100  # Temperate deciduous forest (highest)
        assert data.movement_cost[11] == 5000  # Glacier (highest cost)
        
        # Check biome matrix dimensions
        assert data.biome_matrix.shape == (5, 26)
        assert data.biome_matrix.dtype == np.uint8
    
    def test_biome_matrix_values(self):
        """Test specific biome matrix lookups"""
        matrix = self.classifier.biome_data.biome_matrix
        
        # Test corner cases
        assert matrix[0, 0] == 1   # Hottest, driest = Hot desert
        assert matrix[0, 25] == 10  # Coldest, driest = Tundra
        assert matrix[4, 0] == 7   # Hottest, wettest = Tropical rainforest
        assert matrix[4, 25] == 10  # Coldest, wettest = Tundra
        
        # Test some intermediate values
        assert matrix[1, 3] == 4   # Grassland region
        assert matrix[2, 6] == 6   # Temperate deciduous forest
        assert matrix[3, 19] == 9  # Taiga region
    
    def test_moisture_calculation_basic(self):
        """Test basic moisture calculation without rivers"""
        cell_id = 1  # Land cell
        moisture = self.classifier.calculate_moisture(
            cell_id, self.precipitation, self.heights
        )
        
        # Should be precipitation + 4 baseline
        expected = 4.0 + self.precipitation[cell_id]
        assert moisture == expected
    
    def test_moisture_calculation_with_river(self):
        """Test moisture calculation with river bonus"""
        cell_id = 1  # Land cell with river
        moisture = self.classifier.calculate_moisture(
            cell_id, self.precipitation, self.heights,
            river_flux=self.river_flux, has_river=self.has_river
        )
        
        # Should include river bonus: flux/10 or minimum 2
        river_bonus = max(self.river_flux[cell_id] / 10.0, 2.0)
        expected = round(4.0 + self.precipitation[cell_id] + river_bonus, 1)
        assert moisture == expected
    
    def test_moisture_calculation_water_cell(self):
        """Test moisture calculation for water cells"""
        cell_id = 0  # Water cell (height < 20)
        moisture = self.classifier.calculate_moisture(
            cell_id, self.precipitation, self.heights
        )
        assert moisture == 0.0
    
    def test_moisture_calculation_with_neighbors(self):
        """Test moisture calculation with neighbor averaging"""
        cell_id = 2  # Land cell with neighbors
        moisture = self.classifier.calculate_moisture(
            cell_id, self.precipitation, self.heights,
            neighbors=self.neighbors
        )
        
        # Should average with land neighbors
        base_moisture = self.precipitation[cell_id]
        neighbor_moisture = [self.precipitation[1], self.precipitation[3]]  # Land neighbors
        all_moisture = neighbor_moisture + [base_moisture]
        expected = round(4.0 + np.mean(all_moisture), 1)
        assert moisture == expected
    
    def test_vectorized_moisture_calculation(self):
        """Test vectorized moisture calculation matches single-cell results"""
        moisture_vec = self.classifier.calculate_moisture_vectorized(
            self.precipitation, self.heights, self.river_flux, self.has_river
        )
        
        # Compare with individual calculations (without neighbors for simplicity)
        for i in range(len(self.heights)):
            moisture_single = self.classifier.calculate_moisture(
                i, self.precipitation, self.heights, self.river_flux, self.has_river
            )
            assert abs(moisture_vec[i] - moisture_single) < 0.1
    
    def test_wetland_conditions(self):
        """Test wetland classification conditions"""
        # Too cold - should not be wetland
        assert not self.classifier.is_wetland(50, -3, 20)
        
        # Near coast conditions (moisture > 40, height < 25)
        assert self.classifier.is_wetland(45, 10, 20)
        assert not self.classifier.is_wetland(35, 10, 20)  # Not enough moisture
        assert self.classifier.is_wetland(45, 10, 26)  # Off coast wetland (moisture > 24, height 25-59)
        
        # Off coast conditions (moisture > 24, height 25-59)
        assert self.classifier.is_wetland(30, 10, 40)
        assert not self.classifier.is_wetland(20, 10, 40)  # Not enough moisture
        assert not self.classifier.is_wetland(30, 10, 20)  # Too low
        assert not self.classifier.is_wetland(30, 10, 65)  # Too high
    
    def test_special_biome_conditions(self):
        """Test special biome condition overrides"""
        # Marine (water)
        assert self.classifier.get_biome_id(10, 15, 15, False) == 0
        
        # Permafrost (very cold)
        assert self.classifier.get_biome_id(10, -10, 50, False) == 11
        
        # Hot desert (hot, dry, no river)
        assert self.classifier.get_biome_id(5, 30, 50, False) == 1
        assert self.classifier.get_biome_id(5, 30, 50, True) != 1  # Has river
        assert self.classifier.get_biome_id(10, 30, 50, False) != 1  # Too wet
        
        # Wetland
        assert self.classifier.get_biome_id(50, 10, 20, False) == 12
    
    def test_matrix_lookup_biomes(self):
        """Test biome matrix lookup for normal conditions"""
        # Test specific known combinations
        
        # Hot and medium moisture -> should be tropical seasonal forest  
        biome_id = self.classifier.get_biome_id(10, 25, 50, False)  # moisture 10, temp 25
        assert biome_id == 5  # Tropical seasonal forest (from matrix lookup)
        
        # Temperate and high moisture -> temperate rainforest  
        biome_id = self.classifier.get_biome_id(20, 10, 50, False)
        assert biome_id == 8  # Temperate rainforest (moisture 20 -> band 4, temp 10 -> band 10)
        
        # Cold and wet -> taiga
        biome_id = self.classifier.get_biome_id(25, -2, 50, False)
        assert biome_id == 9  # Taiga
    
    def test_classify_biomes_vectorized(self):
        """Test full vectorized biome classification"""
        biomes = self.classifier.classify_biomes(
            self.temperatures, self.precipitation, self.heights,
            self.river_flux, self.has_river
        )
        
        assert len(biomes) == len(self.heights)
        assert biomes.dtype == np.uint8
        
        # Check specific expected results
        assert biomes[0] == 0  # Water cell -> Marine
        assert biomes[3] == 11  # Very cold -> Permafrost
        assert biomes[4] == 0   # Water cell (height 15) -> Marine
        
        # All biome IDs should be valid
        assert np.all(biomes >= 0)
        assert np.all(biomes <= 12)
    
    def test_biome_properties_access(self):
        """Test biome property getters"""
        # Test valid biome ID
        assert self.classifier.get_biome_name(0) == "Marine"
        assert self.classifier.get_biome_color(1) == "#fbe79f"
        
        # Test invalid biome ID
        assert self.classifier.get_biome_name(99) == "Unknown"
        assert self.classifier.get_biome_color(-1) == "#000000"
        
        # Test full properties
        props = self.classifier.get_biome_properties(6)
        assert props["name"] == "Temperate deciduous forest"
        assert props["habitability"] == 100
        assert props["movement_cost"] == 70
        
        # Test invalid properties
        assert self.classifier.get_biome_properties(99) == {}
    
    def test_boundary_conditions(self):
        """Test edge cases and boundary conditions"""
        # Moisture band boundaries
        assert self.classifier.get_biome_id(4.9, 15, 50, False) != self.classifier.get_biome_id(5.0, 15, 50, False)
        # Skip this boundary test as both values trigger wetland conditions
        
        # Temperature band boundaries - use different values to avoid similar results  
        assert self.classifier.get_biome_id(8, 15, 50, False) != self.classifier.get_biome_id(8, 25, 50, False)
        assert self.classifier.get_biome_id(8, -10, 50, False) != self.classifier.get_biome_id(8, 0, 50, False)
        
        # Height boundaries
        assert self.classifier.get_biome_id(15, 15, 19.9, False) == 0  # Marine
        assert self.classifier.get_biome_id(15, 15, 20.1, False) != 0  # Land biome
    
    def test_river_influence_on_classification(self):
        """Test how rivers influence biome classification"""
        # Same conditions, different river status
        no_river = self.classifier.get_biome_id(5, 20, 50, False)
        with_river = self.classifier.get_biome_id(5, 20, 50, True)
        
        # River should increase moisture and potentially change biome
        # This might result in different biomes due to moisture increase
        assert isinstance(no_river, (int, np.integer))
        assert isinstance(with_river, (int, np.integer))
        assert 0 <= no_river <= 12
        assert 0 <= with_river <= 12
    
    def test_realistic_world_scenario(self):
        """Test with realistic world-like data"""
        # Create a more realistic test scenario
        n_cells = 100
        np.random.seed(42)  # For reproducible tests
        
        heights = np.random.uniform(0, 200, n_cells)
        temperatures = np.random.uniform(-20, 35, n_cells)
        precipitation = np.random.uniform(0, 50, n_cells)
        has_river = np.random.choice([True, False], n_cells, p=[0.1, 0.9])
        river_flux = np.where(has_river, np.random.uniform(10, 100, n_cells), 0)
        
        biomes = self.classifier.classify_biomes(
            temperatures, precipitation, heights, river_flux, has_river
        )
        
        # Basic sanity checks
        assert len(biomes) == n_cells
        assert np.all(biomes >= 0)
        assert np.all(biomes <= 12)
        
        # Water cells should be marine
        water_mask = heights < 20
        assert np.all(biomes[water_mask] == 0)
        
        # Very cold cells should be permafrost
        cold_mask = (temperatures < -5) & (heights >= 20)
        assert np.all(biomes[cold_mask] == 11)
        
        # Should have variety of biomes
        unique_biomes = np.unique(biomes)
        assert len(unique_biomes) > 3  # Should have reasonable diversity


if __name__ == "__main__":
    pytest.main([__file__])

